"""Inference wrappers used by the demo app and the notebook.

    from src.emotion.predict import EmotionClassifier
    clf = EmotionClassifier()                       # loads models/roberta-meld-context
    clf.predict("I got the job!! I'm so excited!")  # {'joy': 0.97, 'surprise': ..., ...}
    clf.predict("What?", context=["Ross and Rachel broke up."])
    clf.predict_conversation(["I failed the exam again.", "Oh no, I'm so sorry."])
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .config import BASELINE_PATH, DEFAULT_TRANSFORMER_DIR, LABELS
from .data import clean_text, model_input


class EmotionClassifier:
    """Fine-tuned RoBERTa that reads the target utterance plus the previous turns."""

    def __init__(self, model_dir: str | Path = DEFAULT_TRANSFORMER_DIR, device: str = "cpu"):
        model_dir = Path(model_dir)
        if not (model_dir / "config.json").exists():
            raise FileNotFoundError(
                f"No fine-tuned model in {model_dir}. Train one with "
                "`python -m src.emotion.train_transformer` (see README).")
        cfg_path = model_dir / "training_config.json"
        cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
        self.context_turns = int(cfg.get("context_turns", 2))
        self.max_length = int(cfg.get("max_length", 128))
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        # Weights are stored in float16 to keep the repo small; compute in float32.
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).float()
        self.model.to(device).eval()
        self.device = device
        self.labels = [self.model.config.id2label[i] for i in range(self.model.config.num_labels)]
        assert self.labels == LABELS, self.labels

    def _context_string(self, context) -> str:
        if not context or self.context_turns == 0:
            return ""
        if isinstance(context, str):
            context = [context]
        turns = [clean_text(t) for t in context if str(t).strip()][-self.context_turns:]
        return " | ".join(turns)

    @torch.no_grad()
    def predict_proba(self, texts: list[str], contexts: list | None = None, batch_size: int = 32) -> np.ndarray:
        contexts = contexts or [None] * len(texts)
        pairs = [model_input(clean_text(t), self._context_string(c)) for t, c in zip(texts, contexts)]
        out = []
        for s in range(0, len(pairs), batch_size):
            chunk = pairs[s:s + batch_size]
            firsts = [a for a, _ in chunk]
            seconds = [b for _, b in chunk]
            if all(b is None for b in seconds):
                enc = self.tokenizer(firsts, truncation=True, max_length=self.max_length,
                                     padding=True, return_tensors="pt")
            else:  # the tokenizer needs a string for every pair; empty context -> single segment
                encs = [self.tokenizer(a, b, truncation="only_second" if b else True,
                                       max_length=self.max_length) for a, b in chunk]
                enc = self.tokenizer.pad({"input_ids": [e["input_ids"] for e in encs]}, return_tensors="pt")
            logits = self.model(input_ids=enc["input_ids"].to(self.device),
                                attention_mask=enc["attention_mask"].to(self.device)).logits
            out.append(torch.softmax(logits.float(), -1).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, len(self.labels)))

    def predict(self, text: str, context=None) -> dict[str, float]:
        probs = self.predict_proba([text], [context])[0]
        return {label: float(p) for label, p in sorted(zip(self.labels, probs), key=lambda x: -x[1])}

    def predict_conversation(self, turns: list[str]) -> list[dict]:
        """Label every turn of a conversation, each using only the turns before it."""
        turns = [t for t in turns if t.strip()]
        contexts = [turns[max(0, i - self.context_turns):i] for i in range(len(turns))]
        probs = self.predict_proba(turns, contexts)
        return [{"text": t, "emotion": self.labels[int(p.argmax())], "confidence": float(p.max())}
                for t, p in zip(turns, probs)]


class BaselineClassifier:
    """TF-IDF + logistic regression (utterance only) -- for side-by-side comparison."""

    def __init__(self, path: str | Path = BASELINE_PATH):
        try:
            self.pipeline = joblib.load(path)
        except Exception:  # missing file or pickle from another scikit-learn version
            from .baseline import build_pipeline
            from .data import load_split
            print("Re-training the TF-IDF baseline (about 30 s)...")
            train = load_split("train")
            self.pipeline = build_pipeline(C=1.0).fit(train["text"], train["label"])
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(self.pipeline, path)

    def predict(self, text: str) -> dict[str, float]:
        probs = self.pipeline.predict_proba([clean_text(text)])[0]
        classes = [LABELS[int(c)] for c in self.pipeline.classes_]
        return {label: float(p) for label, p in sorted(zip(classes, probs), key=lambda x: -x[1])}
