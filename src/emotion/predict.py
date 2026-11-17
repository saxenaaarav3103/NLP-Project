"""Inference wrappers used by the demo app and the notebook.

    from src.emotion.predict import EmotionClassifier
    clf = EmotionClassifier()                       # loads models/roberta-meld-context
    clf.predict("I got the job!! I'm so excited!")  # {'joy': 0.97, 'surprise': ..., ...}
    clf.predict("What?", context=["Ross and Rachel broke up."])
    clf.predict_conversation(["I failed the exam again.", "Oh no, I'm so sorry."])
    with_ctx, without_ctx = clf.predict_pair("What?", ["Ross and Rachel broke up."])  # one forward pass
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .config import BASELINE_PATH, DEFAULT_TRANSFORMER_DIR, LABELS
from .data import clean_text, encode_pair, model_input
from .runtime import autocast_ctx, cpu_supports_bf16


class EmotionClassifier:
    """Fine-tuned RoBERTa that reads the target utterance plus the previous turns."""

    def __init__(self, model_dir: str | Path = DEFAULT_TRANSFORMER_DIR, device: str = "cpu",
                 bf16: bool | str = False):
        """``bf16`` (opt-in, CPU only): run the forward pass under bfloat16 autocast. About
        2-2.7x faster for bulk prediction on a 4-core Xeon with AMX (no gain for a single
        message), but probabilities move by up to ~0.013 and 3 of the 2,610 test predictions
        change, so it is off by default (the reported numbers are float32). ``"auto"`` = on only if the CPU has
        native bfloat16 support; ``True`` on a CPU without it falls back to float32."""
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
        on_cpu = torch.device(device).type == "cpu"
        supported = on_cpu and cpu_supports_bf16()
        if bf16 is True and not supported:
            warnings.warn("bfloat16 autocast needs a CPU with AMX / AVX512-BF16; using float32.")
        self.bf16 = bool(bf16) and supported
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
        if contexts is None:
            contexts = [None] * len(texts)
        if len(contexts) != len(texts):
            raise ValueError(f"got {len(texts)} texts but {len(contexts)} contexts")
        pairs = [model_input(clean_text(t), self._context_string(c)) for t, c in zip(texts, contexts)]
        ids = [encode_pair(self.tokenizer, a, b, self.max_length) for a, b in pairs]
        # Process in order of length so each batch needs little padding, then restore the order.
        order = sorted(range(len(ids)), key=lambda i: len(ids[i]))
        probs = np.zeros((len(ids), len(self.labels)), dtype=np.float32)
        for s in range(0, len(order), batch_size):
            chunk = order[s:s + batch_size]
            enc = self.tokenizer.pad({"input_ids": [ids[i] for i in chunk]}, return_tensors="pt")
            with autocast_ctx(self.bf16):
                logits = self.model(input_ids=enc["input_ids"].to(self.device),
                                    attention_mask=enc["attention_mask"].to(self.device)).logits
            probs[chunk] = torch.softmax(logits.float(), -1).cpu().numpy()
        return probs

    def _as_dict(self, probs: np.ndarray) -> dict[str, float]:
        return {label: float(p) for label, p in sorted(zip(self.labels, probs), key=lambda x: -x[1])}

    def predict(self, text: str, context=None) -> dict[str, float]:
        return self._as_dict(self.predict_proba([text], [context])[0])

    def predict_pair(self, text: str, context=None) -> tuple[dict[str, float], dict[str, float]]:
        """``(predict(text, context), predict(text))`` computed in one batched forward pass,
        about as fast as a single ``predict`` call (for showing what the context changed)."""
        with_ctx, without_ctx = self.predict_proba([text, text], [context, None])
        return self._as_dict(with_ctx), self._as_dict(without_ctx)

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
