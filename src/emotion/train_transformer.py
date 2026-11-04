"""Fine-tune a RoBERTa encoder for 7-way emotion classification on MELD.

Run (defaults reproduce the model shipped in ``models/``):

    python -m src.emotion.train_transformer --base-model roberta-base --context-turns 2

``--base-model`` can be a Hugging Face Hub id (``roberta-base``) or a local directory.
``--context-turns 0`` trains the utterance-only ablation.

Design choices (see the report for the reasoning):
* Input = target utterance (segment A) + up to N previous turns of the dialogue
  (segment B). Emotion in conversation is often only readable from context
  ("What?" after bad news = surprise; after an insult = anger).
* Loss = cross-entropy with *softened* inverse-frequency class weights
  (w_c ~ 1/sqrt(freq_c)). Full inverse weights over-predict the rare classes;
  no weights ignore them. The square root is a common middle ground.
* Model selection: the epoch with the best dev weighted-F1 is kept (early stopping);
  the test set is used exactly once, at the end.
* CPU friendly: dynamic padding + length-bucketed batches, and bfloat16 autocast when
  the CPU supports it.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from .config import ID2LABEL, LABEL2ID, LABELS, RESULTS_DIR, ROOT, SEED
from .data import add_context, load_all, model_input
from .evaluate import compute_metrics, fig_path, plot_confusion, report, save_metrics


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def encode(df, tokenizer, context_turns: int, max_length: int):
    """Tokenise a split into a list of feature dicts (unpadded)."""
    if context_turns > 0:
        df = add_context(df, context_turns)
        pairs = [model_input(t, c) for t, c in zip(df["text"], df["context"])]
    else:
        pairs = [model_input(t) for t in df["text"]]
    features = []
    for (a, b), label in zip(pairs, df["label"]):
        enc = tokenizer(a, b, truncation="only_second" if b else True, max_length=max_length)
        features.append({"input_ids": enc["input_ids"], "label": int(label)})
    return features


def collate(batch, pad_id: int):
    """Dynamic padding: pad only up to the longest sequence in the batch."""
    width = max(len(f["input_ids"]) for f in batch)
    ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(batch), width), dtype=torch.long)
    for i, f in enumerate(batch):
        n = len(f["input_ids"])
        ids[i, :n] = torch.tensor(f["input_ids"])
        mask[i, :n] = 1
    labels = torch.tensor([f["label"] for f in batch])
    return {"input_ids": ids, "attention_mask": mask, "labels": labels}


def bucketed_batches(features, batch_size: int, rng: random.Random):
    """Shuffle, then sort within chunks of 50 batches by length so each batch holds
    sequences of similar length (much less padding -> faster on CPU)."""
    idx = list(range(len(features)))
    rng.shuffle(idx)
    chunk = batch_size * 50
    batches = []
    for s in range(0, len(idx), chunk):
        part = sorted(idx[s:s + chunk], key=lambda i: len(features[i]["input_ids"]))
        batches += [part[j:j + batch_size] for j in range(0, len(part), batch_size)]
    rng.shuffle(batches)
    return batches


def class_weights(labels, power: float = 0.5) -> torch.Tensor:
    counts = np.bincount(labels, minlength=len(LABELS)).astype(float).clip(min=1)
    w = (counts.sum() / counts) ** power
    return torch.tensor(w / w.mean(), dtype=torch.float)


def cpu_supports_bf16() -> bool:
    """True on CPUs with native bfloat16 matrix units (Intel AMX / AVX512-BF16)."""
    try:
        with open("/proc/cpuinfo") as f:
            flags = f.read()
    except OSError:  # macOS / Windows: just use float32
        return False
    return "amx_bf16" in flags or "avx512_bf16" in flags


def autocast_ctx(enabled: bool):
    return torch.autocast("cpu", dtype=torch.bfloat16, enabled=enabled)


@torch.no_grad()
def predict_features(model, features, pad_id: int, batch_size: int = 64, bf16: bool = False):
    """Return class probabilities (N x 7) for already-encoded features, in order."""
    model.eval()
    order = sorted(range(len(features)), key=lambda i: len(features[i]["input_ids"]))
    probs = np.zeros((len(features), len(LABELS)), dtype=np.float32)
    for s in range(0, len(order), batch_size):
        chunk = order[s:s + batch_size]
        batch = collate([features[i] for i in chunk], pad_id)
        with autocast_ctx(bf16):
            logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
        probs[chunk] = torch.softmax(logits.float(), dim=-1).numpy()
    return probs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-model", default="roberta-base")
    ap.add_argument("--context-turns", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup", type=float, default=0.06)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--class-weight-power", type=float, default=0.5,
                    help="0 = unweighted CE, 0.5 = 1/sqrt(freq), 1 = inverse frequency")
    ap.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=None,
                    help="bfloat16 autocast (default: on if the CPU supports it)")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--limit", type=int, default=None, help="debug: use only N train examples")
    ap.add_argument("--name", default=None, help="run name (default derived from settings)")
    ap.add_argument("--output-dir", type=Path, default=None)
    args = ap.parse_args()

    name = args.name or ("roberta_context" if args.context_turns else "roberta_utterance")
    out_dir = args.output_dir or ROOT / "runs" / name
    if args.threads:
        torch.set_num_threads(args.threads)
    bf16 = args.bf16
    if bf16 is None:
        bf16 = cpu_supports_bf16()
    set_seed(args.seed)
    print(f"run={name} base={args.base_model} context_turns={args.context_turns} bf16={bf16} "
          f"threads={torch.get_num_threads()}")

    splits = load_all()
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    feats = {s: encode(df, tokenizer, args.context_turns, args.max_length) for s, df in splits.items()}
    if args.limit:
        feats["train"] = feats["train"][:args.limit]
    for s, f in feats.items():
        lens = [len(x["input_ids"]) for x in f]
        print(f"  {s}: {len(f)} examples, mean {np.mean(lens):.1f} tokens, max {max(lens)}")

    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model, num_labels=len(LABELS), id2label=ID2LABEL, label2id=LABEL2ID)
    pad_id = tokenizer.pad_token_id

    weights = class_weights([f["label"] for f in feats["train"]], args.class_weight_power)
    print("  class weights:", {l: round(float(w), 2) for l, w in zip(LABELS, weights)})
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)

    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
    groups = [
        {"params": [p for n, p in model.named_parameters() if not any(k in n for k in no_decay)],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(k in n for k in no_decay)],
         "weight_decay": 0.0},
    ]
    try:  # the fused kernel is ~1.5x faster on CPU; fall back on older PyTorch
        optimizer = torch.optim.AdamW(groups, lr=args.lr, fused=True)
    except (RuntimeError, TypeError):
        optimizer = torch.optim.AdamW(groups, lr=args.lr)
    steps_per_epoch = math.ceil(len(feats["train"]) / args.batch_size)
    total = steps_per_epoch * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(args.warmup * total), total)

    rng = random.Random(args.seed)
    history, best = [], None
    dev_labels = np.array([f["label"] for f in feats["dev"]])
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running, n_seen = 0.0, 0
        for step, idx in enumerate(bucketed_batches(feats["train"], args.batch_size, rng), 1):
            batch = collate([feats["train"][i] for i in idx], pad_id)
            with autocast_ctx(bf16):
                logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
            loss = loss_fn(logits.float(), batch["labels"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            running += loss.item() * len(idx)
            n_seen += len(idx)
            if step % 100 == 0 or step == steps_per_epoch:
                print(f"  epoch {epoch} step {step}/{steps_per_epoch} loss {running / n_seen:.4f} "
                      f"[{(time.time() - t0) / 60:.1f} min]", flush=True)
        dev_pred = predict_features(model, feats["dev"], pad_id, bf16=bf16).argmax(1)
        m = compute_metrics(dev_labels, dev_pred)
        history.append({"epoch": epoch, "train_loss": running / n_seen, "dev_accuracy": m["accuracy"],
                        "dev_weighted_f1": m["weighted_f1"], "dev_macro_f1": m["macro_f1"],
                        "minutes": (time.time() - t0) / 60})
        print(f"  == epoch {epoch}: dev acc {m['accuracy']:.4f} weighted-F1 {m['weighted_f1']:.4f} "
              f"macro-F1 {m['macro_f1']:.4f}", flush=True)
        if best is None or m["weighted_f1"] > best:
            best = m["weighted_f1"]
            out_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(out_dir)
            tokenizer.save_pretrained(out_dir)
            print(f"     saved best model -> {out_dir}")

    # Final evaluation of the best checkpoint on the held-out test set (float32, so the
    # numbers do not depend on the CPU's bfloat16 support).
    model = AutoModelForSequenceClassification.from_pretrained(out_dir)
    test_probs = predict_features(model, feats["test"], pad_id, bf16=False)
    test_labels = np.array([f["label"] for f in feats["test"]])
    test_pred = test_probs.argmax(1)
    metrics = compute_metrics(test_labels, test_pred)
    metrics.update({"history": history, "args": {k: (str(v) if isinstance(v, Path) else v)
                                                 for k, v in vars(args).items()},
                    "best_dev_weighted_f1": best, "train_minutes": (time.time() - t0) / 60})
    save_metrics(name, metrics)
    np.save(RESULTS_DIR / f"{name}_test_probs.npy", test_probs)
    plot_confusion(test_labels, test_pred, f"RoBERTa {'+ context' if args.context_turns else '(utterance only)'} (test)",
                   fig_path(f"cm_{name}.png"))
    (out_dir / "training_config.json").write_text(json.dumps(
        {"context_turns": args.context_turns, "max_length": args.max_length, "labels": LABELS,
         "base_model": args.base_model}, indent=2))
    print(report(test_labels, test_pred))
    print(f"TEST: acc {metrics['accuracy']:.4f} weighted-F1 {metrics['weighted_f1']:.4f} "
          f"macro-F1 {metrics['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
