"""Export a trained run to the compact model shipped in the repository, then re-check it.

    python scripts/export_model.py runs/roberta_context models/roberta-meld-context

* weights are stored in float16 (half the size; loaded back as float32 for inference);
* split into <100 MB shards so they fit GitHub's per-file limit;
* the exported model is evaluated on the test set again and the metrics file
  ``results/<run>_metrics.json`` is updated with the numbers of the *shipped* model, so
  the report, notebook and demo all describe exactly the same artefact.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.emotion.config import RESULTS_DIR  # noqa: E402
from src.emotion.data import add_context, load_split  # noqa: E402
from src.emotion.evaluate import compute_metrics  # noqa: E402
from src.emotion.predict import EmotionClassifier  # noqa: E402


def main(run_dir: Path, out_dir: Path):
    model = AutoModelForSequenceClassification.from_pretrained(run_dir).float()
    if out_dir.exists():
        shutil.rmtree(out_dir)
    model.half().save_pretrained(out_dir, max_shard_size="95MB")
    AutoTokenizer.from_pretrained(run_dir).save_pretrained(out_dir)
    shutil.copy(run_dir / "training_config.json", out_dir / "training_config.json")
    sizes = {p.name: f"{p.stat().st_size / 1e6:.1f} MB" for p in sorted(out_dir.iterdir())}
    print(json.dumps(sizes, indent=2))

    clf = EmotionClassifier(out_dir)
    test = add_context(load_split("test"), clf.context_turns)
    contexts = [c.split(" | ") if c else None for c in test["context"]] if clf.context_turns else None
    probs = clf.predict_proba(list(test["text"]), contexts)
    metrics = compute_metrics(test["label"].to_numpy(), probs.argmax(1))

    name = run_dir.name
    path = RESULTS_DIR / f"{name}_metrics.json"
    old = json.loads(path.read_text()) if path.exists() else {}
    print(f"training-run test weighted-F1 {old.get('weighted_f1', float('nan')):.4f} -> "
          f"shipped model {metrics['weighted_f1']:.4f}; accuracy {old.get('accuracy', float('nan')):.4f} -> "
          f"{metrics['accuracy']:.4f}")
    keep = {k: old[k] for k in ("history", "args", "best_dev_weighted_f1", "train_minutes") if k in old}
    metrics.update(keep, evaluated_model=str(out_dir.resolve().relative_to(ROOT)))
    path.write_text(json.dumps(metrics, indent=2))
    np.save(RESULTS_DIR / f"{name}_test_probs.npy", probs)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
