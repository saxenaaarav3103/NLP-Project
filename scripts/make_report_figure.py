"""Build the figure used in the 2-page report from the saved results.

    python scripts/make_report_figure.py   ->  results/figures/report_figure.png

Left: per-class test F1 of the baseline and both RoBERTa variants (seed 42).
Right: row-normalised confusion matrix of the final model (RoBERTa + context).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.emotion.config import FIGURES_DIR, LABELS, RESULTS_DIR  # noqa: E402
from src.emotion.data import load_split  # noqa: E402


def main():
    plt.rcParams.update({"font.size": 7.5, "axes.titlesize": 8.5, "axes.spines.top": False,
                         "axes.spines.right": False, "font.family": "DejaVu Sans"})
    runs = [("tfidf_logreg", "TF-IDF + LogReg", "#c7c7c7"),
            ("roberta_utterance", "RoBERTa (utterance only)", "#9ecae1"),
            ("roberta_context", "RoBERTa + context (final)", "#2171b5")]
    metrics = {k: json.loads((RESULTS_DIR / f"{k}_metrics.json").read_text()) for k, _, _ in runs}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.1, 2.55), gridspec_kw={"width_ratios": [1.55, 1]})
    x = np.arange(len(LABELS))
    w = 0.27
    for i, (key, label, color) in enumerate(runs):
        f1 = [metrics[key]["per_class"][c]["f1"] for c in LABELS]
        ax1.bar(x + (i - 1) * w, f1, w, label=label, color=color)
    support = [metrics["roberta_context"]["per_class"][c]["support"] for c in LABELS]
    ax1.set_xticks(x, [f"{c}\n(n={s})" for c, s in zip(LABELS, support)])
    ax1.set_ylabel("F1 (test)")
    ax1.set_ylim(0, 0.85)
    ax1.set_title("(a) Per-class F1 on the test set")
    ax1.legend(frameon=False, fontsize=6.8, loc="upper right")
    ax1.grid(axis="y", alpha=0.3)

    y_true = load_split("test")["label"].to_numpy()
    y_pred = np.load(RESULTS_DIR / "roberta_context_test_probs.npy").argmax(1)
    cm = confusion_matrix(y_true, y_pred, labels=range(len(LABELS)))
    cmn = cm / cm.sum(1, keepdims=True)
    ax2.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    short = ["neu", "joy", "sur", "ang", "sad", "dis", "fea"]
    ax2.set_xticks(range(7), short)
    ax2.set_yticks(range(7), short)
    ax2.set_xlabel("predicted")
    ax2.set_ylabel("true")
    ax2.set_title("(b) Confusion matrix, final model (row %)")
    for i in range(7):
        for j in range(7):
            ax2.text(j, i, f"{cmn[i, j] * 100:.0f}", ha="center", va="center", fontsize=6.3,
                     color="white" if cmn[i, j] > 0.5 else "black")
    for spine in ax2.spines.values():
        spine.set_visible(False)
    fig.tight_layout(w_pad=1.5)
    out = FIGURES_DIR / "report_figure.png"
    fig.savefig(out, dpi=300)
    print("saved", out)


if __name__ == "__main__":
    main()
