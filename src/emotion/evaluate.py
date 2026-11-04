"""Metrics, reports and plots shared by every model."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_recall_fscore_support)

from .config import FIGURES_DIR, LABELS, RESULTS_DIR


def compute_metrics(y_true, y_pred) -> dict:
    """Accuracy plus precision/recall/F1 (weighted and macro) and per-class scores.

    * weighted-F1 is the standard headline metric on MELD (each class counts by its size);
    * macro-F1 treats all seven emotions equally, so it exposes how well the rare
      classes (fear, disgust) are handled -- accuracy alone hides this, because a model
      that always says "neutral" already scores ~48 % accuracy.
    """
    labels = list(range(len(LABELS)))
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    pw, rw, fw, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels,
                                                    average="weighted", zero_division=0)
    pm, rm, fm, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels,
                                                    average="macro", zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "weighted_precision": float(pw), "weighted_recall": float(rw), "weighted_f1": float(fw),
        "macro_precision": float(pm), "macro_recall": float(rm), "macro_f1": float(fm),
        "per_class": {LABELS[i]: {"precision": float(p[i]), "recall": float(r[i]),
                                  "f1": float(f[i]), "support": int(s[i])} for i in labels},
    }


def report(y_true, y_pred) -> str:
    return classification_report(y_true, y_pred, labels=list(range(len(LABELS))),
                                 target_names=LABELS, digits=3, zero_division=0)


def save_metrics(name: str, metrics: dict, results_dir: Path = RESULTS_DIR) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / f"{name}_metrics.json"
    path.write_text(json.dumps(metrics, indent=2))
    return path


def plot_confusion(y_true, y_pred, title: str, path: Path | None = None, ax=None):
    """Row-normalised confusion matrix (each row = recall distribution of a true class)."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(LABELS))))
    cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(6.2, 5.2))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(LABELS)), LABELS, rotation=45, ha="right")
    ax.set_yticks(range(len(LABELS)), LABELS)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            v = cm_norm[i, j]
            ax.text(j, i, f"{v:.2f}\n({cm[i, j]})", ha="center", va="center", fontsize=7,
                    color="white" if v > 0.5 else "black")
    if own:
        fig.colorbar(im, ax=ax, fraction=0.046)
        fig.tight_layout()
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, dpi=150)
        plt.close(fig)
    return cm


def summary_row(name: str, m: dict) -> str:
    return (f"| {name} | {m['accuracy']:.3f} | {m['weighted_precision']:.3f} | "
            f"{m['weighted_recall']:.3f} | {m['weighted_f1']:.3f} | {m['macro_f1']:.3f} |")


def fig_path(name: str) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    return FIGURES_DIR / name


def macro_f1(y_true, y_pred) -> float:
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def as_array(x) -> np.ndarray:
    return np.asarray(x)
