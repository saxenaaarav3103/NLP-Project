"""Classical baseline: TF-IDF features + logistic regression.

Run:  python -m src.emotion.baseline

Word uni/bi-grams capture emotional words ("love", "oh my god"); character 2-5-grams
capture spelling and punctuation cues ("!!", "?!", "sooo") and are robust to typos.
`class_weight="balanced"` up-weights rare emotions so the model does not just predict
"neutral". The regularisation strength C is tuned on the dev set (macro-F1).
"""
from __future__ import annotations

import argparse

import joblib
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from .config import BASELINE_PATH, LABELS, SEED
from .data import load_all
from .evaluate import compute_metrics, fig_path, macro_f1, plot_confusion, report, save_metrics


def build_pipeline(C: float = 1.0) -> Pipeline:
    features = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, lowercase=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2,
                                 sublinear_tf=True, lowercase=False)),
    ])
    clf = LogisticRegression(C=C, class_weight="balanced", max_iter=3000, random_state=SEED)
    return Pipeline([("features", features), ("clf", clf)])


def tune_and_train(train, dev, grid=(0.1, 0.3, 1.0, 3.0, 10.0), verbose=True):
    """Pick C on dev macro-F1, then return the model fitted on the train split."""
    best = None
    for C in grid:
        model = build_pipeline(C).fit(train["text"], train["label"])
        score = macro_f1(dev["label"], model.predict(dev["text"]))
        if verbose:
            print(f"  C={C:<5} dev macro-F1={score:.4f}")
        if best is None or score > best[0]:
            best = (score, C, model)
    if verbose:
        print(f"  -> best C={best[1]}")
    return best[2], best[1]


def top_features(model: Pipeline, k: int = 8) -> dict[str, list[str]]:
    """Most positive word-level features per class -- what the baseline 'looks at'."""
    names = model.named_steps["features"].get_feature_names_out()
    coef = model.named_steps["clf"].coef_
    out = {}
    for i, label in enumerate(LABELS):
        order = np.argsort(coef[i])[::-1]
        words = [names[j].split("__", 1)[1] for j in order if names[j].startswith("word__")][:k]
        out[label] = words
    return out


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    splits = load_all()
    train, dev, test = splits["train"], splits["dev"], splits["test"]

    majority = DummyClassifier(strategy="most_frequent").fit(train["text"], train["label"])
    m_major = compute_metrics(test["label"], majority.predict(test["text"]))
    save_metrics("majority", m_major)
    print(f"Majority class: acc={m_major['accuracy']:.3f} weighted-F1={m_major['weighted_f1']:.3f}")

    print("Tuning TF-IDF + LogisticRegression on dev:")
    model, C = tune_and_train(train, dev)
    pred = model.predict(test["text"])
    metrics = compute_metrics(test["label"], pred)
    metrics["C"] = C
    save_metrics("tfidf_logreg", metrics)
    plot_confusion(test["label"], pred, "TF-IDF + LogReg (test)", fig_path("cm_tfidf_logreg.png"))
    print(report(test["label"], pred))

    BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, BASELINE_PATH)
    print(f"Saved baseline to {BASELINE_PATH}")
    for label, words in top_features(model).items():
        print(f"  {label:9s}: {', '.join(words)}")


if __name__ == "__main__":
    main()
