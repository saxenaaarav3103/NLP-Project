"""Fast checks of the data pipeline and metrics (no model download needed).

    python -m pytest -q
"""
import numpy as np
import pandas as pd

from src.emotion.config import LABELS
from src.emotion.data import add_context, clean_text, load_split, model_input
from src.emotion.evaluate import compute_metrics


def test_clean_text_normalises_but_keeps_emotional_signal():
    assert clean_text("I’m  so… MAD!!") == "I'm so... MAD!!"  # case and "!!" are kept
    assert clean_text("  hello \n world ") == "hello world"


def test_splits_load_with_valid_labels():
    for split, n in [("train", 9989), ("dev", 1109), ("test", 2610)]:
        df = load_split(split)
        assert len(df) == n
        assert df["label"].between(0, len(LABELS) - 1).all()
        assert (df["emotion"] == df["label"].map(dict(enumerate(LABELS)))).all()


def test_context_uses_only_previous_turns_of_same_dialogue():
    df = pd.DataFrame({"dialogue_id": [0, 0, 0, 1, 1], "utterance_id": [0, 1, 2, 0, 1],
                       "speaker": list("ABABA"), "text": ["a", "b", "c", "d", "e"],
                       "emotion": ["neutral"] * 5, "label": [0] * 5})
    ctx = add_context(df, n_turns=2)["context"].tolist()
    assert ctx == ["", "a", "a | b", "", "d"]  # no future turns, no leakage across dialogues


def test_model_input_keeps_target_and_trims_old_context():
    target, context = model_input("What?", " ".join(f"w{i}" for i in range(100)), max_context_words=10)
    assert target == "What?"
    assert context.split() == [f"w{i}" for i in range(90, 100)]  # most recent words kept
    assert model_input("Hi")[1] is None


def test_metrics_on_known_example():
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 0, 1, 0, 2, 1])
    m = compute_metrics(y_true, y_pred)
    assert abs(m["accuracy"] - 4 / 6) < 1e-9
    assert m["per_class"]["neutral"]["recall"] == 1.0
    assert abs(m["per_class"]["neutral"]["precision"] - 2 / 3) < 1e-9
    # classes absent from both y_true and y_pred count as F1 = 0 in the macro average
    assert m["macro_f1"] < m["weighted_f1"]
