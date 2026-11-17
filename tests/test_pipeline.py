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


# ---------------------------------------------------------------- regression tests

import builtins  # noqa: E402

import pytest  # noqa: E402

from src.emotion.config import DEFAULT_TRANSFORMER_DIR  # noqa: E402
from src.emotion.data import encode_pair  # noqa: E402
from src.emotion.runtime import cpu_supports_bf16  # noqa: E402

needs_model = pytest.mark.skipif(not (DEFAULT_TRANSFORMER_DIR / "config.json").exists(),
                                 reason="fine-tuned model not available")


def test_context_is_correct_when_rows_are_unsorted():
    # interleaved dialogues and reversed turn order: context must still come from the
    # previous turns of the *same* dialogue, written back to the right row
    df = pd.DataFrame({"dialogue_id": [1, 0, 1, 0, 0], "utterance_id": [1, 2, 0, 0, 1],
                       "text": ["e", "c", "d", "a", "b"]})
    out = add_context(df, n_turns=2)
    assert out["context"].tolist() == ["d", "a | b", "", "", "a"]
    assert out["text"].tolist() == df["text"].tolist()  # row order unchanged


@pytest.fixture(scope="module")
def tokenizer():
    if not (DEFAULT_TRANSFORMER_DIR / "tokenizer.json").exists():
        pytest.skip("tokenizer not available")
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(DEFAULT_TRANSFORMER_DIR)


def test_encode_pair_truncates_context_not_target(tokenizer):
    ids = encode_pair(tokenizer, "What?", "blah " * 300, max_length=32)
    assert len(ids) == 32
    assert ids[:len(tokenizer("What?")["input_ids"]) - 1] == tokenizer("What?")["input_ids"][:-1]


def test_encode_pair_handles_target_longer_than_max_length(tokenizer):
    # used to raise "Sequence to truncate too short" when a context was given
    long = "This is unbelievable! " * 300
    ids = encode_pair(tokenizer, long, "Hey. | What happened?", max_length=128)
    assert ids == tokenizer(long, truncation=True, max_length=128)["input_ids"]


def test_cpu_supports_bf16_is_false_without_proc_cpuinfo(monkeypatch):
    real_open = builtins.open

    def fake_open(path, *a, **k):
        if str(path) == "/proc/cpuinfo":
            raise OSError("no /proc on macOS")
        return real_open(path, *a, **k)
    monkeypatch.setattr(builtins, "open", fake_open)
    assert cpu_supports_bf16() is False


@pytest.fixture(scope="module")
def clf():
    from src.emotion.predict import EmotionClassifier
    return EmotionClassifier()


@needs_model
def test_predict_edge_cases(clf):
    long = "This is unbelievable! " * 300
    for text, ctx in [("", None), ("   ", None), ("What?", []), ("What?", "He left."), (long, ["Hey."]),
                      ("Café — naïve… 😀", None)]:
        p = clf.predict(text, context=ctx)
        assert list(p) and set(p) == set(LABELS) and abs(sum(p.values()) - 1) < 1e-4
        assert list(p.values()) == sorted(p.values(), reverse=True)
    many = ["one.", "two.", "My dog died.", "I'm so sorry."]
    assert clf.predict("Oh no.", context=many) == clf.predict("Oh no.", context=many[-clf.context_turns:])
    assert clf.predict_conversation([]) == []
    with pytest.raises(ValueError):
        clf.predict_proba(["a", "b"], [None])


@needs_model
def test_predict_pair_matches_two_predict_calls(clf):
    ctx = ["We won the tickets!", "Best weekend ever!"]
    with_ctx, without_ctx = clf.predict_pair("Come on!", ctx)
    for got, want in [(with_ctx, clf.predict("Come on!", context=ctx)), (without_ctx, clf.predict("Come on!"))]:
        assert list(got) == list(want)
        assert max(abs(got[k] - want[k]) for k in want) < 1e-4


@needs_model
def test_bf16_request_falls_back_without_cpu_support(monkeypatch):
    import src.emotion.predict as predict
    monkeypatch.setattr(predict, "cpu_supports_bf16", lambda: False)
    with pytest.warns(UserWarning):
        assert predict.EmotionClassifier(bf16=True).bf16 is False
    assert predict.EmotionClassifier(bf16="auto").bf16 is False
