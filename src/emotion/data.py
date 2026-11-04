"""Loading, cleaning and formatting MELD dialogue data.

MELD (Poria et al., 2019) contains ~13k utterances from ~1.4k dialogues of the TV
series *Friends*. Every utterance is labelled with one of seven emotions.
Utterances are grouped into dialogues, so we can give a model the preceding turns
as *context* -- emotions in conversation depend heavily on what was just said.
"""
from __future__ import annotations

import re
import unicodedata

import pandas as pd

from .config import DATA_DIR, LABEL2ID, SPLIT_FILES

# Typographic characters that appear in MELD transcripts -> plain ASCII.
_CHAR_MAP = str.maketrans({
    "’": "'", "‘": "'", "“": '"', "”": '"',
    "—": " - ", "–": " - ", "…": "...", "\u0092": "'",
    "\u0091": "'", "\u0093": '"', "\u0094": '"', "\u0097": " - ", "\u0085": "...",
})
_WS = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Normalise unicode/punctuation and whitespace. Case and punctuation are kept,
    because they carry emotional signal ("WHAT?!" vs "what.")."""
    text = unicodedata.normalize("NFKC", str(text)).translate(_CHAR_MAP)
    return _WS.sub(" ", text).strip()


def load_split(split: str) -> pd.DataFrame:
    """Load one MELD split with cleaned text and integer labels, sorted in dialogue order."""
    df = pd.read_csv(DATA_DIR / SPLIT_FILES[split], encoding="utf-8")
    df = df.rename(columns={"Utterance": "text", "Speaker": "speaker", "Emotion": "emotion",
                            "Dialogue_ID": "dialogue_id", "Utterance_ID": "utterance_id"})
    df["text"] = df["text"].map(clean_text)
    df["speaker"] = df["speaker"].map(clean_text)
    df["label"] = df["emotion"].map(LABEL2ID)
    assert df["label"].notna().all(), "unexpected emotion label"
    df = df.sort_values(["dialogue_id", "utterance_id"]).reset_index(drop=True)
    return df[["dialogue_id", "utterance_id", "speaker", "text", "emotion", "label"]]


def load_all() -> dict[str, pd.DataFrame]:
    return {split: load_split(split) for split in SPLIT_FILES}


def add_context(df: pd.DataFrame, n_turns: int = 2) -> pd.DataFrame:
    """Add a ``context`` column holding the previous ``n_turns`` utterances of the same
    dialogue. Only *past* turns are used, so the model could run live in a chatbot (it
    never sees the future). Speaker names are deliberately left out so the model cannot
    learn character-specific shortcuts ("Ross is usually ...") that would not transfer
    to new users."""
    df = df.copy()
    contexts = []
    for _, dialog in df.groupby("dialogue_id", sort=False):
        turns = list(dialog["text"])
        for i in range(len(turns)):
            contexts.append(" | ".join(turns[max(0, i - n_turns):i]))
    df["context"] = contexts
    return df


def model_input(text: str, context: str = "", max_context_words: int = 60) -> tuple[str, str | None]:
    """Build the (target, context) text pair fed to the transformer.

    The target utterance goes first so it is never truncated. Very long contexts are
    trimmed from the *left*, keeping the turns closest to the target."""
    words = context.split()
    if len(words) > max_context_words:
        context = " ".join(words[-max_context_words:])
    return text, (context or None)
