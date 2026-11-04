"""Central configuration: paths, label set, and default hyper-parameters."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "raw"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

# MELD's seven emotion classes. Order is fixed so label ids are stable everywhere.
LABELS = ["neutral", "joy", "surprise", "anger", "sadness", "disgust", "fear"]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}

SPLIT_FILES = {
    "train": "train_sent_emo.csv",
    "dev": "dev_sent_emo.csv",
    "test": "test_sent_emo.csv",
}

SEED = 42

# Default location of the fine-tuned transformer used by the demo and notebook.
DEFAULT_TRANSFORMER_DIR = MODELS_DIR / "roberta-meld-context"
BASELINE_PATH = MODELS_DIR / "tfidf_logreg.joblib"
