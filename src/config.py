"""
Central configuration for the Fake News Detection project.

Every other module imports paths / label conventions / the model registry
from here, so there is exactly one place to change a filename or a folder
instead of five copies of the same string literal.
"""

from pathlib import Path

# --- Folders ---------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# --- Label convention (0 = fake, 1 = real) ---------------------------------

FAKE, REAL = 0, 1
LABEL_NAMES = {FAKE: "FAKE", REAL: "REAL"}
TARGET_NAMES = ["fake", "real"]  # index == label, for sklearn reports

# --- Artifact filenames ----------------------------------------------------

VECTORIZER_FILE = "tfidf_vectorizer.joblib"
METRICS_TXT = "metrics.txt"
METRICS_JSON = "metrics.json"
MODEL_CARD = "model_card.md"
RUN_META = "run_metadata.json"

# key -> (human-readable name, artifact filename)
MODEL_REGISTRY = {
    "nb": ("Naive Bayes", "naive_bayes_model.joblib"),
    "svm": ("SVM (Linear, calibrated)", "svm_model.joblib"),
    "lr": ("Logistic Regression", "logistic_regression_model.joblib"),
}

MODEL_KEYS = list(MODEL_REGISTRY)
DISPLAY_TO_KEY = {name: key for key, (name, _) in MODEL_REGISTRY.items()}


def model_path(key: str, models_dir=None) -> Path:
    """Path to a trained model artifact, by registry key ('nb'/'svm'/'lr')."""
    if key not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model key {key!r}. Known keys: {MODEL_KEYS}")
    base = Path(models_dir) if models_dir else MODELS_DIR
    return base / MODEL_REGISTRY[key][1]


def model_name(key: str) -> str:
    return MODEL_REGISTRY[key][0]


# --- Prediction store ------------------------------------------------------

DB_PATH = REPORTS_DIR / "predictions.db"

# --- Defaults shared by train / eval ---------------------------------------

RANDOM_STATE = 42
TEST_SIZE = 0.2
MAX_FEATURES = 5000
NGRAM_RANGE = (1, 2)


def ensure_dirs() -> None:
    """Create the output folders that the pipeline writes into."""
    for folder in (MODELS_DIR, REPORTS_DIR, FIGURES_DIR):
        folder.mkdir(parents=True, exist_ok=True)
