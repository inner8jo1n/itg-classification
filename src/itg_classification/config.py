"""Paths, labels and shared settings of the whole project."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

# Kaggle "Financial Sentiment Analysis" (sbhatti/financial-sentiment-analysis)
RAW_DATA_PATH = DATA_DIR / "raw" / "financial_sentiment.csv"
RAW_COLUMNS = {"Sentence": "text", "Sentiment": "label"}
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "financial_sentiment.csv"
TRAIN_PATH = DATA_DIR / "processed" / "train.csv"
TEST_PATH = DATA_DIR / "processed" / "test.csv"

TEST_SIZE = 0.2
RANDOM_STATE = 42

LABELS = ("negative", "neutral", "positive")
# Ids follow the order of LABELS, so they also go from bad to good
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
ID_TO_LABEL = dict(enumerate(LABELS))

FIGURES_DIR = PROJECT_ROOT / "figures"
DEMO_PATH = PROJECT_ROOT / "demo" / "examples.csv"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"
BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"
# Model card: what the saved model is and how good it was
BEST_MODEL_CARD_PATH = MODELS_DIR / "best_model.json"

# Below it the best model is right in about 57% of test texts, above it
# in 90%, see notebooks/04_error_analysis.ipynb
LOW_CONFIDENCE = 0.6
