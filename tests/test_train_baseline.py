import joblib
import pandas as pd
import pytest

from itg_classification.metrics import load_metrics
from itg_classification.train_baseline import (
    BASELINE_NAME,
    build_baseline,
    main,
    train_baseline,
    train_majority,
)


@pytest.fixture
def train() -> pd.DataFrame:
    texts = {
        0: ["profit fell", "sales fell sharply", "net loss grew"],
        1: ["the company said", "the report is out", "the meeting is today"],
        2: ["profit rose", "sales rose sharply", "net profit grew"],
    }
    rows = [(text, label) for label, group in texts.items() for text in group]
    return pd.DataFrame(rows * 3, columns=["text", "label_id"])


def test_build_baseline_steps():
    pipeline = build_baseline()

    assert list(pipeline.named_steps) == ["tfidf", "model"]


def test_train_baseline_learns_the_classes(train):
    pipeline = train_baseline(train)

    assert pipeline.classes_.tolist() == [0, 1, 2]
    assert pipeline.predict(["profit fell", "profit rose"]).tolist() == [0, 2]


def test_train_baseline_gives_probabilities(train):
    pipeline = train_baseline(train)

    probabilities = pipeline.predict_proba(["the company said"])

    assert probabilities.shape == (1, 3)
    assert probabilities.sum() == pytest.approx(1)


def test_train_baseline_fits_vectorizer_on_train_only(train):
    pipeline = train_baseline(train)

    vocabulary = pipeline.named_steps["tfidf"].vocabulary_
    assert "dividend" not in vocabulary
    assert pipeline.predict_proba(["dividend"]).shape == (1, 3)


def test_train_majority_predicts_the_most_frequent_class(train):
    train = pd.concat([train, train[train["label_id"] == 1]])

    majority = train_majority(train)

    assert majority.predict(["profit fell", "profit rose"]).tolist() == [1, 1]


def test_main_saves_model_metrics_and_figure(train, tmp_path, capsys):
    train.to_csv(tmp_path / "train.csv", index=False)
    train.to_csv(tmp_path / "test.csv", index=False)

    main(
        tmp_path / "train.csv",
        tmp_path / "test.csv",
        tmp_path / "models",
        tmp_path / "results",
        tmp_path / "figures",
    )

    model = joblib.load(tmp_path / "models" / "baseline.joblib")
    assert model.predict(["profit fell"]).tolist() == [0]
    results = load_metrics(tmp_path / "results" / "baseline_metrics.json")
    assert list(results["models"]) == ["Majority class", BASELINE_NAME]
    assert (tmp_path / "results" / "baseline_report.txt").exists()
    assert (tmp_path / "figures" / "confusion_matrix_baseline.png").exists()
    assert "Macro F1" in capsys.readouterr().out
