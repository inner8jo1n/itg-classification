"""Baseline: TF-IDF and logistic regression with default settings.

Run: `make train-baseline`.
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from itg_classification.config import (
    FIGURES_DIR,
    MODELS_DIR,
    RANDOM_STATE,
    RESULTS_DIR,
    TEST_PATH,
    TRAIN_PATH,
)
from itg_classification.features import build_vectorizer
from itg_classification.metrics import (
    confusion_table,
    evaluate,
    format_report,
    plot_confusion_matrix,
    save_metrics,
)
from itg_classification.split import load_split

BASELINE_NAME = "Baseline TF-IDF + LogReg"


def build_baseline() -> Pipeline:
    """TF-IDF and logistic regression with default settings.

    One pipeline object keeps the vectorizer and the model together, so
    the vectorizer is fitted on train only and inference cannot pair
    the model with a different vocabulary. `max_iter` is raised only so
    that the solver converges, it does not change the model.
    """
    return Pipeline(
        [
            ("tfidf", build_vectorizer()),
            (
                "model",
                LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
            ),
        ]
    )


def train_baseline(train: pd.DataFrame) -> Pipeline:
    return build_baseline().fit(train["text"], train["label_id"])


def train_majority(train: pd.DataFrame) -> DummyClassifier:
    """Always predicts the most frequent class: the floor to beat."""
    return DummyClassifier(strategy="most_frequent").fit(
        train["text"], train["label_id"]
    )


def main(
    train_path: Path = TRAIN_PATH,
    test_path: Path = TEST_PATH,
    models_dir: Path = MODELS_DIR,
    results_dir: Path = RESULTS_DIR,
    figures_dir: Path = FIGURES_DIR,
) -> None:
    train, test = load_split(train_path), load_split(test_path)
    models = {
        "Majority class": train_majority(train),
        BASELINE_NAME: train_baseline(train),
    }

    results = {}
    reports = []
    for name, model in models.items():
        predicted = model.predict(test["text"])
        metrics = evaluate(test["label_id"], predicted)
        results[name] = metrics
        confusion = confusion_table(test["label_id"], predicted)
        reports.append(format_report(name, metrics, confusion))

    baseline = models[BASELINE_NAME]
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(baseline, models_dir / "baseline.joblib")
    save_metrics(
        {
            "train_rows": len(train),
            "test_rows": len(test),
            "models": results,
        },
        results_dir / "baseline_metrics.json",
    )
    report = "\n".join(reports)
    (results_dir / "baseline_report.txt").write_text(report)
    plot_confusion_matrix(
        test["label_id"],
        baseline.predict(test["text"]),
        title=BASELINE_NAME,
        path=figures_dir / "confusion_matrix_baseline.png",
    )
    print(report)
    print(f"Saved the model to {models_dir / 'baseline.joblib'}")
    print(f"Saved metrics and report to {results_dir}")


if __name__ == "__main__":
    main()
