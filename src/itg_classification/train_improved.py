"""Improvement experiments, grid search and the best model.

Run: `make train-improved`. Models are chosen by 5-fold CV on train.
"""

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
)
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from itg_classification.config import (
    BEST_MODEL_CARD_PATH,
    BEST_MODEL_PATH,
    FIGURES_DIR,
    ID_TO_LABEL,
    MODELS_DIR,
    RANDOM_STATE,
    RESULTS_DIR,
    TEST_PATH,
    TRAIN_PATH,
)
from itg_classification.features import (
    NEGATION_TOKEN_PATTERN,
    build_vectorizer,
    char_vectorizer,
    word_char_vectorizer,
    word_vectorizer,
)
from itg_classification.metrics import (
    confusion_table,
    evaluate,
    format_report,
    markdown_table,
    plot_confusion_matrix,
    plot_cv_comparison,
    save_metrics,
)
from itg_classification.split import load_split
from itg_classification.train_baseline import BASELINE_NAME, build_baseline

CV_FOLDS = 5


def text_pipeline(
    model: BaseEstimator, vectorizer: BaseEstimator | None = None
) -> Pipeline:
    """A vectorizer and a model, the baseline vectorizer by default."""
    return Pipeline(
        [("tfidf", vectorizer or build_vectorizer()), ("model", model)]
    )


def logreg(**params) -> LogisticRegression:
    """The baseline logistic regression with some settings changed."""
    return LogisticRegression(
        max_iter=1000, random_state=RANDOM_STATE, **params
    )


def balanced_logreg() -> Pipeline:
    """Errors on a class weigh inversely to its size: n / (3 * n_class).

    `negative` gets about 4.7 times the weight of an average text,
    `neutral` about 0.6, so missing a `negative` text costs as much
    as missing eight `neutral` ones.
    """
    return text_pipeline(logreg(class_weight="balanced"))


# Vectorizer variants, each tried with the baseline LogReg and with the
# balanced one: a better vectorizer can hide behind the weak `negative`.
VECTORIZERS: dict[str, Callable[[], BaseEstimator]] = {
    "unigrams": lambda: word_vectorizer(ngram_range=(1, 1)),
    "1-3 grams": lambda: word_vectorizer(ngram_range=(1, 3)),
    "negation tokens": lambda: word_vectorizer(
        token_pattern=NEGATION_TOKEN_PATTERN
    ),
    "char 2-5 grams": char_vectorizer,
    "words + chars": word_char_vectorizer,
}


# Other models on the baseline vectorizer. ComplementNB has no class
# weights: it is built for imbalanced text data by design.
MODELS: dict[str, Callable[[], BaseEstimator]] = {
    "LinearSVC": lambda: LinearSVC(random_state=RANDOM_STATE),
    "LinearSVC balanced": lambda: LinearSVC(
        class_weight="balanced", random_state=RANDOM_STATE
    ),
    "ComplementNB": ComplementNB,
    "RandomForest": lambda: RandomForestClassifier(
        n_estimators=300, n_jobs=-1, random_state=RANDOM_STATE
    ),
    "RandomForest balanced": lambda: RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    ),
}


def model_experiments() -> dict[str, Callable[[], Pipeline]]:
    return {
        name: lambda m=model: text_pipeline(m())
        for name, model in MODELS.items()
    }


def vectorizer_experiments() -> dict[str, Callable[[], Pipeline]]:
    experiments = {}
    for name, vectorizer in VECTORIZERS.items():
        experiments[f"LogReg, {name}"] = lambda v=vectorizer: text_pipeline(
            logreg(), v()
        )
        experiments[f"LogReg balanced, {name}"] = lambda v=vectorizer: (
            text_pipeline(logreg(class_weight="balanced"), v())
        )
    return experiments


EXPERIMENTS: dict[str, Callable[[], Pipeline]] = {
    BASELINE_NAME: build_baseline,
    "LogReg, balanced class weights": balanced_logreg,
    **vectorizer_experiments(),
    **model_experiments(),
}


def cv_splitter(
    folds: int = CV_FOLDS, seed: int = RANDOM_STATE
) -> StratifiedKFold:
    """The same folds for every experiment and every grid search."""
    return StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)


def cross_validate(
    pipeline: Pipeline,
    train: pd.DataFrame,
    folds: int = CV_FOLDS,
    seed: int = RANDOM_STATE,
) -> np.ndarray:
    """Macro F1 of every fold of a stratified K-fold on train.

    This is the score we choose models by: the test set stays untouched
    until the final report, so it is not tuned to.
    """
    return cross_val_score(
        pipeline,
        train["text"],
        train["label_id"],
        scoring="f1_macro",
        cv=cv_splitter(folds, seed),
    )


def word_char_logreg() -> Pipeline:
    return text_pipeline(
        logreg(class_weight="balanced"), word_char_vectorizer()
    )


def word_char_svc() -> Pipeline:
    # A larger C needs more iterations to converge on 100k features
    return text_pipeline(
        LinearSVC(
            class_weight="balanced", max_iter=5000, random_state=RANDOM_STATE
        ),
        word_char_vectorizer(),
    )


# The two best setups of the previous steps. `C` is the inverse of the
# regularization strength: a larger C lets the weights grow and fit
# train closer.
GRIDS: dict[str, tuple[Callable[[], Pipeline], dict[str, list]]] = {
    "LogReg balanced, words + chars, tuned": (
        word_char_logreg,
        {
            "model__C": [0.3, 1, 3, 10, 30],
            "tfidf__chars__ngram_range": [(2, 5), (3, 5)],
        },
    ),
    "LinearSVC balanced, words + chars, tuned": (
        word_char_svc,
        {"model__C": [0.03, 0.1, 0.3, 1, 3]},
    ),
}


def grid_search(
    build: Callable[[], Pipeline], grid: dict[str, list], train: pd.DataFrame
) -> GridSearchCV:
    """Tries every combination of `grid` with CV macro F1 on train."""
    search = GridSearchCV(
        build(), grid, scoring="f1_macro", cv=cv_splitter(), n_jobs=-1
    )
    return search.fit(train["text"], train["label_id"])


def grid_table(search: GridSearchCV) -> pd.DataFrame:
    """Mean and std of the CV macro F1 for every tried combination."""
    results = pd.DataFrame(search.cv_results_)
    params = pd.DataFrame(results["params"].tolist())
    table = params.assign(
        cv_macro_f1=results["mean_test_score"],
        cv_std=results["std_test_score"],
    )
    return table.sort_values("cv_macro_f1", ascending=False)


def with_params(
    build: Callable[[], Pipeline], params: dict
) -> Callable[[], Pipeline]:
    """A factory of `build` pipelines with `params` set."""
    return lambda: build().set_params(**params)


def run_experiment(
    build: Callable[[], Pipeline], train: pd.DataFrame, test: pd.DataFrame
) -> tuple[Pipeline, dict]:
    """CV scores on train, then a fit on the whole train and test scores."""
    scores = cross_validate(build(), train)
    pipeline = build().fit(train["text"], train["label_id"])
    return pipeline, {
        "cv_macro_f1": scores.tolist(),
        "test": evaluate(test["label_id"], pipeline.predict(test["text"])),
    }


def experiments_table(results: dict, reference: str) -> pd.DataFrame:
    """One row per experiment, sorted by the CV macro F1."""
    rows = {}
    for name, result in results.items():
        cv = np.array(result["cv_macro_f1"])
        test = result["test"]
        rows[name] = {
            "cv_macro_f1": cv.mean(),
            "cv_std": cv.std(),
            "test_macro_f1": test["macro_f1"],
            "test_accuracy": test["accuracy"],
            "negative_recall": test["per_class"]["negative"]["recall"],
        }
    table = pd.DataFrame(rows).T
    reference_cv = np.mean(results[reference]["cv_macro_f1"])
    reference_test = results[reference]["test"]["macro_f1"]
    table["cv_change"] = table["cv_macro_f1"] / reference_cv - 1
    table["test_change"] = table["test_macro_f1"] / reference_test - 1
    return table.sort_values("cv_macro_f1", ascending=False)


def run_experiments(
    experiments: dict[str, Callable[[], Pipeline]],
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> dict:
    return {
        name: run_experiment(build, train, test)[1]
        for name, build in experiments.items()
    }


def model_card(name: str, pipeline: Pipeline, result: dict, rows: int) -> dict:
    """What the saved model is, how it was trained and how good it was.

    Inference reads it to check that the model file matches the code:
    the same classes in the same order and the same scikit-learn.
    """
    test = result["test"]
    return {
        "name": name,
        "labels": [
            ID_TO_LABEL[int(label_id)] for label_id in pipeline.classes_
        ],
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "train_rows": rows,
        "sklearn_version": sklearn.__version__,
        "cv_macro_f1": float(np.mean(result["cv_macro_f1"])),
        "test_macro_f1": test["macro_f1"],
        "test_accuracy": test["accuracy"],
    }


# LinearSVC tuned is 0.003 higher on CV, eight times less than the fold
# std, and equal on test. LogReg wins the tie: it finds more `negative`
# texts, varies less between folds, needs no tuning and gives
# probabilities for inference.
BEST_NAME = "LogReg balanced, words + chars"


def main(
    train_path: Path = TRAIN_PATH,
    test_path: Path = TEST_PATH,
    models_dir: Path = MODELS_DIR,
    results_dir: Path = RESULTS_DIR,
    figures_dir: Path = FIGURES_DIR,
    experiments: dict[str, Callable[[], Pipeline]] = EXPERIMENTS,
    grids: dict = GRIDS,
    best_name: str = BEST_NAME,
) -> None:
    train, test = load_split(train_path), load_split(test_path)
    experiments = dict(experiments)
    grid_results = {}
    for name, (build, grid) in grids.items():
        search = grid_search(build, grid, train)
        grid_results[name] = grid_table(search).to_dict("records")
        experiments[name] = with_params(build, search.best_params_)
    results = run_experiments(experiments, train, test)
    table = experiments_table(results, reference=BASELINE_NAME)

    best = experiments[best_name]().fit(train["text"], train["label_id"])
    predicted = best.predict(test["text"])
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(best, models_dir / BEST_MODEL_PATH.name)
    save_metrics(
        model_card(best_name, best, results[best_name], len(train)),
        models_dir / BEST_MODEL_CARD_PATH.name,
    )
    save_metrics(
        {
            "reference": BASELINE_NAME,
            "best": best_name,
            "cv_folds": CV_FOLDS,
            "experiments": results,
            "grid_search": grid_results,
        },
        results_dir / "improved_metrics.json",
    )
    (results_dir / "comparison.md").write_text(markdown_table(table))
    (results_dir / "best_report.txt").write_text(
        format_report(
            best_name,
            results[best_name]["test"],
            confusion_table(test["label_id"], predicted),
        )
    )
    plot_cv_comparison(table, path=figures_dir / "experiments_comparison.png")
    plot_confusion_matrix(
        test["label_id"],
        predicted,
        title=best_name,
        path=figures_dir / "confusion_matrix_best.png",
    )
    print(table.round(3).to_string())
    print(f"\nBest: {best_name}, saved to {models_dir / BEST_MODEL_PATH.name}")


if __name__ == "__main__":
    main()
