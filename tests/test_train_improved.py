import joblib
import pandas as pd
import pytest
import sklearn

from itg_classification.metrics import evaluate, load_metrics
from itg_classification.train_baseline import BASELINE_NAME, build_baseline
from itg_classification.train_improved import (
    BEST_NAME,
    EXPERIMENTS,
    GRIDS,
    MODELS,
    balanced_logreg,
    cross_validate,
    experiments_table,
    grid_search,
    grid_table,
    logreg,
    main,
    model_experiments,
    run_experiment,
    text_pipeline,
    with_params,
)


@pytest.fixture
def train() -> pd.DataFrame:
    texts = {
        0: ["profit fell", "sales fell sharply", "net loss grew"],
        1: ["the company said", "the report is out", "the meeting is today"],
        2: ["profit rose", "sales rose sharply", "net profit grew"],
    }
    rows = [(text, label) for label, group in texts.items() for text in group]
    return pd.DataFrame(rows * 5, columns=["text", "label_id"])


def test_cross_validate_gives_one_score_per_fold(train):
    scores = cross_validate(build_baseline(), train, folds=5)

    assert len(scores) == 5
    assert all(0 <= score <= 1 for score in scores)


def test_cross_validate_is_reproducible(train):
    first = cross_validate(build_baseline(), train, folds=3, seed=0)
    second = cross_validate(build_baseline(), train, folds=3, seed=0)

    assert first.tolist() == second.tolist()


def test_run_experiment_fits_on_train_and_scores_test(train):
    pipeline, result = run_experiment(build_baseline, train, train)

    assert pipeline.predict(["profit fell"]).tolist() == [0]
    assert len(result["cv_macro_f1"]) == 5
    assert result["test"]["macro_f1"] == 1


def make_result(cv: list[float], y_pred: list[int]) -> dict:
    return {"cv_macro_f1": cv, "test": evaluate([0, 1, 2, 2], y_pred)}


def test_experiments_table_sorts_by_cv_and_compares_with_reference():
    results = {
        "Baseline": make_result([0.6, 0.6], [1, 1, 2, 2]),
        "Better": make_result([0.7, 0.8], [0, 1, 2, 2]),
    }

    table = experiments_table(results, reference="Baseline")

    assert table.index.tolist() == ["Better", "Baseline"]
    assert table.loc["Better", "cv_macro_f1"] == pytest.approx(0.75)
    assert table.loc["Better", "cv_std"] == pytest.approx(0.05)
    assert table.loc["Better", "cv_change"] == pytest.approx(0.75 / 0.6 - 1)
    assert table.loc["Baseline", "test_change"] == 0
    assert table.loc["Better", "negative_recall"] == 1


def test_text_pipeline_uses_baseline_vectorizer_by_default():
    pipeline = text_pipeline(logreg())

    assert pipeline.named_steps["tfidf"].get_params()["ngram_range"] == (1, 2)


def test_logreg_keeps_baseline_settings():
    model = logreg(C=10)

    assert (model.C, model.max_iter) == (10, 1000)


def test_balanced_logreg_weights_classes():
    model = balanced_logreg().named_steps["model"]

    assert model.class_weight == "balanced"


@pytest.mark.parametrize("name", list(EXPERIMENTS))
def test_every_experiment_builds_a_fresh_pipeline(name, train):
    build = EXPERIMENTS[name]

    assert build() is not build()
    assert build().fit(train["text"], train["label_id"]).predict(
        ["profit fell"]
    ).tolist() == [0]


def test_model_experiments_use_baseline_vectorizer():
    experiments = model_experiments()

    assert list(experiments) == list(MODELS)
    for build in experiments.values():
        vectorizer = build().named_steps["tfidf"]
        assert vectorizer.get_params()["ngram_range"] == (1, 2)


def test_model_experiments_keep_their_own_model():
    models = {
        name: type(build().named_steps["model"]).__name__
        for name, build in model_experiments().items()
    }

    assert models == {
        "LinearSVC": "LinearSVC",
        "LinearSVC balanced": "LinearSVC",
        "ComplementNB": "ComplementNB",
        "RandomForest": "RandomForestClassifier",
        "RandomForest balanced": "RandomForestClassifier",
    }


def test_grid_search_tries_every_combination(train):
    search = grid_search(
        build_baseline, {"model__C": [0.1, 1], "tfidf__min_df": [1, 2]}, train
    )

    table = grid_table(search)

    assert len(table) == 4
    assert table.columns.tolist() == [
        "model__C",
        "tfidf__min_df",
        "cv_macro_f1",
        "cv_std",
    ]
    assert table["cv_macro_f1"].is_monotonic_decreasing


def test_with_params_sets_params_on_a_fresh_pipeline():
    build = with_params(build_baseline, {"model__C": 10})

    first, second = build(), build()

    assert first is not second
    assert first.named_steps["model"].C == 10


@pytest.mark.parametrize("name", list(GRIDS))
def test_grids_match_their_pipelines(name):
    build, grid = GRIDS[name]

    for param, values in grid.items():
        build().set_params(**{param: values[0]})


def test_best_name_is_an_experiment():
    assert BEST_NAME in EXPERIMENTS


def test_main_saves_best_model_and_results(train, tmp_path, capsys):
    train.to_csv(tmp_path / "train.csv", index=False)
    train.to_csv(tmp_path / "test.csv", index=False)

    main(
        tmp_path / "train.csv",
        tmp_path / "test.csv",
        tmp_path / "models",
        tmp_path / "results",
        tmp_path / "figures",
        experiments={
            BASELINE_NAME: build_baseline,
            "Balanced": balanced_logreg,
        },
        grids={"Tuned": (build_baseline, {"model__C": [0.1, 1]})},
        best_name="Balanced",
    )

    model = joblib.load(tmp_path / "models" / "best_model.joblib")
    assert model.named_steps["model"].class_weight == "balanced"
    card = load_metrics(tmp_path / "models" / "best_model.json")
    assert card["name"] == "Balanced"
    assert card["labels"] == ["negative", "neutral", "positive"]
    assert card["train_rows"] == len(train)
    assert card["sklearn_version"] == sklearn.__version__
    saved = load_metrics(tmp_path / "results" / "improved_metrics.json")
    assert saved["best"] == "Balanced"
    assert set(saved["experiments"]) == {BASELINE_NAME, "Balanced", "Tuned"}
    assert len(saved["grid_search"]["Tuned"]) == 2
    assert (
        "| Balanced |" in (tmp_path / "results" / "comparison.md").read_text()
    )
    assert (tmp_path / "results" / "best_report.txt").exists()
    assert (tmp_path / "figures" / "experiments_comparison.png").exists()
    assert (tmp_path / "figures" / "confusion_matrix_best.png").exists()
    assert "Best: Balanced" in capsys.readouterr().out
