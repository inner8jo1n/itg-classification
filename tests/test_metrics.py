import pandas as pd
import pytest

from itg_classification.metrics import (
    comparison_table,
    confusion_table,
    evaluate,
    format_report,
    load_metrics,
    markdown_table,
    per_class_table,
    plot_confusion_matrix,
    plot_cv_comparison,
    save_metrics,
)

Y_TRUE = [0, 0, 1, 1, 1, 1, 2, 2]
Y_PRED = [0, 1, 1, 1, 1, 2, 2, 2]


def test_evaluate_overall_scores():
    metrics = evaluate(Y_TRUE, Y_PRED)

    assert metrics["accuracy"] == pytest.approx(6 / 8)
    # F1: negative 2/3, neutral 3/4, positive 4/5
    assert metrics["macro_f1"] == pytest.approx((2 / 3 + 3 / 4 + 4 / 5) / 3)
    assert metrics["weighted_f1"] == pytest.approx(
        (2 * 2 / 3 + 4 * 3 / 4 + 2 * 4 / 5) / 8
    )


def test_evaluate_per_class_scores():
    negative = evaluate(Y_TRUE, Y_PRED)["per_class"]["negative"]

    assert negative == {
        "precision": 1.0,
        "recall": 0.5,
        "f1": pytest.approx(2 / 3),
        "support": 2,
    }


def test_evaluate_scores_never_predicted_class_as_zero():
    metrics = evaluate([0, 1, 2], [1, 1, 1])

    assert metrics["per_class"]["negative"]["precision"] == 0
    assert metrics["macro_f1"] == pytest.approx(1 / 3 * 0.5)


def test_per_class_table():
    table = per_class_table(evaluate(Y_TRUE, Y_PRED))

    assert table.index.tolist() == ["negative", "neutral", "positive"]
    assert table.columns.tolist() == ["precision", "recall", "f1", "support"]
    assert table["support"].tolist() == [2, 4, 2]


def test_confusion_table():
    table = confusion_table(Y_TRUE, Y_PRED)

    assert table.to_dict("index") == {
        "negative": {"negative": 1, "neutral": 1, "positive": 0},
        "neutral": {"negative": 0, "neutral": 3, "positive": 1},
        "positive": {"negative": 0, "neutral": 0, "positive": 2},
    }


def test_plot_confusion_matrix_saves_figure(tmp_path):
    path = tmp_path / "cm.png"

    figure = plot_confusion_matrix(Y_TRUE, Y_PRED, title="Baseline", path=path)

    assert path.exists()
    assert figure.axes[0].get_title() == "Baseline"


def test_format_report():
    metrics = evaluate(Y_TRUE, Y_PRED)

    report = format_report(
        "Baseline", metrics, confusion_table(Y_TRUE, Y_PRED)
    )

    assert report.startswith("Model: Baseline\n")
    assert f"Macro F1:    {metrics['macro_f1']:.4f}" in report
    assert "Confusion matrix" in report


def test_save_and_load_metrics(tmp_path):
    metrics = evaluate(Y_TRUE, Y_PRED)
    path = tmp_path / "results" / "metrics.json"

    save_metrics({"baseline": metrics}, path)

    assert load_metrics(path) == {"baseline": metrics}


def test_comparison_table():
    models = {
        "Baseline": evaluate(Y_TRUE, Y_PRED),
        "Perfect": evaluate(Y_TRUE, Y_TRUE),
    }

    table = comparison_table(models, reference="Baseline")

    assert table.index.tolist() == ["Baseline", "Perfect"]
    assert table.loc["Baseline", "macro_f1_change"] == 0
    assert table.loc["Perfect", "macro_f1"] == 1
    assert table.loc["Perfect", "macro_f1_change"] == pytest.approx(
        1 / models["Baseline"]["macro_f1"] - 1
    )
    assert table.loc["Baseline", "negative_recall"] == 0.5


@pytest.fixture
def experiments() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cv_macro_f1": [0.75, 0.62],
            "cv_std": [0.02, 0.019],
            "test_macro_f1": [0.748, 0.672],
            "test_accuracy": [0.816, 0.795],
            "negative_recall": [0.638, 0.261],
            "cv_change": [0.21, 0.0],
            "test_change": [0.113, 0.0],
        },
        index=["Best", "Baseline"],
    )


def test_markdown_table(experiments):
    lines = markdown_table(experiments).splitlines()

    assert lines[0].startswith("| Method | CV macro F1 |")
    assert lines[2] == (
        "| Best | 0.750 | 0.020 | 0.748 | 0.816 | 0.638 | +21.0% | +11.3% |"
    )
    assert len(lines) == 4


def test_plot_cv_comparison_saves_figure(experiments, tmp_path):
    path = tmp_path / "experiments.png"

    figure = plot_cv_comparison(experiments, path)

    assert path.exists()
    labels = [label.get_text() for label in figure.axes[0].get_yticklabels()]
    assert labels == ["Baseline", "Best"]
