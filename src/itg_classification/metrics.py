"""Metrics, reports, comparison tables and their plots."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from numpy.typing import ArrayLike
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from itg_classification.config import ID_TO_LABEL

LABEL_IDS = list(ID_TO_LABEL)


def evaluate(y_true: ArrayLike, y_pred: ArrayLike) -> dict:
    """Accuracy, macro and weighted F1, and per-class scores.

    Macro F1 is the main metric: it averages the classes equally, so
    the 7% `negative` class counts as much as the 54% `neutral` one.
    `zero_division=0` scores a class that is never predicted as 0
    instead of warning, which happens with weak models.
    """
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=LABEL_IDS, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=LABEL_IDS,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=LABEL_IDS,
                average="weighted",
                zero_division=0,
            )
        ),
        "per_class": {
            ID_TO_LABEL[label_id]: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, label_id in enumerate(LABEL_IDS)
        },
    }


def per_class_table(metrics: dict) -> pd.DataFrame:
    """Per-class scores as a table, one row per class."""
    table = pd.DataFrame(metrics["per_class"]).T
    return table.astype({"support": int})


def confusion_table(y_true: ArrayLike, y_pred: ArrayLike) -> pd.DataFrame:
    """Confusion matrix with label names: rows true, columns predicted."""
    names = list(ID_TO_LABEL.values())
    return pd.DataFrame(
        confusion_matrix(y_true, y_pred, labels=LABEL_IDS),
        index=pd.Index(names, name="true"),
        columns=pd.Index(names, name="predicted"),
    )


def draw_confusion_matrix(
    axis: plt.Axes,
    y_true: ArrayLike,
    y_pred: ArrayLike,
    title: str,
    colorbar: bool = True,
) -> None:
    """Counts, with every row normalized for the color: recall per class."""
    table = confusion_table(y_true, y_pred)
    recall = table.div(table.sum(axis=1).clip(lower=1), axis=0)
    sns.heatmap(
        recall,
        annot=table,
        fmt="d",
        cmap="Blues",
        vmin=0,
        vmax=1,
        cbar=colorbar,
        cbar_kws={"label": "Share of the true class"} if colorbar else None,
        ax=axis,
    )
    axis.set(title=title, xlabel="Predicted", ylabel="True")


def plot_confusion_matrix(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    title: str = "Confusion matrix",
    path: Path | None = None,
) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(5.5, 4.5))
    draw_confusion_matrix(axis, y_true, y_pred, title)
    figure.tight_layout()
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=150)
    return figure


def format_report(name: str, metrics: dict, confusion: pd.DataFrame) -> str:
    """Human-readable summary of one model on the test set."""
    return "\n".join(
        [
            f"Model: {name}",
            "",
            f"Accuracy:    {metrics['accuracy']:.4f}",
            f"Macro F1:    {metrics['macro_f1']:.4f}",
            f"Weighted F1: {metrics['weighted_f1']:.4f}",
            "",
            per_class_table(metrics).round(4).to_string(),
            "",
            "Confusion matrix (rows: true, columns: predicted)",
            confusion.to_string(),
            "",
        ]
    )


def save_metrics(results: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2) + "\n")


def load_metrics(path: Path) -> dict:
    return json.loads(path.read_text())


def comparison_table(models: dict, reference: str) -> pd.DataFrame:
    """Main scores of every model and the macro F1 change vs `reference`.

    `models` maps a model name to the output of `evaluate`. The change
    is relative: +5% means macro F1 is 1.05 times the reference one.
    """
    rows = {
        name: {
            "macro_f1": metrics["macro_f1"],
            "accuracy": metrics["accuracy"],
            "weighted_f1": metrics["weighted_f1"],
            "negative_recall": metrics["per_class"]["negative"]["recall"],
        }
        for name, metrics in models.items()
    }
    table = pd.DataFrame(rows).T
    reference_f1 = models[reference]["macro_f1"]
    table["macro_f1_change"] = table["macro_f1"] / reference_f1 - 1
    return table


def markdown_table(table: pd.DataFrame) -> str:
    """Experiments table as Markdown, scores rounded, changes in %."""
    header = (
        "| Method | CV macro F1 | CV std | Test macro F1 | Test accuracy "
        "| Negative recall | CV change | Test change |"
    )
    lines = [header, "|---|---|---|---|---|---|---|---|"]
    for name, row in table.iterrows():
        lines.append(
            f"| {name} | {row['cv_macro_f1']:.3f} | {row['cv_std']:.3f} "
            f"| {row['test_macro_f1']:.3f} | {row['test_accuracy']:.3f} "
            f"| {row['negative_recall']:.3f} | {row['cv_change']:+.1%} "
            f"| {row['test_change']:+.1%} |"
        )
    return "\n".join(lines) + "\n"


def plot_cv_comparison(
    table: pd.DataFrame, path: Path | None = None
) -> plt.Figure:
    """CV macro F1 with the fold std as error bars, test as dots."""
    table = table.iloc[::-1]
    figure, axis = plt.subplots(figsize=(9, 0.4 * len(table) + 1.5))
    positions = range(len(table))
    axis.barh(
        positions,
        table["cv_macro_f1"],
        xerr=table["cv_std"],
        color="tab:blue",
        alpha=0.7,
        label="CV macro F1 (± std)",
    )
    axis.scatter(
        table["test_macro_f1"],
        positions,
        color="tab:orange",
        zorder=3,
        label="Test macro F1",
    )
    axis.set_yticks(list(positions), table.index)
    axis.set(xlabel="Macro F1", title="Experiments", xlim=(0.5, 0.85))
    axis.legend(loc="lower right")
    figure.tight_layout()
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=150)
    return figure
