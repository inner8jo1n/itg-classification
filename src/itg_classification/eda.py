"""Exploratory data analysis: structure, duplicates, classes, lengths."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from itg_classification.config import LABELS


def normalize_text(texts: pd.Series) -> pd.Series:
    """Lowercases and collapses whitespace, to catch near duplicates."""
    return texts.str.lower().str.replace(r"\s+", " ", regex=True).str.strip()


def column_overview(frame: pd.DataFrame) -> pd.DataFrame:
    """Dtype, missing and blank values of every column."""
    return pd.DataFrame(
        {
            "dtype": frame.dtypes.astype(str),
            "non_null": frame.notna().sum(),
            "missing": frame.isna().sum(),
            "blank": frame.apply(
                lambda column: column.astype(str).str.strip().eq("").sum()
            ),
        }
    )


def conflicting_texts(frame: pd.DataFrame) -> pd.Series:
    """Label sets of texts that occur with more than one label."""
    labels = frame.groupby("text")["label"].agg(
        lambda x: tuple(sorted(set(x)))
    )
    return labels[labels.map(len) > 1]


def conflicting_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Every occurrence of a conflicting text, grouped by text."""
    rows = frame[frame["text"].isin(conflicting_texts(frame).index)]
    return rows.sort_values(["text", "label"], kind="stable")


def duplicates_summary(frame: pd.DataFrame) -> pd.Series:
    """Counts of the kinds of repetition in the dataset."""
    return pd.Series(
        {
            "rows": len(frame),
            "unique_texts": frame["text"].nunique(),
            "duplicate_rows": int(frame.duplicated().sum()),
            "duplicate_texts": int(frame["text"].duplicated().sum()),
            "normalized_duplicate_texts": int(
                normalize_text(frame["text"]).duplicated().sum()
            ),
            "conflicting_texts": len(conflicting_texts(frame)),
        }
    )


def class_distribution(
    frame: pd.DataFrame, labels: tuple[str, ...] = LABELS
) -> pd.DataFrame:
    """Count and share of every class, in the order of `labels`."""
    counts = frame["label"].value_counts().reindex(labels, fill_value=0)
    return pd.DataFrame({"count": counts, "share": counts / counts.sum()})


def imbalance_ratio(distribution: pd.DataFrame) -> float:
    """How many times the largest class outnumbers the smallest one."""
    counts = distribution["count"]
    return float(counts.max() / counts.min())


def plot_class_distribution(
    frame: pd.DataFrame, path: Path | None = None
) -> plt.Figure:
    """Bar chart of class counts, annotated with their shares."""
    distribution = class_distribution(frame)
    figure, axis = plt.subplots(figsize=(6, 4))
    sns.barplot(
        x=distribution.index,
        y=distribution["count"],
        hue=distribution.index,
        ax=axis,
    )
    for position, row in enumerate(distribution.itertuples()):
        axis.annotate(
            f"{row.count} ({row.share:.0%})",
            (position, row.count),
            ha="center",
            va="bottom",
        )
    axis.set(title="Class distribution", xlabel="Label", ylabel="Texts")
    axis.margins(y=0.1)
    figure.tight_layout()
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=150)
    return figure


def text_lengths(texts: pd.Series) -> pd.DataFrame:
    """Length of every text in characters and in words."""
    return pd.DataFrame(
        {"chars": texts.str.len(), "words": texts.str.count(r"\S+")}
    )


def length_stats(
    frame: pd.DataFrame, labels: tuple[str, ...] = LABELS
) -> pd.DataFrame:
    """Length statistics per class, one row for each unit and class."""
    lengths = text_lengths(frame["text"]).groupby(frame["label"])
    return pd.concat(
        {
            unit: lengths[unit].describe().reindex(labels)
            for unit in ["chars", "words"]
        },
        names=["unit"],
    )


def plot_text_lengths(
    frame: pd.DataFrame, path: Path | None = None
) -> plt.Figure:
    """Histograms of text lengths, one curve per class.

    Classes are normalized separately, so the shapes stay comparable
    despite the class imbalance.
    """
    lengths = text_lengths(frame["text"]).assign(label=frame["label"])
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    for axis, unit in zip(axes, ["chars", "words"], strict=True):
        sns.histplot(
            data=lengths,
            x=unit,
            hue="label",
            hue_order=LABELS,
            stat="density",
            common_norm=False,
            element="step",
            ax=axis,
        )
        axis.set(xlabel=f"Length, {unit}", ylabel="Share of class texts")
    figure.suptitle("Text length distribution")
    figure.tight_layout()
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=150)
    return figure


def sample_per_class(
    frame: pd.DataFrame,
    n: int,
    labels: tuple[str, ...] = LABELS,
    seed: int = 42,
) -> pd.DataFrame:
    """`n` random texts of every class, indexed by label and row."""
    parts = {
        label: frame.loc[frame["label"] == label, ["text"]].sample(
            n, random_state=seed
        )
        for label in labels
    }
    return pd.concat(parts, names=["label", "row"])


TEXT_PATTERNS = {
    "cashtag": r"\$[A-Za-z]{1,6}\b",
    "url": r"https?://",
}


def pattern_shares(
    frame: pd.DataFrame,
    patterns: dict[str, str] = TEXT_PATTERNS,
    labels: tuple[str, ...] = LABELS,
) -> pd.DataFrame:
    """Share of texts of every class that contain each pattern."""
    return pd.DataFrame(
        {
            name: frame["text"]
            .str.contains(pattern, regex=True)
            .groupby(frame["label"])
            .mean()
            .reindex(labels)
            for name, pattern in patterns.items()
        }
    ).T
