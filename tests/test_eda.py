import pandas as pd
import pytest

from itg_classification.eda import (
    class_distribution,
    column_overview,
    conflicting_rows,
    conflicting_texts,
    duplicates_summary,
    imbalance_ratio,
    length_stats,
    normalize_text,
    pattern_shares,
    plot_class_distribution,
    plot_text_lengths,
    sample_per_class,
    text_lengths,
)


@pytest.fixture
def frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "text": [
                "Shares rose",
                "Shares rose",
                "Profit fell",
                "Profit fell",
                "profit  FELL ",
                "Sales flat",
            ],
            "label": [
                "positive",
                "positive",
                "negative",
                "neutral",
                "negative",
                "neutral",
            ],
        }
    )


def test_normalize_text():
    texts = pd.Series(["  Profit\tFELL  now "])

    assert normalize_text(texts).tolist() == ["profit fell now"]


def test_column_overview_counts_missing_and_blank():
    frame = pd.DataFrame({"text": ["a", None, "  "], "label": ["x", "y", "z"]})

    overview = column_overview(frame)

    assert overview.loc["text", "missing"] == 1
    assert overview.loc["text", "blank"] == 1
    assert overview.loc["label", "non_null"] == 3


def test_conflicting_texts(frame):
    conflicts = conflicting_texts(frame)

    assert conflicts.to_dict() == {"Profit fell": ("negative", "neutral")}


def test_conflicting_rows_keeps_every_occurrence(frame):
    rows = conflicting_rows(frame)

    assert rows["label"].tolist() == ["negative", "neutral"]


def test_duplicates_summary(frame):
    summary = duplicates_summary(frame)

    assert summary.to_dict() == {
        "rows": 6,
        "unique_texts": 4,
        "duplicate_rows": 1,
        "duplicate_texts": 2,
        "normalized_duplicate_texts": 3,
        "conflicting_texts": 1,
    }


def test_class_distribution_follows_label_order(frame):
    distribution = class_distribution(frame)

    assert distribution.index.tolist() == ["negative", "neutral", "positive"]
    assert distribution["count"].tolist() == [2, 2, 2]
    assert distribution["share"].sum() == pytest.approx(1)


def test_class_distribution_keeps_missing_classes():
    frame = pd.DataFrame({"text": ["a"], "label": ["neutral"]})

    distribution = class_distribution(frame)

    assert distribution["count"].tolist() == [0, 1, 0]


def test_imbalance_ratio():
    distribution = pd.DataFrame({"count": [10, 30, 20]})

    assert imbalance_ratio(distribution) == 3


def test_plot_class_distribution_saves_figure(frame, tmp_path):
    path = tmp_path / "figures" / "classes.png"

    figure = plot_class_distribution(frame, path)

    assert path.exists()
    assert figure.axes[0].get_title() == "Class distribution"


def test_text_lengths():
    lengths = text_lengths(pd.Series(["Shares  rose", "Up"]))

    assert lengths.to_dict("list") == {"chars": [12, 2], "words": [2, 1]}


def test_length_stats_per_unit_and_class(frame):
    stats = length_stats(frame)

    assert stats.index.tolist() == [
        ("chars", "negative"),
        ("chars", "neutral"),
        ("chars", "positive"),
        ("words", "negative"),
        ("words", "neutral"),
        ("words", "positive"),
    ]
    assert stats.loc[("words", "positive"), "mean"] == 2
    assert stats.loc[("chars", "neutral"), "max"] == len("Profit fell")


def test_plot_text_lengths_saves_figure(frame, tmp_path):
    path = tmp_path / "lengths.png"

    figure = plot_text_lengths(frame, path)

    assert path.exists()
    assert len(figure.axes) == 2


def test_sample_per_class(frame):
    sample = sample_per_class(frame, n=2)

    texts = sample["text"].groupby(level="label").agg(sorted).to_dict()
    assert texts == {
        "negative": ["Profit fell", "profit  FELL "],
        "neutral": ["Profit fell", "Sales flat"],
        "positive": ["Shares rose", "Shares rose"],
    }


def test_sample_per_class_is_reproducible(frame):
    first = sample_per_class(frame, n=1, seed=0)
    second = sample_per_class(frame, n=1, seed=0)

    pd.testing.assert_frame_equal(first, second)


def test_pattern_shares():
    frame = pd.DataFrame(
        {
            "text": [
                "$AAPL up",
                "Profit fell",
                "Sales flat http://x.co",
                "Shares rose",
            ],
            "label": ["negative", "negative", "neutral", "positive"],
        }
    )

    shares = pattern_shares(frame, {"cashtag": r"\$[A-Z]+", "url": "http"})

    assert shares.to_dict("index") == {
        "cashtag": {"negative": 0.5, "neutral": 0, "positive": 0},
        "url": {"negative": 0, "neutral": 1, "positive": 0},
    }
