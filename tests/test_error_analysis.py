import joblib
import pandas as pd
import pytest

from itg_classification.error_analysis import (
    confidence_table,
    error_examples,
    error_types,
    errors,
    failure_masks,
    failure_table,
    main,
    plot_confidence,
    plot_confusion_by_source,
    predictions_frame,
    source_table,
    ticker_agreement,
    word_label_shares,
)
from itg_classification.train_baseline import train_baseline


@pytest.fixture
def train() -> pd.DataFrame:
    texts = {
        0: ["profit fell", "sales fell sharply", "net loss grew"],
        1: ["the company said", "the report is out", "the meeting is today"],
        2: ["profit rose", "sales rose sharply", "net profit grew"],
    }
    rows = [(text, label) for label, group in texts.items() for text in group]
    return pd.DataFrame(rows * 3, columns=["text", "label_id"])


@pytest.fixture
def test() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "raw_text": ["Profit fell .", "$ABC profit rose", "The report"],
            "text": ["profit fell.", "$abc profit rose", "the report"],
            "label": ["negative", "negative", "neutral"],
        }
    )


def test_predictions_frame(train, test):
    frame = predictions_frame(train_baseline(train), test)

    assert frame["predicted"].tolist() == ["negative", "positive", "neutral"]
    assert frame["correct"].tolist() == [True, False, True]
    assert frame["source"].tolist() == ["news", "tweet", "news"]
    assert frame["words"].tolist() == [2, 3, 2]
    probabilities = frame[["p_negative", "p_neutral", "p_positive"]]
    assert probabilities.sum(axis=1).round(6).eq(1).all()
    assert frame["confidence"].equals(probabilities.max(axis=1))


@pytest.fixture
def frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "label": ["negative", "negative", "neutral", "positive"],
            "predicted": ["positive", "positive", "positive", "positive"],
            "confidence": [0.6, 0.9, 0.7, 0.8],
            "correct": [False, False, False, True],
            "source": ["tweet", "news", "news", "news"],
        }
    )


def test_errors_sorted_by_confidence(frame):
    assert errors(frame)["confidence"].tolist() == [0.9, 0.7, 0.6]


def test_error_types(frame):
    table = error_types(frame)

    assert table.to_dict("list") == {
        "label": ["negative", "neutral"],
        "predicted": ["positive", "positive"],
        "errors": [2, 1],
        "share_of_errors": [2 / 3, 1 / 3],
    }


def test_main_saves_errors_report_and_figures(train, test, tmp_path, capsys):
    joblib.dump(train_baseline(train), tmp_path / "model.joblib")
    names = {0: "negative", 1: "neutral", 2: "positive"}
    train.assign(label=train["label_id"].map(names)).to_csv(
        tmp_path / "train.csv", index=False
    )
    test.to_csv(tmp_path / "test.csv", index=False)

    main(
        tmp_path / "model.joblib",
        tmp_path / "train.csv",
        tmp_path / "test.csv",
        tmp_path,
        tmp_path,
    )

    saved = pd.read_csv(tmp_path / "errors.csv")
    assert saved["raw_text"].tolist() == ["$ABC profit rose"]
    report = (tmp_path / "error_analysis.txt").read_text()
    assert report.startswith("Error analysis: 1 errors of 3 test texts")
    assert "Conclusions" in report
    assert "$ABC profit rose" in report
    assert (tmp_path / "confusion_by_source.png").exists()
    assert (tmp_path / "confidence.png").exists()
    assert "Failure cases" in capsys.readouterr().out


def test_source_table(frame):
    table = source_table(frame)

    assert table.to_dict("index") == {
        "news": {
            "texts": 3,
            "errors": 2,
            "error_rate": 2 / 3,
            "share_of_errors": 2 / 3,
        },
        "tweet": {
            "texts": 1,
            "errors": 1,
            "error_rate": 1.0,
            "share_of_errors": 1 / 3,
        },
    }


def test_confidence_table(frame):
    table = confidence_table(frame, bins=[0.0, 0.75, 1.0])

    assert table["texts"].tolist() == [2, 2]
    assert table["accuracy"].tolist() == [0.0, 0.5]


def test_plot_confusion_by_source_has_three_panels(frame):
    figure = plot_confusion_by_source(frame)

    titles = [axis.get_title() for axis in figure.axes if axis.get_title()]
    assert titles == ["All texts (4)", "News (3)", "Tweets (cashtag) (1)"]


def test_plot_confidence(frame):
    figure = plot_confidence(frame)

    assert figure.axes[0].get_title() == "Model confidence"


def test_error_examples_per_kind():
    frame = pd.DataFrame(
        {
            "raw_text": ["a", "b", "c", "d", "e"],
            "label": ["negative"] * 3 + ["neutral", "positive"],
            "predicted": ["positive"] * 3 + ["positive", "positive"],
            "confidence": [0.9, 0.5, 0.7, 0.6, 0.8],
            "correct": [False, False, False, False, True],
            "source": ["news"] * 5,
        }
    )

    examples = error_examples(frame, n=1)

    assert examples[["error", "kind", "raw_text"]].values.tolist() == [
        ["negative → positive", "confident", "a"],
        ["negative → positive", "borderline", "b"],
        ["neutral → positive", "confident", "d"],
    ]


def test_failure_masks():
    frame = pd.DataFrame(
        {
            "text": [
                "$fio won't stay down",
                "sales were down 14% and up 27%",
                "profit rose",
            ],
            "source": ["tweet", "news", "news"],
            "words": [4, 7, 2],
        }
    )

    masks = failure_masks(frame)

    assert {name: mask.tolist() for name, mask in masks.items()} == {
        "tweet": [True, False, False],
        "negation": [True, False, False],
        "down words": [True, True, False],
        "up and down words": [False, True, False],
        "number or %": [False, True, False],
        "short, 6 words or less": [True, False, True],
    }


def test_failure_table():
    frame = pd.DataFrame({"correct": [False, False, True, True]})
    masks = {"case": pd.Series([True, True, True, False])}

    table = failure_table(frame, masks)

    assert table.loc["case"].to_dict() == {
        "texts": 3,
        "errors": 2,
        "error_rate": 2 / 3,
        "error_rate_without": 0.0,
    }


def test_ticker_agreement():
    train = pd.DataFrame(
        {
            "text": ["$tsla down", "$tsla falls", "$aapl up"],
            "label": ["negative", "negative", "positive"],
        }
    )
    test = pd.DataFrame(
        {
            "text": ["$tsla up big", "$aapl up", "$xyz up", "profit rose"],
            "label": ["positive", "positive", "positive", "positive"],
        }
    )

    assert ticker_agreement(test, train).tolist() == [
        "disagrees",
        "agrees",
        "unseen ticker",
        "no ticker",
    ]


def test_word_label_shares():
    train = pd.DataFrame(
        {
            "text": ["net loss narrowed", "loss grew", "profit rose"],
            "label": ["positive", "negative", "positive"],
        }
    )

    table = word_label_shares(
        train, ["loss", "fell"], ("negative", "neutral", "positive")
    )

    assert table.loc["loss"].to_dict() == {
        "texts": 2,
        "negative": 0.5,
        "neutral": 0,
        "positive": 0.5,
    }
    assert table.loc["fell", "texts"] == 0
