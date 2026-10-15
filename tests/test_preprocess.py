import numpy as np
import pandas as pd
import pytest

from itg_classification.load_data import load_dataset
from itg_classification.preprocess import (
    clean_text,
    decode_labels,
    drop_bad_rows,
    encode_labels,
    main,
    normalize_text,
    preprocess_dataset,
    preprocess_text,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (
            "$FB Big Money Is Pouring Into Facebook;http://stks.co/sOwG",
            "$FB Big Money Is Pouring Into Facebook;",
        ),
        (
            "$SAP Q1 disappoints https://t.co/jNDphllzq5 today",
            "$SAP Q1 disappoints today",
        ),
        ("Results at www.cargotec.com .", "Results at ."),
        (
            "RT @bman1 $LNG looks primed also",
            "$LNG looks primed also",
        ),
        ("short it @cnnbrk now", "short it now"),
        (
            "disappoints as #software licenses down",
            "disappoints as software licenses down",
        ),
        (
            "Marimekko 's `` Unikko '' floral pattern",
            'Marimekko \'s " Unikko " floral pattern',
        ),
        (
            "Whats up with $LULU?  Numbers looked good",
            "Whats up with $LULU? Numbers looked good",
        ),
        ("  Profit fell .  ", "Profit fell ."),
        ("ART sales rose", "ART sales rose"),
        ("http://stks.co/sOwG", ""),
    ],
)
def test_clean_text(raw, expected):
    assert clean_text(raw) == expected


def test_clean_text_is_idempotent():
    text = "RT @user $TSLA `` big '' #news http://t.co/x"

    assert clean_text(clean_text(text)) == clean_text(text)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Profit FELL", "profit fell"),
        ("Ruukki 's order book", "ruukki's order book"),
        (
            "Financial details were n't disclosed .",
            "financial details weren't disclosed.",
        ),
        ("they 're all for sale", "they're all for sale"),
        ("in 2009 , sales rose 30 % .", "in 2009, sales rose 30%."),
        ("( ADP News ) - Feb 11 , 2009", "(adp news) - feb 11, 2009"),
        ('gave " buy " - " hold "', "gave buy - hold"),
        ("down $1.50 to $2.50", "down $1.50 to $2.50"),
        ("Inc. iPhone", "inc. iphone"),
        ("Wait ... what ?", "wait... what?"),
        ("Q1 's", "q1's"),
        ("It 's not .", "it's not."),
        ("'best' deal", "'best' deal"),
        ('gave " hold " , three', "gave hold, three"),
    ],
)
def test_normalize_text(raw, expected):
    assert normalize_text(raw) == expected


def test_normalize_text_is_idempotent():
    text = "( ADP ) Ruukki 's sales were n't up 30 % , \" he \" said ."

    assert normalize_text(normalize_text(text)) == normalize_text(text)


def test_drop_bad_rows():
    frame = pd.DataFrame(
        {
            "text": [
                "profit fell.",
                "",
                "24.62 $",
                "net profit decreased.",
                "net profit decreased.",
                "sales rose.",
                "sales rose.",
                "sales rose.",
            ],
            "label": [
                "negative",
                "neutral",
                "positive",
                "negative",
                "neutral",
                "positive",
                "positive",
                "positive",
            ],
        }
    )

    kept, removed = drop_bad_rows(frame)

    assert kept.to_dict("list") == {
        "text": ["profit fell.", "sales rose."],
        "label": ["negative", "positive"],
    }
    assert removed.to_dict() == {
        "empty": 1,
        "no_letters": 1,
        "conflicting_label": 2,
        "duplicate": 2,
    }


def test_drop_bad_rows_keeps_clean_frame_unchanged():
    frame = pd.DataFrame(
        {
            "text": ["profit fell.", "sales rose."],
            "label": ["negative", "positive"],
        }
    )

    kept, removed = drop_bad_rows(frame)

    pd.testing.assert_frame_equal(kept, frame)
    assert removed.sum() == 0


def test_encode_labels():
    labels = pd.Series(["positive", "negative", "neutral", "negative"])

    assert encode_labels(labels).tolist() == [2, 0, 1, 0]


def test_encode_labels_rejects_unknown_label():
    with pytest.raises(ValueError, match="bullish"):
        encode_labels(pd.Series(["neutral", "bullish"]))


def test_decode_labels_accepts_model_predictions():
    assert decode_labels(np.array([2, 0, 1])) == [
        "positive",
        "negative",
        "neutral",
    ]


def test_decode_labels_reverts_encode_labels():
    labels = pd.Series(["neutral", "positive", "negative"])

    assert decode_labels(encode_labels(labels)) == labels.tolist()


def test_preprocess_text_is_idempotent_on_raw_dataset():
    texts = load_dataset()["text"].map(preprocess_text)

    changed = texts[texts.map(preprocess_text) != texts]
    assert changed.tolist() == []


def test_preprocess_text_cleans_then_normalizes():
    text = "RT @user `` Ruukki 's `` sales rose 30 % http://t.co/x"

    assert preprocess_text(text) == "ruukki's sales rose 30%"


@pytest.fixture
def raw_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "text": [
                "Profit FELL .",
                "http://stks.co/sOwG",
                "Sales rose http://t.co/a",
                "Sales rose http://t.co/b",
                "Net profit decreased .",
                "Net profit decreased .",
            ],
            "label": [
                "negative",
                "neutral",
                "positive",
                "positive",
                "negative",
                "neutral",
            ],
        }
    )


def test_preprocess_dataset(raw_frame):
    processed, removed = preprocess_dataset(raw_frame)

    assert processed.to_dict("list") == {
        "raw_text": ["Profit FELL .", "Sales rose http://t.co/a"],
        "text": ["profit fell.", "sales rose"],
        "label": ["negative", "positive"],
        "label_id": [0, 2],
    }
    assert removed.to_dict() == {
        "empty": 1,
        "no_letters": 0,
        "conflicting_label": 2,
        "duplicate": 1,
    }


def test_main_saves_processed_csv(raw_frame, tmp_path, capsys):
    raw_path = tmp_path / "raw.csv"
    output_path = tmp_path / "processed" / "data.csv"
    raw_frame.rename(
        columns={"text": "Sentence", "label": "Sentiment"}
    ).to_csv(raw_path, index=False)

    main(raw_path, output_path)

    saved = pd.read_csv(output_path)
    assert saved.columns.tolist() == ["raw_text", "text", "label", "label_id"]
    assert len(saved) == 2
    assert "Saved 2 rows" in capsys.readouterr().out
