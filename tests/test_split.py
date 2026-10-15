import pandas as pd
import pytest

from itg_classification.split import load_split, main, split_dataset


@pytest.fixture
def frame() -> pd.DataFrame:
    labels = ["negative"] * 10 + ["neutral"] * 50 + ["positive"] * 40
    return pd.DataFrame(
        {
            "text": [f"text {index}" for index in range(len(labels))],
            "label": labels,
            "label_id": [
                {"negative": 0, "neutral": 1, "positive": 2}[label]
                for label in labels
            ],
        }
    )


def test_split_dataset_sizes(frame):
    train, test = split_dataset(frame, test_size=0.2)

    assert (len(train), len(test)) == (80, 20)


def test_split_dataset_keeps_class_shares(frame):
    train, test = split_dataset(frame, test_size=0.2)

    assert test["label"].value_counts().to_dict() == {
        "neutral": 10,
        "positive": 8,
        "negative": 2,
    }
    assert train["label"].value_counts().to_dict() == {
        "neutral": 40,
        "positive": 32,
        "negative": 8,
    }


def test_split_dataset_has_no_shared_texts(frame):
    train, test = split_dataset(frame)

    assert set(train["text"]).isdisjoint(test["text"])


def test_split_dataset_is_reproducible(frame):
    first, _ = split_dataset(frame, seed=0)
    second, _ = split_dataset(frame, seed=0)

    pd.testing.assert_frame_equal(first, second)


def test_load_split_keeps_null_like_texts(tmp_path):
    path = tmp_path / "part.csv"
    pd.DataFrame({"text": ["null", "nan"], "label_id": [1, 1]}).to_csv(
        path, index=False
    )

    assert load_split(path)["text"].tolist() == ["null", "nan"]


def test_main_saves_both_parts(frame, tmp_path, capsys):
    input_path = tmp_path / "processed.csv"
    frame.to_csv(input_path, index=False)

    main(input_path, tmp_path / "train.csv", tmp_path / "test.csv")

    assert len(pd.read_csv(tmp_path / "train.csv")) == 80
    assert len(pd.read_csv(tmp_path / "test.csv")) == 20
    assert "Saved 20 test rows" in capsys.readouterr().out
