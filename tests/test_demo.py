import pandas as pd
import pytest

from itg_classification.config import DEMO_PATH
from itg_classification.demo import format_demo, run_demo
from itg_classification.predict import LoadedModel
from itg_classification.train_baseline import train_baseline


@pytest.fixture
def model() -> LoadedModel:
    texts = {
        0: ["profit fell", "sales fell sharply", "net loss grew"],
        1: ["the company said", "the report is out", "the meeting is today"],
        2: ["profit rose", "sales rose sharply", "net profit grew"],
    }
    rows = [(text, label) for label, group in texts.items() for text in group]
    train = pd.DataFrame(rows * 3, columns=["text", "label_id"])
    return LoadedModel(train_baseline(train), {})


@pytest.fixture
def examples(tmp_path):
    path = tmp_path / "examples.csv"
    pd.DataFrame(
        {
            "kind": ["news", "news", "no words"],
            "expected": ["positive", "neutral", ""],
            "text": ["Profit rose", "Profit fell", "http://t.co/x"],
        }
    ).to_csv(path, index=False)
    return path


def test_run_demo(model, examples):
    results = run_demo(model, examples)

    assert results["predicted"].tolist() == ["positive", "negative", ""]
    assert results["correct"].tolist() == [True, False, True]


def test_format_demo(model, examples):
    text = format_demo(run_demo(model, examples))

    lines = text.splitlines()
    assert lines[1].startswith("✓ news")
    assert lines[2].startswith("✗ news")
    assert "http://t.co/x" in lines[3]
    assert lines[-1] == "2 of 3 as expected"


def test_demo_examples_file_is_valid():
    examples = pd.read_csv(DEMO_PATH, keep_default_na=False)

    assert examples.columns.tolist() == ["kind", "expected", "text"]
    assert set(examples["expected"]) <= {"negative", "neutral", "positive", ""}
    assert examples["text"].str.len().gt(0).all()


def test_demo_runs_on_the_examples_file(model):
    results = run_demo(model)

    assert len(results) == len(pd.read_csv(DEMO_PATH))
