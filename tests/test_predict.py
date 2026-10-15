import json

import joblib
import pandas as pd
import pytest
import sklearn

from itg_classification import predict as predict_module
from itg_classification.config import TEST_PATH, TRAIN_PATH
from itg_classification.metrics import save_metrics
from itg_classification.predict import (
    EMPTY_ERROR,
    LoadedModel,
    ModelError,
    Prediction,
    format_prediction,
    load_model,
    main,
    predict,
    predict_file,
    predictions_table,
)
from itg_classification.preprocess import decode_labels, preprocess_text
from itg_classification.split import load_split
from itg_classification.train_baseline import train_baseline
from itg_classification.train_improved import word_char_logreg


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
def card() -> dict:
    return {
        "name": "Test model",
        "labels": ["negative", "neutral", "positive"],
        "sklearn_version": sklearn.__version__,
    }


@pytest.fixture
def saved(train, card, tmp_path):
    """Paths of a saved model and its card."""
    model_path, card_path = tmp_path / "model.joblib", tmp_path / "model.json"
    joblib.dump(train_baseline(train), model_path)
    save_metrics(card, card_path)
    return model_path, card_path


def test_load_model(saved, card):
    loaded = load_model(*saved)

    assert isinstance(loaded, LoadedModel)
    assert loaded.card == card
    assert loaded.pipeline.predict(["profit fell"]).tolist() == [0]


def test_load_model_without_files(tmp_path):
    with pytest.raises(ModelError, match="make train-improved"):
        load_model(tmp_path / "model.joblib", tmp_path / "model.json")


def test_load_model_rejects_other_sklearn(saved, card):
    model_path, card_path = saved
    save_metrics({**card, "sklearn_version": "0.1"}, card_path)

    with pytest.raises(ModelError, match="scikit-learn 0.1"):
        load_model(model_path, card_path)


def test_load_model_rejects_other_classes(saved, card, train, tmp_path):
    model_path, card_path = saved
    two_classes = train[train["label_id"] != 1]
    joblib.dump(train_baseline(two_classes), model_path)

    with pytest.raises(ModelError, match="expects"):
        load_model(model_path, card_path)


@pytest.fixture
def model(saved) -> LoadedModel:
    return load_model(*saved)


def test_predict_one_text(model):
    [prediction] = predict("Profit FELL", model)

    assert prediction.text == "Profit FELL"
    assert prediction.label == "negative"
    assert prediction.confidence == max(prediction.probabilities.values())
    assert list(prediction.probabilities) == [
        "negative",
        "neutral",
        "positive",
    ]
    assert sum(prediction.probabilities.values()) == pytest.approx(1)
    assert prediction.error is None


def test_predict_keeps_order(model):
    texts = ["Profit rose", "The company said", "Profit fell"]

    labels = [prediction.label for prediction in predict(texts, model)]

    assert labels == ["positive", "neutral", "negative"]


def test_predict_flags_low_confidence(model):
    [sure] = predict("profit fell", model, threshold=0.0)
    [unsure] = predict("profit fell", model, threshold=1.0)

    assert not sure.low_confidence
    assert unsure.low_confidence


def test_predict_does_not_classify_texts_without_words(model):
    predictions = predict(
        ["", "http://t.co/x", "12345 678", "Profit rose"], model
    )

    assert [p.label for p in predictions] == [None, None, None, "positive"]
    assert predictions[0] == Prediction(
        "", None, None, {}, False, error=EMPTY_ERROR
    )


def test_predict_only_empty_texts(model):
    assert [p.error for p in predict(["  "], model)] == [EMPTY_ERROR]


def test_predict_empty_list(model):
    assert predict([], model) == []


def test_predict_rejects_non_strings(model):
    with pytest.raises(TypeError, match="int"):
        predict(["Profit rose", 42], model)  # ty: ignore[invalid-argument-type]


def test_prediction_to_dict(model):
    [prediction] = predict("Profit rose", model)

    assert list(prediction.to_dict()) == [
        "text",
        "label",
        "confidence",
        "probabilities",
        "low_confidence",
        "error",
    ]


def test_predictions_table(model):
    table = predictions_table(predict(["Profit rose", ""], model))

    assert table.columns.tolist() == [
        "text",
        "label",
        "confidence",
        "p_negative",
        "p_neutral",
        "p_positive",
        "low_confidence",
        "error",
    ]
    assert table["label"].iloc[0] == "positive"
    assert table["label"].isna().tolist() == [False, True]
    assert table["error"].iloc[1] == EMPTY_ERROR


def test_format_prediction():
    sure = Prediction("Profit rose", "positive", 0.96, {}, False)
    unsure = Prediction("Hmm", "neutral", 0.41, {}, True)
    empty = Prediction("", None, None, {}, False, error=EMPTY_ERROR)

    assert format_prediction(sure) == "positive  0.96  Profit rose"
    assert format_prediction(unsure).endswith("Hmm  (low confidence)")
    assert format_prediction(empty).endswith(f"({EMPTY_ERROR})")


def test_predict_file(model, tmp_path):
    input_path = tmp_path / "news.csv"
    pd.DataFrame({"headline": ["Profit rose", "null"]}).to_csv(
        input_path, index=False
    )

    table = predict_file(
        input_path, tmp_path / "out" / "predictions.csv", model, "headline"
    )

    saved = pd.read_csv(
        tmp_path / "out" / "predictions.csv", keep_default_na=False
    )
    assert saved["text"].tolist() == ["Profit rose", "null"]
    assert len(table) == 2


def test_predict_file_rejects_missing_column(model, tmp_path):
    input_path = tmp_path / "news.csv"
    pd.DataFrame({"headline": ["Profit rose"]}).to_csv(input_path, index=False)

    with pytest.raises(ValueError, match="no column 'text'"):
        predict_file(input_path, tmp_path / "out.csv", model)


@pytest.fixture
def cli_model(model, monkeypatch):
    """Makes the command line use the small test model."""
    monkeypatch.setattr(predict_module, "load_model", lambda: model)
    return model


def test_main_prints_one_line_per_text(cli_model, capsys):
    code = main(["Profit rose", "Profit fell"])

    lines = capsys.readouterr().out.splitlines()
    assert code == 0
    assert [line.split()[0] for line in lines] == ["positive", "negative"]


def test_main_prints_json(cli_model, capsys):
    main(["Profit rose", "--json"])

    output = json.loads(capsys.readouterr().out)
    assert output["label"] == "positive"


def test_main_classifies_a_file(cli_model, tmp_path, capsys):
    input_path = tmp_path / "news.csv"
    pd.DataFrame({"text": ["Profit rose"]}).to_csv(input_path, index=False)

    code = main(
        ["--input", str(input_path), "--output", str(tmp_path / "o.csv")]
    )

    assert code == 0
    assert "Saved 1 predictions" in capsys.readouterr().out


def test_main_reports_model_errors(monkeypatch, capsys):
    def missing_model():
        raise ModelError("models/best_model.joblib not found")

    monkeypatch.setattr(predict_module, "load_model", missing_model)

    assert main(["Profit rose"]) == 1
    assert "Error: models/best_model.joblib" in capsys.readouterr().err


@pytest.mark.parametrize(
    "argv", [[], ["--input", "news.csv"], ["--output", "out.csv", "text"]]
)
def test_main_rejects_wrong_arguments(argv):
    with pytest.raises(SystemExit):
        main(argv)


def test_predict_preprocesses_every_text(model, monkeypatch):
    seen = []

    def spy(text: str) -> str:
        seen.append(text)
        return preprocess_text(text)

    monkeypatch.setattr(predict_module, "preprocess_text", spy)

    predict(["RT @x Profit ROSE http://t.co/a", "Profit fell"], model)

    assert seen == ["RT @x Profit ROSE http://t.co/a", "Profit fell"]


@pytest.fixture(scope="module")
def real_model() -> LoadedModel:
    """The best model setup trained on the real train split.

    Trained here, not loaded from models/, so the test does not depend
    on a file that is not in git.
    """
    train = load_split(TRAIN_PATH)
    pipeline = word_char_logreg().fit(train["text"], train["label_id"])
    return LoadedModel(pipeline, {})


def test_predict_on_raw_texts_matches_training_pipeline(real_model):
    """Raw test texts through `predict` get exactly the predictions the
    model gives on the texts preprocessed on day 2."""
    test = load_split(TEST_PATH)

    from_raw = [
        p.label for p in predict(test["raw_text"].tolist(), real_model)
    ]

    expected = decode_labels(real_model.pipeline.predict(test["text"]))
    assert from_raw == expected


def test_raw_texts_without_preprocessing_differ(real_model):
    """Skipping preprocessing changes what the model sees: the inputs
    would no longer match the ones it was trained on."""
    test = load_split(TEST_PATH)
    pipeline = real_model.pipeline

    unprocessed = pipeline.predict(test["raw_text"])
    processed = pipeline.predict(test["text"])

    assert (unprocessed != processed).sum() > 0
