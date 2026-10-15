"""Inference: the saved best model on new texts.

    from itg_classification.predict import load_model, predict
    predictions = predict(["Sales rose 20%"], load_model())

Input: a string or a list of strings, raw. Output: one `Prediction` per
text, in the same order. Command line: `itg-classification --help`.
"""

import argparse
import json
import re
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.pipeline import Pipeline

from itg_classification.config import (
    BEST_MODEL_CARD_PATH,
    BEST_MODEL_PATH,
    LABELS,
    LOW_CONFIDENCE,
)
from itg_classification.metrics import load_metrics
from itg_classification.preprocess import (
    LETTER,
    decode_labels,
    preprocess_text,
)


class ModelError(Exception):
    """The saved model cannot be used with this code."""


@dataclass(frozen=True)
class LoadedModel:
    pipeline: Pipeline
    card: dict


def load_model(
    model_path: Path = BEST_MODEL_PATH, card_path: Path = BEST_MODEL_CARD_PATH
) -> LoadedModel:
    """Loads the saved pipeline and checks it against its model card.

    Fails with a clear message instead of predicting garbage: when the
    files are missing, when the classes differ from the ones this code
    knows, or when scikit-learn differs from the one the model was
    trained with (pickled models are not portable between versions).
    """
    for path in (model_path, card_path):
        if not path.exists():
            raise ModelError(
                f"{path} not found, train the model with `make train-improved`"
            )
    card = load_metrics(card_path)
    trained_with = card["sklearn_version"]
    if trained_with != sklearn.__version__:
        raise ModelError(
            f"The model was trained with scikit-learn {trained_with}, "
            f"but {sklearn.__version__} is installed: retrain it"
        )
    pipeline = joblib.load(model_path)
    classes = decode_labels(pipeline.classes_)
    if classes != list(LABELS) or card["labels"] != list(LABELS):
        raise ModelError(
            f"The model predicts {classes}, the code expects {list(LABELS)}"
        )
    return LoadedModel(pipeline, card)


@dataclass(frozen=True)
class Prediction:
    """The answer for one text.

    `label` and `confidence` are None when the text cannot be
    classified, then `error` says why.
    """

    text: str
    label: str | None
    confidence: float | None
    probabilities: dict[str, float]
    low_confidence: bool
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


EMPTY_ERROR = "no words to classify after preprocessing"


def predict(
    texts: str | Sequence[str],
    model: LoadedModel | None = None,
    threshold: float = LOW_CONFIDENCE,
) -> list[Prediction]:
    """Classifies new texts with the saved model, without retraining.

    Every text goes through the same `preprocess_text` as the training
    data. A text without a single letter after it, like a bare link or
    only numbers, gets no label: such texts were dropped from training
    too. One prediction per input text, in the same order.
    """
    if isinstance(texts, str):
        texts = [texts]
    for text in texts:
        if not isinstance(text, str):
            raise TypeError(
                f"Texts must be strings, got {type(text).__name__}"
            )
    model = model or load_model()
    cleaned = [preprocess_text(text) for text in texts]
    filled = [
        index for index, text in enumerate(cleaned) if re.search(LETTER, text)
    ]
    probabilities = (
        model.pipeline.predict_proba([cleaned[index] for index in filled])
        if filled
        else []
    )
    by_index = dict(zip(filled, probabilities, strict=True))

    predictions = []
    for index, text in enumerate(texts):
        if index not in by_index:
            predictions.append(
                Prediction(text, None, None, {}, False, error=EMPTY_ERROR)
            )
            continue
        row = by_index[index]
        best = int(row.argmax())
        confidence = float(row[best])
        predictions.append(
            Prediction(
                text=text,
                label=LABELS[best],
                confidence=confidence,
                probabilities={
                    label: float(value)
                    for label, value in zip(LABELS, row, strict=True)
                },
                low_confidence=confidence < threshold,
            )
        )
    return predictions


def predictions_table(predictions: list[Prediction]) -> pd.DataFrame:
    """Predictions as flat rows, one probability column per class."""
    return pd.DataFrame(
        [
            {
                "text": p.text,
                "label": p.label,
                "confidence": p.confidence,
                **{
                    f"p_{label}": p.probabilities.get(label)
                    for label in LABELS
                },
                "low_confidence": p.low_confidence,
                "error": p.error,
            }
            for p in predictions
        ]
    )


def format_prediction(prediction: Prediction) -> str:
    """One line for a person: label, confidence and the text."""
    if prediction.label is None:
        return f"{'-':8}  {'-':>4}  {prediction.text}  ({prediction.error})"
    flag = "  (low confidence)" if prediction.low_confidence else ""
    return (
        f"{prediction.label:8}  {prediction.confidence:.2f}  "
        f"{prediction.text}{flag}"
    )


def predict_file(
    input_path: Path,
    output_path: Path,
    model: LoadedModel,
    column: str = "text",
    threshold: float = LOW_CONFIDENCE,
) -> pd.DataFrame:
    """Classifies the `column` of a CSV and saves the predictions."""
    frame = pd.read_csv(input_path, keep_default_na=False)
    if column not in frame.columns:
        raise ValueError(f"{input_path} has no column {column!r}")
    texts = frame[column].astype(str).tolist()
    table = predictions_table(predict(texts, model, threshold))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_path, index=False)
    return table


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="itg-classification",
        description="Sentiment of financial texts: negative, neutral or "
        "positive, with the saved best model.",
    )
    parser.add_argument("texts", nargs="*", help="texts to classify")
    parser.add_argument(
        "--input", type=Path, help="CSV file with the texts to classify"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="where to save the predictions for --input, as CSV",
    )
    parser.add_argument(
        "--column", default="text", help="text column of --input"
    )
    parser.add_argument(
        "--json", action="store_true", help="print JSON lines for texts"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=LOW_CONFIDENCE,
        help="flag predictions below this confidence (default: %(default)s)",
    )
    args = parser.parse_args(argv)
    if bool(args.input) != bool(args.output):
        parser.error("--input and --output go together")
    if not args.texts and not args.input:
        parser.error("give texts to classify or --input and --output")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        model = load_model()
        if args.input:
            table = predict_file(
                args.input, args.output, model, args.column, args.threshold
            )
            print(f"Saved {len(table)} predictions to {args.output}")
        for prediction in predict(args.texts, model, args.threshold):
            if args.json:
                print(json.dumps(prediction.to_dict()))
            else:
                print(format_prediction(prediction))
    except (ModelError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
