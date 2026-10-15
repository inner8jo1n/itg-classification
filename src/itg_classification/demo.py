"""Demo: the saved best model on hand-written examples.

Run: `make demo`. The examples in demo/examples.csv cover news, tweets
and the known hard cases, each with the label a person would give.
"""

from pathlib import Path

import pandas as pd

from itg_classification.config import DEMO_PATH
from itg_classification.predict import (
    LoadedModel,
    load_model,
    predict,
    predictions_table,
)


def run_demo(model: LoadedModel, path: Path = DEMO_PATH) -> pd.DataFrame:
    """Predictions for every example next to its expected label."""
    examples = pd.read_csv(path, keep_default_na=False)
    table = predictions_table(predict(examples["text"].tolist(), model))
    return examples.assign(
        predicted=table["label"].fillna(""),
        confidence=table["confidence"],
        low_confidence=table["low_confidence"],
        correct=table["label"].fillna("") == examples["expected"],
    )


def format_demo(results: pd.DataFrame) -> str:
    """A table for the terminal and the number of right answers."""
    lines = [
        f"{'':2}{'kind':22}{'expected':10}{'predicted':11}{'conf.':7}text"
    ]
    for row in results.itertuples():
        mark = "✓" if row.correct else "✗"
        confidence = "" if pd.isna(row.confidence) else f"{row.confidence:.2f}"
        flag = " (low confidence)" if row.low_confidence else ""
        lines.append(
            f"{mark} {row.kind:22}{row.expected or '-':10}"
            f"{row.predicted or '-':11}{confidence:7}{row.text}{flag}"
        )
    right = int(results["correct"].sum())
    lines.append(f"\n{right} of {len(results)} as expected")
    return "\n".join(lines)


def main() -> None:
    print(format_demo(run_demo(load_model())))


if __name__ == "__main__":
    main()
