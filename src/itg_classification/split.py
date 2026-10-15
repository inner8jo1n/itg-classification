"""Stratified train/test split, saved so every model sees one test.

Run: `make split`.
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from itg_classification.config import (
    PROCESSED_DATA_PATH,
    RANDOM_STATE,
    TEST_PATH,
    TEST_SIZE,
    TRAIN_PATH,
)


def split_dataset(
    frame: pd.DataFrame,
    test_size: float = TEST_SIZE,
    seed: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified train/test split, so both keep the class shares."""
    train, test = train_test_split(
        frame,
        test_size=test_size,
        random_state=seed,
        stratify=frame["label_id"],
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)


def load_split(path: Path) -> pd.DataFrame:
    # Texts like "null" or "nan" must stay strings, not become NaN
    return pd.read_csv(path, keep_default_na=False)


def main(
    input_path: Path = PROCESSED_DATA_PATH,
    train_path: Path = TRAIN_PATH,
    test_path: Path = TEST_PATH,
) -> None:
    frame = load_split(input_path)
    train, test = split_dataset(frame)
    for part, path in [(train, train_path), (test, test_path)]:
        path.parent.mkdir(parents=True, exist_ok=True)
        part.to_csv(path, index=False)
    shares = pd.DataFrame(
        {
            name: part["label"].value_counts(normalize=True)
            for name, part in [
                ("all", frame),
                ("train", train),
                ("test", test),
            ]
        }
    )
    print(f"Saved {len(train)} train rows to {train_path}")
    print(f"Saved {len(test)} test rows to {test_path}")
    print(shares.round(3).to_string())


if __name__ == "__main__":
    main()
