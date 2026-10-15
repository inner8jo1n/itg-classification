"""Text cleaning and normalization, row filtering, label encoding.

Run: `make preprocess`. `preprocess_text` is shared by training and
inference.
"""

import re
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from itg_classification.config import (
    ID_TO_LABEL,
    LABEL_TO_ID,
    PROCESSED_DATA_PATH,
    RAW_DATA_PATH,
)
from itg_classification.eda import conflicting_texts
from itg_classification.load_data import load_dataset

URL = re.compile(r"https?://\S+|\bwww\.\S+")
RETWEET = re.compile(r"^RT\s+")
MENTION = re.compile(r"@\w+")
HASHTAG = re.compile(r"#(\w+)")
# News sentences use LaTeX-style quotes: ``Unikko''
LATEX_QUOTES = re.compile(r"``|''")
WHITESPACE = re.compile(r"\s+")

# News sentences are tokenized: "Ruukki 's", "2009 ,", "30 %", "( ADP )"
DETACHED_CONTRACTION = re.compile(r" (n't|'(?:s|re|m|ve|ll|d))\b")
SPACE_BEFORE_PUNCT = re.compile(r" ([.,;:!?%)]+)(?=\s|$)")
SPACE_AFTER_BRACKET = re.compile(r"\( ")
DOUBLE_QUOTE = re.compile(r'"')

LETTER = r"[a-z]"


def clean_text(text: str) -> str:
    """Removes the parts of a text that carry no sentiment.

    Drops links, the retweet marker and @mentions, keeps the word of a
    hashtag and turns LaTeX-style quotes into plain ones. Cashtags like
    `$TSLA` stay, they name the company the text is about.
    """
    text = URL.sub(" ", text)
    text = RETWEET.sub("", text)
    text = MENTION.sub(" ", text)
    text = HASHTAG.sub(r"\1", text)
    text = LATEX_QUOTES.sub('"', text)
    return WHITESPACE.sub(" ", text).strip()


def normalize_text(text: str) -> str:
    """Brings news and tweets to one lowercase, untokenized style.

    New texts at inference are written normally, like "Ruukki's" and
    "30%", so the training texts must look the same. Quotes are dropped
    because they carry no sentiment and their spacing is inconsistent.
    """
    text = text.lower()
    # Quotes go first: dropping them later would detach punctuation again
    text = DOUBLE_QUOTE.sub(" ", text)
    text = WHITESPACE.sub(" ", text).strip()
    text = DETACHED_CONTRACTION.sub(r"\1", text)
    text = SPACE_BEFORE_PUNCT.sub(r"\1", text)
    return SPACE_AFTER_BRACKET.sub("(", text)


def drop_bad_rows(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Removes rows that would hurt training, on already cleaned texts.

    In order: empty texts, texts without a single letter, every copy of
    a text that has more than one label (we cannot tell which label is
    right), then repeated rows. Returns the kept rows and how many rows
    every rule removed.
    """
    removed = {}

    def drop(name: str, mask: pd.Series) -> None:
        nonlocal frame
        removed[name] = int(mask.sum())
        frame = frame[~mask]

    drop("empty", frame["text"].eq(""))
    drop("no_letters", ~frame["text"].str.contains(LETTER))
    drop(
        "conflicting_label",
        frame["text"].isin(conflicting_texts(frame).index),
    )
    drop("duplicate", frame.duplicated(subset=["text", "label"]))
    return frame.reset_index(drop=True), pd.Series(removed, name="removed")


def encode_labels(labels: pd.Series) -> pd.Series:
    """Label names to integer ids, failing on an unknown name."""
    ids = labels.map(LABEL_TO_ID)
    unknown = labels[ids.isna()].unique().tolist()
    if unknown:
        raise ValueError(f"Unknown labels {unknown}")
    return ids.astype(int)


def decode_labels(ids: Iterable[int]) -> list[str]:
    """Integer ids, like model predictions, back to label names."""
    return [ID_TO_LABEL[int(index)] for index in ids]


def preprocess_text(text: str) -> str:
    """The full text transformation, the same for training and inference."""
    return normalize_text(clean_text(text))


def preprocess_dataset(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Turns the loaded dataset into rows ready for training.

    Keeps the original text in `raw_text` for the error analysis, puts
    the preprocessed one in `text` and adds the integer `label_id`.
    Returns the rows and how many rows every filter removed.
    """
    frame = frame.assign(
        raw_text=frame["text"], text=frame["text"].map(preprocess_text)
    )
    frame, removed = drop_bad_rows(frame)
    frame = frame.assign(label_id=encode_labels(frame["label"]))
    return frame[["raw_text", "text", "label", "label_id"]], removed


def main(
    input_path: Path = RAW_DATA_PATH, output_path: Path = PROCESSED_DATA_PATH
) -> None:
    raw = load_dataset(input_path)
    processed, removed = preprocess_dataset(raw)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed.to_csv(output_path, index=False)
    print(f"Loaded {len(raw)} rows from {input_path}")
    for rule, count in removed.items():
        print(f"  removed {count:5} {rule}")
    print(f"Saved {len(processed)} rows to {output_path}")


if __name__ == "__main__":
    main()
