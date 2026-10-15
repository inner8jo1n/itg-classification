"""Text vectorizers: word and character TF-IDF, and top features."""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline

from itg_classification.config import ID_TO_LABEL


def build_vectorizer() -> TfidfVectorizer:
    """Baseline TF-IDF over words and word pairs.

    Bigrams keep short phrases like "profit fell" or "net loss".
    `min_df=2` drops n-grams seen in one text only: it cuts the
    vocabulary from 49k to 11k without losing anything a model could
    generalize from. Stop words stay, because "not", "down" and "up"
    are on sklearn's list and they carry sentiment.
    """
    return TfidfVectorizer(ngram_range=(1, 2), min_df=2)


# The default pattern turns "can't" into "can" and loses the negation
NEGATION_TOKEN_PATTERN = r"(?u)\b\w+n't\b|\b\w\w+\b"


def word_vectorizer(
    ngram_range: tuple[int, int] = (1, 2),
    token_pattern: str = r"(?u)\b\w\w+\b",
) -> TfidfVectorizer:
    """The baseline vectorizer with other n-grams or tokens."""
    return TfidfVectorizer(
        ngram_range=ngram_range, min_df=2, token_pattern=token_pattern
    )


def char_vectorizer() -> TfidfVectorizer:
    """Character n-grams of 2 to 5 inside word boundaries.

    "fell", "falls" and "falling" share "fal", and a typo or a broken
    encoding changes only a few n-grams instead of the whole word.
    `sublinear_tf` dampens n-grams repeated many times in one text.
    """
    return TfidfVectorizer(
        analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True
    )


def word_char_vectorizer() -> FeatureUnion:
    """Word and character features side by side."""
    return FeatureUnion(
        [("words", build_vectorizer()), ("chars", char_vectorizer())]
    )


def top_features(pipeline: Pipeline, n: int = 15) -> pd.DataFrame:
    """The n-grams with the largest weights for every class.

    Works for a fitted TF-IDF + linear model pipeline: a large positive
    weight means the n-gram pushes a text towards that class.
    """
    names = pipeline.named_steps["tfidf"].get_feature_names_out()
    weights = pipeline.named_steps["model"].coef_
    return pd.DataFrame(
        {
            ID_TO_LABEL[int(label_id)]: names[np.argsort(row)[::-1][:n]]
            for label_id, row in zip(pipeline.classes_, weights, strict=True)
        }
    )
