import pandas as pd

from itg_classification.features import (
    NEGATION_TOKEN_PATTERN,
    build_vectorizer,
    char_vectorizer,
    top_features,
    word_char_vectorizer,
    word_vectorizer,
)
from itg_classification.train_baseline import train_baseline


def test_build_vectorizer_uses_words_and_bigrams():
    analyzer = build_vectorizer().build_analyzer()

    assert analyzer("profit fell sharply") == [
        "profit",
        "fell",
        "sharply",
        "profit fell",
        "fell sharply",
    ]


def test_build_vectorizer_keeps_stop_words():
    analyzer = build_vectorizer().build_analyzer()

    assert "not" in analyzer("sales did not grow")


def test_build_vectorizer_drops_rare_ngrams():
    texts = ["profit fell", "profit rose", "sales fell"]

    vectorizer = build_vectorizer().fit(texts)

    assert sorted(vectorizer.vocabulary_) == ["fell", "profit"]


def test_build_vectorizer_returns_a_fresh_object():
    assert build_vectorizer() is not build_vectorizer()


def test_top_features_per_class():
    texts = {
        0: ["profit fell", "sales fell", "loss fell"],
        1: ["the report", "the meeting", "the board"],
        2: ["profit rose", "sales rose", "loss rose"],
    }
    rows = [(text, label) for label, group in texts.items() for text in group]
    train = pd.DataFrame(rows * 2, columns=["text", "label_id"])

    table = top_features(train_baseline(train), n=1)

    assert table.to_dict("list") == {
        "negative": ["fell"],
        "neutral": ["the"],
        "positive": ["rose"],
    }


def test_word_vectorizer_ngram_range():
    analyzer = word_vectorizer(ngram_range=(1, 3)).build_analyzer()

    assert "profit fell sharply" in analyzer("profit fell sharply")


def test_negation_token_pattern_keeps_negations():
    analyzer = word_vectorizer(
        ngram_range=(1, 1), token_pattern=NEGATION_TOKEN_PATTERN
    ).build_analyzer()

    assert analyzer("they can't and didn't grow, benefon's sales") == [
        "they",
        "can't",
        "and",
        "didn't",
        "grow",
        "benefon",
        "sales",
    ]


def test_char_vectorizer_shares_ngrams_between_word_forms():
    analyzer = char_vectorizer().build_analyzer()

    shared = set(analyzer("falls")) & set(analyzer("falling"))

    assert {" fa", " fal", " fall"} <= shared


def test_word_char_vectorizer_joins_both_feature_sets():
    texts = ["profit fell", "profit rose", "sales fell", "sales rose"]

    names = word_char_vectorizer().fit(texts).get_feature_names_out()

    assert "words__profit" in names
    assert "chars__ fel" in names
