"""Error analysis of the best model on the test set.

Run: `make error-analysis`. Writes results/errors.csv,
results/error_analysis.txt and two figures.
"""

import re
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.pipeline import Pipeline

from itg_classification.config import (
    BEST_MODEL_PATH,
    FIGURES_DIR,
    ID_TO_LABEL,
    LABELS,
    RESULTS_DIR,
    TEST_PATH,
    TRAIN_PATH,
)
from itg_classification.eda import TEXT_PATTERNS, text_lengths
from itg_classification.metrics import draw_confusion_matrix
from itg_classification.preprocess import encode_labels
from itg_classification.split import load_split


def predictions_frame(model: Pipeline, test: pd.DataFrame) -> pd.DataFrame:
    """Every test text with its prediction, probabilities and context.

    `source` tells tweets from news by a cashtag in the raw text. It is
    an approximation: a few headlines without a cashtag are tweets too.
    """
    probabilities = model.predict_proba(test["text"])
    names = [ID_TO_LABEL[int(label_id)] for label_id in model.classes_]
    frame = pd.DataFrame(
        {
            "raw_text": test["raw_text"],
            "text": test["text"],
            "label": test["label"],
            "predicted": [names[index] for index in probabilities.argmax(1)],
        }
    )
    for index, name in enumerate(names):
        frame[f"p_{name}"] = probabilities[:, index]
    frame["confidence"] = probabilities.max(axis=1)
    frame["correct"] = frame["label"] == frame["predicted"]
    frame["source"] = (
        test["raw_text"]
        .str.contains(TEXT_PATTERNS["cashtag"])
        .map({True: "tweet", False: "news"})
    )
    frame["words"] = text_lengths(test["text"])["words"]
    return frame


def errors(frame: pd.DataFrame) -> pd.DataFrame:
    """Wrong predictions only, the most confident first."""
    wrong = frame[~frame["correct"]]
    return wrong.sort_values("confidence", ascending=False)


def error_types(frame: pd.DataFrame) -> pd.DataFrame:
    """How many errors of every true → predicted kind, largest first."""
    wrong = errors(frame)
    counts = wrong.groupby(["label", "predicted"]).size()
    table = counts.rename("errors").reset_index()
    table["share_of_errors"] = table["errors"] / len(wrong)
    return table.sort_values("errors", ascending=False, ignore_index=True)


def error_examples(frame: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """The `n` most confident and `n` least confident errors of every
    true → predicted kind, with the original text.

    Confident errors show what the model learned wrong or where a label
    is doubtful, borderline ones show texts it cannot decide on.
    """
    parts = []
    for (label, predicted), group in errors(frame).groupby(
        ["label", "predicted"], sort=False
    ):
        for kind, rows in [
            ("confident", group.head(n)),
            ("borderline", group.tail(n).iloc[::-1]),
        ]:
            parts.append(
                rows.assign(error=f"{label} → {predicted}", kind=kind)
            )
    examples = pd.concat(parts).drop_duplicates(subset=["raw_text", "error"])
    columns = ["error", "kind", "confidence", "source", "raw_text"]
    return examples[columns].reset_index(drop=True)


def source_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Texts, errors and error rate of tweets and news."""
    grouped = frame.groupby("source")["correct"]
    table = pd.DataFrame(
        {"texts": grouped.size(), "errors": grouped.agg(lambda c: (~c).sum())}
    )
    table["error_rate"] = table["errors"] / table["texts"]
    table["share_of_errors"] = table["errors"] / table["errors"].sum()
    return table


def plot_confusion_by_source(
    frame: pd.DataFrame, path: Path | None = None
) -> plt.Figure:
    """Confusion matrices of all texts, news only and tweets only."""
    parts = {
        "All texts": frame,
        "News": frame[frame["source"] == "news"],
        "Tweets (cashtag)": frame[frame["source"] == "tweet"],
    }
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for index, (axis, (title, part)) in enumerate(
        zip(axes, parts.items(), strict=True)
    ):
        draw_confusion_matrix(
            axis,
            encode_labels(part["label"]),
            encode_labels(part["predicted"]),
            title=f"{title} ({len(part)})",
            colorbar=index == len(parts) - 1,
        )
    figure.tight_layout()
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=150)
    return figure


CONFIDENCE_BINS = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def confidence_table(
    frame: pd.DataFrame, bins: list[float] = CONFIDENCE_BINS
) -> pd.DataFrame:
    """Accuracy of the predictions in every confidence range."""
    ranges = pd.cut(frame["confidence"], bins)
    grouped = frame.groupby(ranges, observed=True)["correct"]
    return pd.DataFrame({"texts": grouped.size(), "accuracy": grouped.mean()})


def plot_confidence(
    frame: pd.DataFrame, path: Path | None = None
) -> plt.Figure:
    """Confidence of correct and wrong predictions, each class of
    outcome normalized separately."""
    figure, axis = plt.subplots(figsize=(7, 4))
    sns.histplot(
        data=frame.assign(
            outcome=frame["correct"].map({True: "correct", False: "error"})
        ),
        x="confidence",
        hue="outcome",
        hue_order=["correct", "error"],
        stat="density",
        common_norm=False,
        bins=20,
        element="step",
        ax=axis,
    )
    axis.set(
        title="Model confidence",
        xlabel="Probability of the predicted class",
        ylabel="Density",
    )
    figure.tight_layout()
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=150)
    return figure


# Hypotheses about where the model fails, as patterns on cleaned texts
NEGATION = r"\b(?:not|no|never|nothing|without|cannot)\b|n't\b"
UP_WORDS = (
    r"\b(?:up|rise[sn]?|rising|rose|increas\w*|gr[eo]w\w*|gain\w*|higher"
    r"|improv\w*|strong\w*|boost\w*)\b"
)
DOWN_WORDS = (
    r"\b(?:down|fall\w*|fell|decreas\w*|declin\w*|drop\w*|lower|loss\w*"
    r"|weak\w*|cut\w*)\b"
)


def failure_masks(frame: pd.DataFrame) -> dict[str, pd.Series]:
    """Which test texts fall into every suspected failure case."""
    text = frame["text"]
    up, down = text.str.contains(UP_WORDS), text.str.contains(DOWN_WORDS)
    return {
        "tweet": frame["source"] == "tweet",
        "negation": text.str.contains(NEGATION),
        "down words": down,
        "up and down words": up & down,
        "number or %": text.str.contains(r"\d"),
        "short, 6 words or less": frame["words"] <= 6,
    }


def failure_table(
    frame: pd.DataFrame, masks: dict[str, pd.Series]
) -> pd.DataFrame:
    """Error rate of texts with and without every case.

    A case matters when its error rate is clearly above the rest, and
    it matters a lot when it also covers many errors.
    """
    rows = {}
    for name, mask in masks.items():
        wrong = ~frame["correct"]
        rows[name] = {
            "texts": int(mask.sum()),
            "errors": int(wrong[mask].sum()),
            "error_rate": wrong[mask].mean(),
            "error_rate_without": wrong[~mask].mean(),
        }
    return pd.DataFrame(rows).T


CASHTAG = re.compile(r"\$([a-z]{1,6})\b")


def ticker_agreement(test: pd.DataFrame, train: pd.DataFrame) -> pd.Series:
    """Does the train label of a text's tickers match its true label?

    For every ticker the most common label of its train texts is the
    label the model can memorize. "disagrees" means the test text goes
    against that memory.
    """
    tickers = train.assign(ticker=train["text"].str.findall(CASHTAG))
    tickers = tickers.explode("ticker").dropna(subset=["ticker"])
    memory = tickers.groupby("ticker")["label"].agg(lambda x: x.mode()[0])

    def status(row: pd.Series) -> str:
        seen = [t for t in CASHTAG.findall(row["text"]) if t in memory.index]
        if not CASHTAG.search(row["text"]):
            return "no ticker"
        if not seen:
            return "unseen ticker"
        if any(memory[ticker] == row["label"] for ticker in seen):
            return "agrees"
        return "disagrees"

    return test.apply(status, axis=1).rename("ticker")


def word_label_shares(
    train: pd.DataFrame, words: list[str], labels: tuple[str, ...]
) -> pd.DataFrame:
    """How the train texts with every word are labeled."""
    rows = {}
    for word in words:
        mask = train["text"].str.contains(rf"\b{re.escape(word)}\b")
        shares = train.loc[mask, "label"].value_counts(normalize=True)
        rows[word] = {"texts": int(mask.sum()), **shares.to_dict()}
    columns = ["texts", *labels]
    return pd.DataFrame(rows).T.reindex(columns=columns).fillna(0)


CONCLUSIONS = """\
Conclusions

1. Business events without evaluative words are the largest failure
   case. 117 of 177 errors are on the positive / neutral border, and 13
   of the 26 errors with confidence 0.75+ have doubtful labels: news
   like "will expand the chain to 45 stores" is labeled positive with
   no sentiment word in it. A bag-of-words model has a ceiling here.
2. The train set has almost no negative news. Dropping the conflicting
   texts on day 2 (option A) removed 514 of 642 negative news, all of
   the "net profit decreased" kind. In train, "fell" and "decreased"
   are never negative and "loss" is positive in 81% of texts, so
   "Profit fell sharply" is predicted positive.
3. Tweets are memorized by ticker: 38% errors when a ticker's usual
   train label disagrees with the true one, 13% when it agrees.
4. Tweet slang and negation: "more bleeding", "sinking ship",
   "won't stay down". Negated texts have 24% errors against 18%.
5. Neutral tweets: 0 of 7 right.
Not a problem: numbers, percentages, short texts.

Confidence is meaningful: above 0.9 the model is right 99% of the
time, at 0.5 or below 42%.

What to improve next, most promising first

1. Keep one negative copy of every conflicting news text (option B)
   instead of dropping both: brings back 514 negative news.
2. Replace every cashtag with one token like "$ticker", so the model
   cannot memorize companies and handles unseen ones.
3. Mark negation scope: "won't stay down" -> "won't stay not_down".
4. Add a financial sentiment lexicon (Loughran-McDonald) as features,
   for slang and words rare in train.
5. Try a contextual model (FinBERT): business events and mixed
   sentences need meaning, not word counts.
6. At inference, show the confidence and flag predictions below 0.6.
"""


def format_analysis(
    frame: pd.DataFrame, train: pd.DataFrame, test: pd.DataFrame
) -> str:
    """The error analysis as a text report: numbers, examples and
    conclusions."""
    wrong = errors(frame)
    tickers = frame.assign(ticker=ticker_agreement(test, train))
    by_ticker = tickers.groupby("ticker")["correct"]
    examples = error_examples(frame, n=3)
    sections = [
        f"Error analysis: {len(wrong)} errors of {len(frame)} test texts",
        "Errors by kind\n" + error_types(frame).round(3).to_string(),
        "Errors by source\n" + source_table(frame).round(3).to_string(),
        "Accuracy by confidence\n"
        + confidence_table(frame).round(3).to_string(),
        "Failure cases\n"
        + failure_table(frame, failure_masks(frame)).round(3).to_string(),
        "Error rate by ticker memory\n"
        + (1 - by_ticker.mean()).round(3).to_string(),
        "Train labels of texts with a word\n"
        + word_label_shares(
            train, ["fell", "decreased", "loss", "rose"], LABELS
        )
        .round(2)
        .to_string(),
        "Examples\n"
        + "\n".join(
            f"[{row.error}, {row.kind}, {row.confidence:.2f}] {row.raw_text}"
            for row in examples.itertuples()
        ),
        CONCLUSIONS,
    ]
    return "\n\n".join(sections)


def main(
    model_path: Path = BEST_MODEL_PATH,
    train_path: Path = TRAIN_PATH,
    test_path: Path = TEST_PATH,
    results_dir: Path = RESULTS_DIR,
    figures_dir: Path = FIGURES_DIR,
) -> None:
    model = joblib.load(model_path)
    train, test = load_split(train_path), load_split(test_path)
    frame = predictions_frame(model, test)
    results_dir.mkdir(parents=True, exist_ok=True)
    errors(frame).to_csv(results_dir / "errors.csv", index=False)
    report = format_analysis(frame, train, test)
    (results_dir / "error_analysis.txt").write_text(report)
    plot_confusion_by_source(frame, figures_dir / "confusion_by_source.png")
    plot_confidence(frame, figures_dir / "confidence.png")
    print(report)


if __name__ == "__main__":
    main()
