"""The models compared in train.py. All are scikit-learn pipelines."""

from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from .text import normalize


def _word() -> TfidfVectorizer:
    # token_pattern keeps emoji and @user as tokens; emoji carry a lot of the signal
    return TfidfVectorizer(
        preprocessor=normalize,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
        token_pattern=r"(?u)@?\w+|[^\w\s]",
    )


def _char() -> TfidfVectorizer:
    # char n-grams within word boundaries cope with spelling variation and missing tone marks
    return TfidfVectorizer(
        preprocessor=normalize,
        analyzer="char_wb",
        ngram_range=(2, 5),
        min_df=2,
        sublinear_tf=True,
    )


def _logreg(C: float) -> LogisticRegression:
    return LogisticRegression(C=C, max_iter=2000, class_weight="balanced")


def build(name: str, C: float = 1.0) -> Pipeline:
    if name == "majority":
        return Pipeline([("clf", DummyClassifier(strategy="most_frequent"))])
    if name == "word":
        return Pipeline([("tfidf", _word()), ("clf", _logreg(C))])
    if name == "char":
        return Pipeline([("tfidf", _char()), ("clf", _logreg(C))])
    if name == "word+char":
        feats = FeatureUnion([("word", _word()), ("char", _char())])
        return Pipeline([("tfidf", feats), ("clf", _logreg(C))])
    raise ValueError(f"unknown model: {name}")


MODELS = ["majority", "word", "char", "word+char"]
C_GRID = [0.3, 1.0, 3.0, 10.0]
