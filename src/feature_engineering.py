"""Leakage-safe feature engineering for review text."""

from typing import Iterable

from sklearn.feature_extraction.text import TfidfVectorizer


class TextFeaturePipeline:
    """Build review-text TF-IDF features without fitting on held-out data."""

    def __init__(self) -> None:
        """Initialize the training-only TF-IDF vectorizer."""
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=10000,
            min_df=5,
            max_df=0.8,
            sublinear_tf=True,
        )

    def fit_transform(self, text_series: Iterable[str]):
        """Fit on training text and return the resulting TF-IDF matrix."""
        return self.vectorizer.fit_transform(text_series)

    def transform(self, text_series: Iterable[str]):
        """Transform validation or test text using the fitted vectorizer."""
        return self.vectorizer.transform(text_series)
