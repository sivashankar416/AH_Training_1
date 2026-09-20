import pytest
from scipy import sparse
from sklearn.exceptions import NotFittedError

from src.feature_engineering import TextFeaturePipeline

TRAINING_TEXTS = [
    "alpha document one",
    "alpha document two",
    "alpha document three",
    "alpha document four",
    "alpha document five",
    "beta document six",
    "beta document seven",
    "beta document eight",
    "beta document nine",
    "beta document ten",
]

VALIDATION_TEXTS = [
    "omega validation",
]


def test_fit_transform_returns_sparse_matrix():
    matrix = TextFeaturePipeline().fit_transform(TRAINING_TEXTS)

    assert sparse.issparse(matrix)


def test_training_matrix_has_two_dimensions():
    matrix = TextFeaturePipeline().fit_transform(TRAINING_TEXTS)

    assert matrix.ndim == 2


def test_transform_returns_sparse_matrix():
    pipeline = TextFeaturePipeline()
    pipeline.fit_transform(TRAINING_TEXTS)

    assert sparse.issparse(pipeline.transform(VALIDATION_TEXTS))


def test_training_and_validation_have_same_number_of_columns():
    pipeline = TextFeaturePipeline()
    training_matrix = pipeline.fit_transform(TRAINING_TEXTS)
    validation_matrix = pipeline.transform(VALIDATION_TEXTS)

    assert training_matrix.shape[1] == validation_matrix.shape[1]


def test_unseen_validation_word_does_not_raise_error():
    pipeline = TextFeaturePipeline()
    pipeline.fit_transform(TRAINING_TEXTS)

    pipeline.transform(["word unseen during training"])


def test_transform_before_fitting_raises_not_fitted_error():
    with pytest.raises(NotFittedError):
        TextFeaturePipeline().transform(VALIDATION_TEXTS)
