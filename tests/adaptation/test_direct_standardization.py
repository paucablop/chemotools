"""
Test for DirectStandardization
"""

# Authors: Ruggero Guerrini
# License: MIT
import numpy as np
import pytest
from sklearn.utils.estimator_checks import check_estimator
from sklearn.exceptions import NotFittedError

from chemotools.adaptation._direct_standardization import (
    DirectStandardization,
)


def data_diff(dataset_ref, dataset_test):
    diff_norm = np.linalg.norm(dataset_ref - dataset_test)
    ref_norm = np.linalg.norm(dataset_ref)
    difference = diff_norm / ref_norm
    return difference


@pytest.fixture
def sample_data():
    rng = np.random.default_rng(17)
    y = rng.normal(size=(100, 20))
    X = y * 2 - rng.normal(size=(100, 20)) * 0.02
    return y, X


def test_compliance_DirectStandardization():
    # Arrange
    transformer = DirectStandardization()
    # Act & Assert
    check_estimator(transformer)


def test_improvement(sample_data):
    # Arrange
    y, X = sample_data

    # Fit model
    model = DirectStandardization().fit(X, y)

    # Act
    X_transformed = model.transform(X)
    before = data_diff(y, X)
    after = data_diff(y, X_transformed)

    # Assert
    assert before > after


def test_transform_preserves_shape(sample_data):
    # Arrange
    y, X = sample_data
    # Act
    model = DirectStandardization().fit(X, y)
    X_transformed = model.transform(X)
    # Assert
    assert X_transformed.shape == X.shape
    assert X_transformed.shape == y.shape


def test_fit_sets_attributes(sample_data):
    # Arrange
    y, X = sample_data
    # Act
    model = DirectStandardization().fit(X, y)
    # Assert
    assert hasattr(model, "T_")


def test_transform_improves_match_to_master(sample_data):
    # Arrange
    y, X = sample_data
    model = DirectStandardization().fit(X, y)
    # Act
    X_transformed = model.transform(X)
    before = data_diff(y, X)
    after = data_diff(y, X_transformed)
    # Assert
    assert after < before


def test_transform_before_fit_raises(sample_data):
    # Arrange
    _, X = sample_data
    # Act
    model = DirectStandardization()

    # Assert
    with pytest.raises(NotFittedError):
        model.transform(X)
