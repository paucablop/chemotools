"""
Test for DirectStandardization
"""

# Authors: Ruggero Guerrini
# License: MIT

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.utils.estimator_checks import check_estimator

from chemotools.adaptation._direct_standardization import DirectStandardization


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


def test_transform_improves_match_to_target(sample_data):
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


def test_transform_does_not_modify_input(sample_data):
    # Arrange
    X_target, X_source = sample_data
    X_source_original = X_source.copy()
    X_target_original = X_target.copy()

    # Act
    model = DirectStandardization().fit(X_source, X_target)
    model.transform(X_source)

    # Assert
    np.testing.assert_array_equal(X_source, X_source_original)
    np.testing.assert_array_equal(X_target, X_target_original)


def test_transform_is_idempotent_on_input(sample_data):
    # Arrange
    X_target, X_source = sample_data

    # Act
    model = DirectStandardization().fit(X_source, X_target)
    result1 = model.transform(X_source)
    result2 = model.transform(X_source)

    # Assert
    np.testing.assert_array_equal(result1, result2)


# Transform must works on unseen data with the same shape
def test_transform_on_unseen_data(sample_data):
    # Arrange
    X_target, X_source = sample_data
    rng = np.random.default_rng(17)
    X_new = rng.normal(size=X_source.shape)

    # Act
    model = DirectStandardization().fit(X_source, X_target)
    X_transformed = model.transform(X_new)

    # Assert
    assert X_transformed.shape == X_new.shape


# Trasnform must not works with data with a different shape
def test_transform_raises_on_wrong_n_features(sample_data):
    # Arrange
    X_target, X_source = sample_data
    rng = np.random.default_rng(99)

    # Act
    X_wrong = rng.normal(size=(100, 15))  # 15 invece di 20
    model = DirectStandardization().fit(X_source, X_target)

    # Assert
    with pytest.raises(ValueError):
        model.transform(X_wrong)
