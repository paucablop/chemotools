"""
Test for PiercewiseDirectStandardization
"""

# Authors: Ruggero Guerrini
# License: MIT
import numpy as np
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.utils.estimator_checks import check_estimator

from chemotools.adaptation._piercewise_direct_standardization import (
    PiercewiseDirectStandardization,
)


@pytest.fixture
def sample_data():
    rng = np.random.default_rng(17)
    X_master = rng.normal(size=(100, 20))
    X_slave = X_master * 2 - rng.normal(size=(100, 20)) * 0.02
    return X_master, X_slave


def data_diff(dataset_ref, dataset_test):
    diff_norm = np.linalg.norm(dataset_ref - dataset_test)
    ref_norm = np.linalg.norm(dataset_ref)
    difference = diff_norm / ref_norm
    return difference


def test_compliance_DirectStandardization():
    # Arrange
    transformer = PiercewiseDirectStandardization()
    # Act & Assert
    check_estimator(transformer)


def test_fit_sets_attributes(sample_data):
    # Arrange
    X_master, X_slave = sample_data
    # Act
    model = PiercewiseDirectStandardization().fit(X_slave, X_master)
    # Assert
    assert hasattr(model, "n_samples_")
    assert hasattr(model, "n_features_")
    assert hasattr(model, "pls_")
    assert model.n_samples_ == X_master.shape[0]
    assert model.n_features_ == X_master.shape[1]
    assert len(model.pls_) == X_master.shape[1]


def test_transform_preserves_shape(sample_data):
    # Arrange
    X_master, X_slave = sample_data
    # Act
    model = PiercewiseDirectStandardization().fit(X_slave, X_master)
    X_transformed = model.transform(X_slave)
    # Assert
    assert X_transformed.shape == X_slave.shape
    assert X_transformed.shape == X_master.shape


def test_transform_improves_match_to_master(sample_data):
    # Arrange
    X_master, X_slave = sample_data
    model = PiercewiseDirectStandardization().fit(X_slave, X_master)
    # Act
    X_transformed = model.transform(X_slave)
    before = data_diff(X_master, X_slave)
    after = data_diff(X_master, X_transformed)
    # Assert
    assert after < before


def test_fit_raises_on_shape_mismatch():
    # Arrange
    rng = np.random.default_rng(17)
    X_master = rng.normal(size=(100, 20))
    X_slave = rng.normal(size=(90, 20))
    # Act
    model = PiercewiseDirectStandardization()
    # Assert
    with pytest.raises(ValueError, match="same shape"):
        model.fit(X_slave, X_master)


def test_fit_raises_when_n_components_too_large(sample_data):
    # Arrange
    X_master, X_slave = sample_data
    # Act
    model = PiercewiseDirectStandardization(window_length=2, n_components=4)
    # Assert
    with pytest.raises(ValueError, match="n_components"):
        model.fit(X_slave, X_master)


def test_transform_before_fit_raises(sample_data):
    # Arrange
    _, X_slave = sample_data
    # Act
    model = PiercewiseDirectStandardization()

    # Assert
    with pytest.raises(NotFittedError):
        model.transform(X_slave)
