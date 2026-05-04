"""
Test for PiecewiseDirectStandardization
"""

# Authors: Ruggero Guerrini
# License: MIT
import numpy as np
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.utils.estimator_checks import check_estimator

from chemotools.adaptation._piecewise_direct_standardization import (
    PiecewiseDirectStandardization,
)
from chemotools.scatter import StandardNormalVariate


@pytest.fixture
def sample_data():
    rng = np.random.default_rng(17)
    X_target = rng.normal(size=(100, 20))
    X_source = X_target * 2 - rng.normal(size=(100, 20)) * 0.02
    return X_target, X_source


def data_diff(dataset_ref, dataset_test):
    diff_norm = np.linalg.norm(dataset_ref - dataset_test)
    ref_norm = np.linalg.norm(dataset_ref)
    difference = diff_norm / ref_norm
    return difference


def test_compliance_PiecewiseDirectStandardization():
    # Arrange
    transformer = PiecewiseDirectStandardization()

    # Act & Assert
    check_estimator(transformer)


def test_fit_sets_attributes(sample_data):
    # Arrange
    X_target, X_source = sample_data

    # Act
    model = PiecewiseDirectStandardization().fit(X_source, X_target)

    # Assert
    assert hasattr(model, "n_samples_")
    assert hasattr(model, "n_features_")
    assert hasattr(model, "pls_")
    assert model.n_samples_ == X_target.shape[0]
    assert model.n_features_ == X_target.shape[1]
    assert len(model.pls_) == X_target.shape[1]


def test_transform_preserves_shape(sample_data):
    # Arrange
    X_target, X_source = sample_data

    # Act
    model = PiecewiseDirectStandardization().fit(X_source, X_target)
    X_transformed = model.transform(X_source)

    # Assert
    assert X_transformed.shape == X_source.shape
    assert X_transformed.shape == X_target.shape


def test_transform_improves_match_to_target(sample_data):
    # Arrange
    X_target, X_source = sample_data
    model = PiecewiseDirectStandardization().fit(X_source, X_target)

    # Act
    X_transformed = model.transform(X_source)
    before = data_diff(X_target, X_source)
    after = data_diff(X_target, X_transformed)

    # Assert
    assert after < before


def test_fit_raises_on_shape_mismatch():
    # Arrange
    rng = np.random.default_rng(17)
    X_target = rng.normal(size=(100, 20))
    X_source = rng.normal(size=(90, 20))

    # Act
    model = PiecewiseDirectStandardization()

    # Assert
    with pytest.raises(ValueError, match="same shape"):
        model.fit(X_source, X_target)


def test_fit_raises_when_n_components_too_large(sample_data):
    # Arrange
    X_target, X_source = sample_data

    # Act
    model = PiecewiseDirectStandardization(window_length=2, n_components=4)

    # Assert
    with pytest.raises(ValueError, match="n_components"):
        model.fit(X_source, X_target)


def test_transform_before_fit_raises(sample_data):
    # Arrange
    _, X_source = sample_data

    # Act
    model = PiecewiseDirectStandardization()

    # Assert
    with pytest.raises(NotFittedError):
        model.transform(X_source)


def test_transform_does_not_modify_input(sample_data):
    # Arrange
    X_target, X_source = sample_data
    X_source_original = X_source.copy()
    X_target_original = X_target.copy()

    # Act
    model = PiecewiseDirectStandardization().fit(X_source, X_target)
    model.transform(X_source)

    # Assert
    np.testing.assert_array_equal(X_source, X_source_original)
    np.testing.assert_array_equal(X_target, X_target_original)


def test_transform_is_idempotent_on_input(sample_data):
    # Arrange
    X_target, X_source = sample_data

    # Act
    model = PiecewiseDirectStandardization().fit(X_source, X_target)
    result1 = model.transform(X_source)
    result2 = model.transform(X_source)

    # Assert
    np.testing.assert_array_equal(result1, result2)


# Verify that there are always all the attributes
def test_pls_params_keys(sample_data):
    # Arrange
    X_target, X_source = sample_data

    # Act
    model = PiecewiseDirectStandardization().fit(X_source, X_target)

    # Assert
    for params in model.pls_:
        assert "x_mean_" in params
        assert "coef_" in params
        assert "intercept_" in params


# Transform must works on unseen data with the same shape
def test_transform_on_unseen_data(sample_data):
    # Arrange
    X_target, X_source = sample_data
    rng = np.random.default_rng(17)
    X_new = rng.normal(size=X_source.shape)

    # Act
    model = PiecewiseDirectStandardization().fit(X_source, X_target)
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
    model = PiecewiseDirectStandardization().fit(X_source, X_target)

    # Assert
    with pytest.raises(ValueError):
        model.transform(X_wrong)


# Test Pipeline
def test_pipeline(sample_data):
    # Arrange
    X_target, X_source = sample_data

    # Act
    pipe = Pipeline(
        [
            ("scaler", StandardNormalVariate()),
            (
                "model",
                PiecewiseDirectStandardization(
                    window_length=25, n_components=2, scale=True
                ),
            ),
        ]
    )

    pipe.fit(X_source, X_target)
    X_transformed = pipe.transform(X_source)

    # Assert
    assert X_transformed.shape == X_source.shape == X_target.shape


# Test Pipeline, GridSearchCV
def pds_score(estimator, X, y):
    X_transformed = estimator.transform(X)
    return -data_diff(y, X_transformed)


def test_pipeline_gridsearchcv(sample_data):
    # Arrange
    X_target, X_source = sample_data
    pipe = Pipeline(
        [
            ("scaler", StandardNormalVariate()),
            (
                "model",
                PiecewiseDirectStandardization(),
            ),
        ]
    )
    param_grid = {
        "model__window_length": [15, 25, 35],
        "model__n_components": [1, 2],
        "model__scale": [True, False],
    }
    grid = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        cv=3,
        scoring=pds_score,
        error_score="raise",
    )

    # Act
    grid.fit(X_source, X_target)

    # Assert
    assert grid.best_estimator_ is not None
