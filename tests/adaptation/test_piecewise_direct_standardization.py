"""
Test for PiecewiseDirectStandardization
"""

# Authors: Ruggero Guerrini
# License: MIT
import numpy as np
import pytest
import sklearn
from sklearn.cross_decomposition import PLSRegression
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils.metadata_routing import MetadataRouter

from chemotools.adaptation._piecewise_direct_standardization import (
    PiecewiseDirectStandardization,
)
from chemotools.derivative import SavitzkyGolay


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
    X_source, X = sample_data

    # Act
    model = PiecewiseDirectStandardization().fit(X, X_source=X_source)

    # Assert
    assert hasattr(model, "n_features_in_")
    assert hasattr(model, "x_mean_")
    assert hasattr(model, "coef_")
    assert hasattr(model, "intercept_")
    assert model.x_mean_.shape == (X.shape[1], 2 * model.window_length + 1)
    assert model.coef_.shape == (X.shape[1], 2 * model.window_length + 1)
    assert model.intercept_.shape == (X.shape[1],)
    assert model.n_features_in_ == X.shape[1]


def test_transform_preserves_shape(sample_data):
    # Arrange
    X_source, X = sample_data

    # Act
    model = PiecewiseDirectStandardization().fit(X, X_source=X_source)
    X_transformed = model.transform(X)

    # Assert
    assert X_transformed.shape == X_source.shape
    assert X_transformed.shape == X.shape


def test_transform_improves_match_to_target(sample_data):
    # Arrange
    X_source, X = sample_data
    model = PiecewiseDirectStandardization().fit(X, X_source=X_source)

    # Act
    X_transformed = model.transform(X)
    before = data_diff(X, X_source)
    after = data_diff(X_transformed, X_source)

    # Assert
    assert after < before


def test_fit_raises_on_shape_mismatch(sample_data):
    # Arrange
    X_source, X = sample_data

    # Act
    model = PiecewiseDirectStandardization()

    # Assert
    with pytest.raises(ValueError, match="must have the same shape"):
        model.fit(X, X_source=X_source[:-1, :])


def test_fit_raises_when_n_components_too_large(sample_data):
    # Arrange
    X_source, X = sample_data

    # Act
    model = PiecewiseDirectStandardization(window_length=2, n_components=4)

    # Assert
    with pytest.raises(ValueError, match="n_components"):
        model.fit(X, X_source=X_source)


def test_transform_before_fit_raises(sample_data):
    # Arrange
    _, X = sample_data

    # Act
    model = PiecewiseDirectStandardization()

    # Assert
    with pytest.raises(NotFittedError):
        model.transform(X)


def test_transform_does_not_modify_input(sample_data):
    # Arrange
    X_source, X = sample_data
    X_original = X.copy()
    X_source_original = X_source.copy()

    # Act
    model = PiecewiseDirectStandardization().fit(X, X_source=X_source)
    model.transform(X)

    # Assert
    np.testing.assert_array_equal(X, X_original)
    np.testing.assert_array_equal(X_source, X_source_original)


def test_transform_is_deterministic(sample_data):
    # Arrange
    X_source, X = sample_data

    # Act
    model = PiecewiseDirectStandardization().fit(X, X_source=X_source)
    result1 = model.transform(X)
    result2 = model.transform(X)

    # Assert
    np.testing.assert_array_equal(result1, result2)


def get_metadata_routing(self):
    router = MetadataRouter(owner=self.__class__.__name__)
    router.add_self_request(self)
    return router


def test_pipeline_gridsearchcv_pls_metadata_routing(sample_data):
    # Arrange
    X_source, X = sample_data
    rng = np.random.default_rng(42)
    y_concentration = rng.normal(size=(100, 1))

    sklearn.set_config(enable_metadata_routing=True)

    pipe = Pipeline(
        [
            ("scaler", SavitzkyGolay()),
            ("model", PiecewiseDirectStandardization().set_fit_request(X_source=True)),
            ("pls", PLSRegression()),
        ]
    )
    param_grid = {
        "scaler__window_length": [15, 25],
        "scaler__polyorder": [2, 3],
        "scaler__deriv": [1, 2],
        "model__window_length": [10, 15, 20],
        "model__n_components": [2, 3, 5],
        "pls__n_components": [2, 3],
    }
    grid = GridSearchCV(pipe, param_grid, cv=3, error_score="raise")

    # Act — X_target passa come kwarg, sklearn lo smista a DS con gli indici corretti
    grid.fit(X, y_concentration, X_source=X_source)

    # Assert
    assert grid.best_estimator_ is not None

    # Cleanup — reset config per non sporcare altri test
    sklearn.set_config(enable_metadata_routing=False)


def test_identity_transformation_when_X_source_is_none():
    # Arrange
    rng = np.random.default_rng(42)
    X = rng.normal(size=(50, 10))
    model = PiecewiseDirectStandardization()

    # Act - fit with X_source=None should trigger identity transformation
    with pytest.warns(UserWarning, match="identity transformation"):
        model.fit(X, X_source=None)

    X_transformed = model.transform(X)

    # Assert - should return X unchanged
    np.testing.assert_array_equal(X_transformed, X)
    assert hasattr(model, "x_source_provided_")
    assert model.x_source_provided_ is False
