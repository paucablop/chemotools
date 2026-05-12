"""
Test for DirectStandardization
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

from chemotools.adaptation._direct_standardization import (
    DirectStandardization,
)
from chemotools.derivative import SavitzkyGolay
from chemotools.scatter import StandardNormalVariate


def data_diff(dataset_ref, dataset_test):
    diff_norm = np.linalg.norm(dataset_ref - dataset_test)
    ref_norm = np.linalg.norm(dataset_ref)
    difference = diff_norm / ref_norm
    return difference


@pytest.fixture
def sample_data():
    rng = np.random.default_rng(17)
    X_target = rng.normal(size=(100, 20))
    X = X_target * 2 - rng.normal(size=(100, 20)) * 0.02
    return X_target, X


def test_compliance_DirectStandardization():
    # Arrange
    transformer = DirectStandardization()

    # Act & Assert
    check_estimator(transformer)


def test_init_and_fit_time_X_target_give_same_result(sample_data):
    X_target, X_source = sample_data

    model_init = DirectStandardization(X_target=X_target).fit(X_source)
    model_fit = DirectStandardization().fit(X_source, X_target=X_target)

    np.testing.assert_array_equal(model_init.T_, model_fit.T_)


def test_improvement(sample_data):
    # Arrange
    X_target, X = sample_data

    # Fit model
    model = DirectStandardization(X_target=X_target).fit(X)

    # Act
    X_transformed = model.transform(X)
    before = data_diff(X_target, X)
    after = data_diff(X_target, X_transformed)

    # Assert
    assert before > after


def test_transform_preserves_shape(sample_data):
    # Arrange
    X_target, X = sample_data

    # Act
    model = DirectStandardization(X_target=X_target).fit(X)
    X_transformed = model.transform(X)

    # Assert
    assert X_transformed.shape == X.shape
    assert X_transformed.shape == X_target.shape


def test_fit_sets_attributes(sample_data):
    # Arrange
    X_target, X = sample_data

    # Act
    model = DirectStandardization(X_target=X_target).fit(X)

    # Assert
    assert hasattr(model, "T_")


def test_transform_improves_match_to_target(sample_data):
    # Arrange
    X_target, X = sample_data
    model = DirectStandardization(X_target=X_target).fit(X)

    # Act
    X_transformed = model.transform(X)
    before = data_diff(X_target, X)
    after = data_diff(X_target, X_transformed)

    # Assert
    assert after < before


def test_transform_before_fit_raises(sample_data):
    # Arrange
    X_target, X = sample_data

    # Act
    model = DirectStandardization(X_target=X_target)

    # Assert
    with pytest.raises(NotFittedError):
        model.transform(X)


def test_transform_does_not_modifX_target_input(sample_data):
    # Arrange
    X_target, X_source = sample_data
    X_source_original = X_source.copy()
    X_target_original = X_target.copy()

    # Act
    model = DirectStandardization(X_target=X_target).fit(X_source)
    model.transform(X_source)

    # Assert
    np.testing.assert_array_equal(X_source, X_source_original)
    np.testing.assert_array_equal(X_target, X_target_original)


def test_transform_is_idempotent_on_input(sample_data):
    # Arrange
    X_target, X_source = sample_data

    # Act
    model = DirectStandardization(X_target=X_target).fit(X_source)
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
    model = DirectStandardization(X_target=X_target).fit(X_source)
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
    model = DirectStandardization(X_target=X_target).fit(X_source)

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
                DirectStandardization(X_target=X_target),
            ),
        ]
    )

    pipe.fit(X_source)
    X_transformed = pipe.transform(X_source)

    # Assert
    assert X_transformed.shape == X_source.shape == X_target.shape


def get_metadata_routing(self):
    router = MetadataRouter(owner=self.__class__.__name__)
    router.add_self_request(self)
    return router


def test_pipeline_gridsearchcv_pls_metadata_routing(sample_data):
    # Arrange
    X_target, X_source = sample_data
    rng = np.random.default_rng(42)
    y_concentration = rng.normal(size=(100, 1))

    sklearn.set_config(enable_metadata_routing=True)

    pipe = Pipeline(
        [
            ("scaler", SavitzkyGolay()),
            ("ds", DirectStandardization().set_fit_request(X_target=True)),
            ("pls", PLSRegression()),
        ]
    )
    param_grid = {
        "scaler__window_length": [15, 25],
        "scaler__polyorder": [2, 3],
        "scaler__deriv": [1, 2],
        "pls__n_components": [2, 3],
    }
    grid = GridSearchCV(pipe, param_grid, cv=3, error_score="raise")

    # Act — X_target passa come kwarg, sklearn lo smista a DS con gli indici corretti
    grid.fit(X_source, y_concentration, X_target=X_target)

    # Assert
    assert grid.best_estimator_ is not None

    # Cleanup — reset config per non sporcare altri test
    sklearn.set_config(enable_metadata_routing=False)
