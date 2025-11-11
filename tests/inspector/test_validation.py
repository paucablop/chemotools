import numpy as np
import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from chemotools.inspector._validation import (
    _validate_model,
    _validate_and_extract_model,
    _validate_datasets_consistency,
)


def test_validate_model_accepts_fitted_estimators():
    X = np.random.default_rng(0).normal(size=(12, 4))
    pca = PCA(n_components=2).fit(X)
    assert _validate_model(pca) is pca

    pipeline = Pipeline(
        [
            ("scale", StandardScaler()),
            ("pca", PCA(n_components=2)),
        ]
    ).fit(X)
    assert _validate_model(pipeline) is pipeline


def test_validate_model_rejects_unfitted_or_wrong_type():
    X = np.random.default_rng(1).normal(size=(8, 3))

    unfitted = PCA(n_components=2)
    with pytest.raises(NotFittedError):
        _validate_model(unfitted)

    bad_pipeline = Pipeline([("scale", StandardScaler())]).fit(X)
    with pytest.raises(TypeError, match="Model must be"):  # final step not PCA/PLS
        _validate_model(bad_pipeline)


def test_validate_and_extract_model_handles_pipeline_and_estimators():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(15, 5))
    y = rng.normal(size=(15, 2))

    full_pipeline = Pipeline(
        [
            ("scale", StandardScaler()),
            ("pca", PCA(n_components=3)),
        ]
    ).fit(X)
    estimator, transformer = _validate_and_extract_model(full_pipeline)
    assert isinstance(estimator, PCA)
    assert isinstance(transformer, Pipeline)
    assert list(transformer.named_steps) == ["scale"]

    single_step = Pipeline([("pls", PLSRegression(n_components=2))]).fit(X, y)
    estimator_single, transformer_single = _validate_and_extract_model(single_step)
    assert isinstance(estimator_single, PLSRegression)
    assert transformer_single is None

    plain_pls = PLSRegression(n_components=2).fit(X, y)
    extracted, wrapper = _validate_and_extract_model(plain_pls)
    assert extracted is plain_pls
    assert wrapper is None


def test_validate_and_extract_model_rejects_invalid_or_unfitted():
    X = np.random.default_rng(3).normal(size=(10, 4))

    bad_pipeline = Pipeline([("scale", StandardScaler())]).fit(X)
    with pytest.raises(TypeError, match="Model must be"):
        _validate_and_extract_model(bad_pipeline)

    unfitted = Pipeline([("pca", PCA(n_components=2))])
    with pytest.raises(NotFittedError):
        _validate_and_extract_model(unfitted)


def test_validate_datasets_consistency_passes_with_matching_shapes():
    X_train = np.ones((6, 3))
    y_train = np.arange(6)
    X_test = np.ones((2, 3))
    y_test = np.arange(2)
    X_val = np.ones((3, 3))
    y_val = np.arange(3)

    _validate_datasets_consistency(
        X_train,
        y_train,
        X_test,
        y_test,
        X_val,
        y_val,
        supervised=True,
    )

    _validate_datasets_consistency(
        X_train,
        None,
        X_test,
        None,
        X_val,
        None,
        supervised=False,
    )


def test_validate_datasets_consistency_detects_shape_and_label_issues():
    X_train = np.ones((5, 4))
    y_train = np.arange(5)
    X_test = np.ones((2, 4))
    y_test = np.arange(2)
    X_val = np.ones((3, 4))
    y_val = np.arange(3)

    with pytest.raises(ValueError, match="y_train must have"):
        _validate_datasets_consistency(
            X_train,
            y_train[:-1],
            None,
            None,
            None,
            None,
            supervised=False,
        )

    with pytest.raises(ValueError, match="X_test must have same"):
        _validate_datasets_consistency(
            X_train,
            y_train,
            np.ones((2, 5)),
            y_test,
            None,
            None,
            supervised=False,
        )

    with pytest.raises(ValueError, match="y_train required"):
        _validate_datasets_consistency(
            X_train,
            None,
            X_test,
            y_test,
            X_val,
            y_val,
            supervised=True,
        )

    with pytest.raises(ValueError, match="y_test required"):
        _validate_datasets_consistency(
            X_train,
            y_train,
            X_test,
            None,
            X_val,
            y_val,
            supervised=True,
        )

    with pytest.raises(ValueError, match="y_val required"):
        _validate_datasets_consistency(
            X_train,
            y_train,
            None,
            None,
            X_val,
            None,
            supervised=True,
        )

    with pytest.raises(ValueError, match="y_test must have"):
        _validate_datasets_consistency(
            X_train,
            y_train,
            X_test,
            y_test[:-1],
            None,
            None,
            supervised=False,
        )

    with pytest.raises(ValueError, match="y_val must have"):
        _validate_datasets_consistency(
            X_train,
            y_train,
            None,
            None,
            X_val,
            y_val[:-1],
            supervised=False,
        )
