"""Tests for OrthogonalSignalCorrection."""

import numpy as np
import pytest
from sklearn.utils.estimator_checks import check_estimator

from chemotools.cross_decomposition import OrthogonalSignalCorrection

# Test compliance with scikit-learn


def test_compliance_osc():
    # Arrange
    transformer = OrthogonalSignalCorrection()

    # Act & Assert
    # n_iter_ is per-component (array), but sklearn's check_transformer_n_iter
    # only handles this for its own hardcoded CROSS_DECOMPOSITION list.
    check_estimator(
        transformer,
        expected_failed_checks={
            "check_transformer_n_iter": "n_iter_ is stored per component (array) "
            "instead of scalar"
        },
    )


@pytest.mark.parametrize(
    "y_factory",
    [
        lambda rng, n_samples: rng.normal(size=n_samples),
        lambda rng, n_samples: rng.normal(size=(n_samples, 2)),
    ],
)
def test_sjoblom_method_preserves_expected_shapes(y_factory):
    # Arrange
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 6))
    y = y_factory(rng, X.shape[0])
    transformer = OrthogonalSignalCorrection(method="sjoblom", n_components=2)

    # Act
    Xt = transformer.fit_transform(X, y)

    # Assert
    assert Xt.shape == X.shape
    assert transformer.scores_.shape == (X.shape[0], 2)
    assert transformer.weights_.shape == (X.shape[1], 2)
    assert transformer.loadings_.shape == (X.shape[1], 2)
    assert np.isfinite(transformer.scores_).all()
    assert np.isfinite(transformer.weights_).all()
    assert np.isfinite(transformer.loadings_).all()
