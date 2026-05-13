"""Tests for OrthogonalPLS."""

import numpy as np
import pytest
from sklearn.utils.estimator_checks import check_estimator

from chemotools.projection import OrthogonalPLS


# Test compliance with scikit-learn
def test_compliance_opls():
    """Check sklearn estimator compliance for the OPLS transformer."""
    # Arrange
    transformer = OrthogonalPLS()

    # Act & Assert
    check_estimator(transformer)


# Test OPLS against literature example
def test_opls_correctness():
    """
    Test the correctness of the OPLS implementation against the example provided in
    the original paper by Trygg and Wold (2002) [1].
    """
    # Arrange
    X = np.array([[-2.18, 1.84, -0.48, 0.83], [-2.18, -0.16, 1.52, 0.83]]).T
    y = np.array([2, 2, 0, -4])

    # Reference values are taken from [1]
    x_weights_orth_ref = np.array([-0.89, 0.45])
    x_loadings_orth_ref = np.array([-1.16, -0.09])
    x_scores_orth_ref = np.array([0.97, -1.71, 1.11, -0.37])

    # Act
    opls = OrthogonalPLS(n_components=1).fit(X, y)

    # Assert
    # Absolute tolerance 1e-2 because of reference's precision
    np.testing.assert_allclose(
        opls.x_weights_orth_.flatten(), x_weights_orth_ref, atol=1e-2
    )
    np.testing.assert_allclose(
        opls.x_loadings_orth_.flatten(), x_loadings_orth_ref, atol=1e-2
    )
    np.testing.assert_allclose(
        opls.x_scores_orth_.flatten(), x_scores_orth_ref, atol=1e-2
    )

    # Numerical stability
    np.testing.assert_allclose(
        opls.retained_variance_ratio_, 0.5743272092463821, atol=1e-8
    )


def test_fit_rejects_single_sample():
    """Reject datasets with fewer than two samples."""
    # Arrange
    X = np.array([[1.0, 2.0, 3.0]])
    y = np.array([1.0])
    transformer = OrthogonalPLS()

    # Act / Assert
    with pytest.raises(ValueError, match="At least 2 samples are required"):
        transformer.fit(X, y)


def test_fit_rejects_n_components_exceeding_rank():
    """Reject n_components larger than min(n_samples - 1, n_features)."""
    # Arrange: 3 samples x 2 features → max components = min(2, 2) = 2
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    y = np.array([1.0, 2.0, 3.0])
    transformer = OrthogonalPLS(n_components=3)

    # Act / Assert
    with pytest.raises(ValueError, match="n_components=3 is too large"):
        transformer.fit(X, y)


def test_fit_rejects_zero_variance_X():
    """Reject X with zero variance after mean-centering (all-constant matrix)."""
    # Arrange
    X = np.ones((3, 2))
    y = np.array([1.0, 2.0, 3.0])
    transformer = OrthogonalPLS(n_components=1)

    # Act / Assert
    with pytest.raises(ValueError, match="X has zero variance after mean-centering"):
        transformer.fit(X, y)
