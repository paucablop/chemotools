"""Tests for DirectOrthogonalization."""

import numpy as np
import pytest
from sklearn.utils.estimator_checks import check_estimator

from chemotools.projection import DirectOrthogonalization


# Test complience with scikit-learn
def test_compliance_direct_orthogonalization():
    """
    Check sklearn estimator compliance for the DirectOrthogonalization transformer.
    """
    # Arrange
    do = DirectOrthogonalization()

    # Act & Assert
    check_estimator(do)


# Test functionality
def test_direct_orthogonalization_correctness():
    """
    Test the correctness of the DirectOrthogonalization implementation against the 
    example provided in the original paper by Trygg and Wold (2002) [1].
    """
    # Arrange
    X = np.array([[-2.18, 1.84, -0.48, 0.83], [-2.18, -0.16, 1.52, 0.83]]).T
    y = np.array([2, 2, 0, -4])

    # Act
    do = DirectOrthogonalization(n_components=1).fit(X, y)

    # Assert
    # Calculated value used to assess numerical stability
    np.testing.assert_allclose(
        do.retained_variance_ratio_, 0.7495221388680522, atol=1e-8
    )


def test_direct_orthogonalization_raises_error_many_components():
    """
    Test that DirectOrthogonalization raises an error when the number of components
    requested is greater than the number of features.
    """
    # Arrange
    X = np.array([[-2.18, 1.84, -0.48, 0.83], [-2.18, -0.16, 1.52, 0.83]]).T
    y = np.array([2, 2, 0, -4])

    # Act / Assert
    with pytest.raises(ValueError, match="Number of components must be less than or" \
    " equal to the number of features"):
        DirectOrthogonalization(n_components=3).fit(X, y)


def test_fit_rejects_single_sample():
    """Reject datasets with fewer than two samples."""
    # Arrange
    X = np.array([[1.0, 2.0, 3.0]])
    y = np.array([1.0])
    do = DirectOrthogonalization()

    # Act / Assert
    with pytest.raises(ValueError, match="At least 2 samples are required"):
        do.fit(X, y)
