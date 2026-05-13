"""Tests for DirectOrthogonalization."""

import numpy as np
import pytest
from sklearn.utils.estimator_checks import check_estimator

from chemotools.projection import DirectOrthogonalization


# Test compliance with scikit-learn
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

    # Values calculated for numerical stability
    x_weights_orth_ref = np.array([0.85718287, 0.51501217])
    x_loadings_orth_ref = np.array([0.85718287, 0.51501217])
    x_scores_orth_ref = np.array([-2.99481566, 1.49138404, 0.36794023, 1.13549139])
    x_transformed_ref = np.array(
        [
            [0.387105, -0.637633],
            [0.561611, -0.928081],
            [-0.795392, 1.330506],
            [-0.143324, 0.245208],
        ]
    )

    # Act
    do = DirectOrthogonalization(n_components=1).fit(X, y)
    transformed = do.transform(X)

    # Assert
    # Calculated value used to assess numerical stability
    np.testing.assert_allclose(
        do.x_weights_orth_.flatten(), x_weights_orth_ref, atol=1e-8
    )

    np.testing.assert_allclose(
        do.x_loadings_orth_.flatten(), x_loadings_orth_ref, atol=1e-8
    )

    np.testing.assert_allclose(
        do.x_scores_orth_.flatten(), x_scores_orth_ref, atol=1e-8
    )

    np.testing.assert_allclose(
        do.removed_variance_ratio_, 0.7495221388680522, atol=1e-8
    )

    np.testing.assert_allclose(transformed, x_transformed_ref, atol=1e-6)


def test_direct_orthogonalization_raises_error_many_components():
    """
    Test that DirectOrthogonalization raises an error when the number of components
    requested is greater than the number of features.
    """
    # Arrange
    X = np.array([[-2.18, 1.84, -0.48, 0.83], [-2.18, -0.16, 1.52, 0.83]]).T
    y = np.array([2, 2, 0, -4])

    # Act / Assert
    with pytest.raises(
        ValueError,
        match="Number of components must be less than or"
        " equal to the number of features",
    ):
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


def test_fit_rejects_zero_variance_X():
    """Reject X with zero variance after mean-centering (all-constant matrix)."""
    # Arrange
    X = np.ones((3, 2))
    y = np.array([1.0, 2.0, 3.0])
    transformer = DirectOrthogonalization(n_components=1)

    # Act / Assert
    with pytest.raises(ValueError, match="X has zero variance after mean-centering"):
        transformer.fit(X, y)
