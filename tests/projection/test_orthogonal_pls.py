"""Tests for OrthogonalPLS."""

from sklearn.utils.estimator_checks import check_estimator

from chemotools.projection import OrthogonalPLS


# Test complience with scikit-learn
def test_compliance_opls():
    """Check sklearn estimator compliance for the OPLS transformer."""
    # Arrange
    transformer = OrthogonalPLS()

    # Act & Assert
    check_estimator(transformer)
