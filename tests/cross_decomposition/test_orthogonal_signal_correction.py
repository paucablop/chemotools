"""Tests for OrthogonalSignalCorrection."""

from sklearn.utils.estimator_checks import check_estimator

from chemotools.cross_decomposition import OrthogonalSignalCorrection


# Test compliance with scikit-learn
def test_compliance_osc():
    # Arrange
    transformer = OrthogonalSignalCorrection()
    # Act & Assert
    check_estimator(transformer)
