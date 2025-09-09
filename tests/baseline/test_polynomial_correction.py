from sklearn.utils.estimator_checks import check_estimator

from chemotools.baseline import PolynomialCorrection


# Test compliance with scikit-learn
def test_compliance_polynomial_correction():
    # Arrange
    transformer = PolynomialCorrection()
    # Act & Assert
    check_estimator(transformer)


# Test functionality
# TODO: Implement test for PolynomialCorrection
