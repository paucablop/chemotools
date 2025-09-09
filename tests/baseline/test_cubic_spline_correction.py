from sklearn.utils.estimator_checks import check_estimator

from chemotools.baseline import CubicSplineCorrection


# Test compliance with scikit-learn
def test_compliance_cubic_spline_correction():
    # Arrange
    transformer = CubicSplineCorrection()
    # Act & Assert
    check_estimator(transformer)


# Test functionality
# TODO: Add a real test for CubicSplineCorrection
