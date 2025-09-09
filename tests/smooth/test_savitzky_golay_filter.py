from sklearn.utils.estimator_checks import check_estimator

from chemotools.smooth import SavitzkyGolayFilter


# Test compliance with scikit-learn
def test_compliance_savitzky_golay_filter():
    # Arrange
    transformer = SavitzkyGolayFilter()
    # Act & Assert
    check_estimator(transformer)


# Test functionality
# TODO: Add functionality tests
