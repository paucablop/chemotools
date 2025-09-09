from sklearn.utils.estimator_checks import check_estimator

from chemotools.baseline import LinearCorrection


# Test compliance with scikit-learn
def test_compliance_linear_correction():
    # Arrange
    transformer = LinearCorrection()
    # Act & Assert
    check_estimator(transformer)


# Test functionality
def test_linear_correction(spectrum):
    # Arrange
    linear_correction = LinearCorrection()

    # Act
    spectrum_corrected = linear_correction.fit_transform(spectrum)

    # Assert
    assert spectrum_corrected[0][0] == 0
    assert spectrum_corrected[-1][0] == 0
