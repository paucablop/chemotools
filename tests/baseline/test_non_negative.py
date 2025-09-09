import numpy as np
from sklearn.utils.estimator_checks import check_estimator

from chemotools.baseline import NonNegative


# Test compliance with scikit-learn
def test_compliance_non_negative():
    # Arrange
    transformer = NonNegative()
    # Act & Assert
    check_estimator(transformer)


# Test functionality
def test_non_negative_zeroes():
    # Arrange
    spectrum = np.array([[-1, 0, 1]])
    non_negative = NonNegative(mode="zero")

    # Act
    spectrum_corrected = non_negative.fit_transform(spectrum)

    # Assert
    assert np.allclose(spectrum_corrected[0], [0, 0, 1], atol=1e-8)


def test_non_negative_absolute():
    # Arrange
    spectrum = np.array([[-1, 0, 1]])
    non_negative = NonNegative(mode="abs")

    # Act
    spectrum_corrected = non_negative.fit_transform(spectrum)

    # Assert
    assert np.allclose(spectrum_corrected[0], [1, 0, 1], atol=1e-8)
