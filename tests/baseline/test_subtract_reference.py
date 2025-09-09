import numpy as np
from sklearn.utils.estimator_checks import check_estimator

from chemotools.baseline import SubtractReference


# Test compliance with scikit-learn
def test_compliance_subtract_reference():
    # Arrange
    transformer = SubtractReference()
    # Act & Assert
    check_estimator(transformer)


# Test functionality
def test_subtract_reference(spectrum):
    # Arrange
    baseline = SubtractReference(reference=spectrum)

    # Act
    spectrum_corrected = baseline.fit_transform(spectrum)

    # Assert
    assert np.allclose(spectrum_corrected[0], np.zeros(len(spectrum)), atol=1e-8)


def test_subtract_reference_without_reference(spectrum):
    # Arrange
    baseline = SubtractReference()

    # Act
    spectrum_corrected = baseline.fit_transform(spectrum)

    # Assert
    assert np.allclose(spectrum_corrected[0], spectrum, atol=1e-8)
