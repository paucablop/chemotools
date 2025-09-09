import numpy as np
from sklearn.utils.estimator_checks import check_estimator

from chemotools.baseline import ConstantBaselineCorrection


# Test compliance with scikit-learn
def test_compliance_constant_baseline_correction():
    # Arrange
    transformer = ConstantBaselineCorrection()
    # Act & Assert
    check_estimator(transformer)


# Test functionality
def test_constant_baseline_correction():
    # Arrange
    spectrum = np.array([1, 1, 1, 1, 1, 1, 1, 2, 2, 1]).reshape(1, -1)
    constant_baseline_correction = ConstantBaselineCorrection(start=7, end=8)

    # Act
    spectrum_corrected = constant_baseline_correction.fit_transform(spectrum)

    # Assert
    expected = np.array([-1, -1, -1, -1, -1, -1, -1, 0, 0, -1])
    assert np.allclose(spectrum_corrected[0], expected, atol=1e-8)


def test_constant_baseline_correction_with_wavenumbers():
    # Arrange
    spectrum = np.array([1, 1, 1, 1, 1, 1, 1, 2, 2, 1]).reshape(1, -1)
    wavenumbers = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    constant_baseline_correction = ConstantBaselineCorrection(
        start=9, end=10, wavenumbers=wavenumbers
    )

    # Act
    spectrum_corrected = constant_baseline_correction.fit_transform(spectrum)

    # Assert
    expected = np.array([-1, -1, -1, -1, -1, -1, -1, 0, 0, -1])
    assert np.allclose(spectrum_corrected[0], expected, atol=1e-8)
