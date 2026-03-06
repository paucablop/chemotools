import numpy as np
import pytest
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
        start=9, end=10, x_axis=wavenumbers
    )

    # Act
    spectrum_corrected = constant_baseline_correction.fit_transform(spectrum)

    # Assert
    expected = np.array([-1, -1, -1, -1, -1, -1, -1, 0, 0, -1])
    assert np.allclose(spectrum_corrected[0], expected, atol=1e-8)


# --- Deprecation tests ---
def test_constant_baseline_wavenumbers_deprecated():
    """Using the old `wavenumbers` parameter emits a FutureWarning."""
    # Arrange
    spectrum = np.array([1, 1, 1, 1, 1, 1, 1, 2, 2, 1]).reshape(1, -1)
    wavenumbers = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    cbc = ConstantBaselineCorrection(start=9, end=10, wavenumbers=wavenumbers)

    # Act
    with pytest.warns(FutureWarning, match="wavenumbers"):
        cbc.fit(spectrum)


def test_constant_baseline_wavenumbers_conflict():
    """Passing both `x_axis` and `wavenumbers` raises ValueError."""
    # Arrange
    spectrum = np.array([1, 1, 1, 1, 1, 1, 1, 2, 2, 1]).reshape(1, -1)
    wavenumbers = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    cbc = ConstantBaselineCorrection(
        start=9, end=10, x_axis=wavenumbers, wavenumbers=wavenumbers
    )

    # Act
    with pytest.raises(ValueError) as exc_info:
        cbc.fit(spectrum)

    # Assert
    assert "Only one of" in str(exc_info.value)
