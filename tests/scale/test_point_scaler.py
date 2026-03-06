import numpy as np
import pytest
from sklearn.utils.estimator_checks import check_estimator

from chemotools.scale import PointScaler


# Test compliance with scikit-learn
def test_compliance_point_scaler():
    # Arrange
    transformer = PointScaler()
    # Act & Assert
    check_estimator(transformer)


# Test functionality
def test_point_scaler(spectrum):
    # Arrange
    index_scaler = PointScaler(point=0)
    reference_spectrum = [value / spectrum[0][0] for value in spectrum[0]]

    # Act
    spectrum_corrected = index_scaler.fit_transform(spectrum)

    # Assert
    assert np.allclose(spectrum_corrected[0], reference_spectrum, atol=1e-8)


def test_point_scaler_with_wavenumbers():
    # Arrange
    wavenumbers = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    spectrum = np.array([[10.0, 12.0, 14.0, 16.0, 14.0, 12.0, 10.0, 12.0, 14.0, 16.0]])

    # Act
    index_scaler = PointScaler(point=4, x_axis=wavenumbers)
    spectrum_corrected = index_scaler.fit_transform(spectrum)

    # Assert
    assert np.allclose(spectrum_corrected[0], spectrum[0] / spectrum[0][3], atol=1e-8)


# --- Deprecation tests ---
def test_point_scaler_wavenumbers_deprecated():
    """Using the old `wavenumbers` parameter emits a FutureWarning."""
    # Arrange
    wavenumbers = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    spectrum = np.array([[10.0, 12.0, 14.0, 16.0, 14.0]])
    scaler = PointScaler(point=2, wavenumbers=wavenumbers)

    # Act
    with pytest.warns(FutureWarning, match="wavenumbers"):
        scaler.fit(spectrum)


def test_point_scaler_wavenumbers_conflict():
    """Passing both `x_axis` and `wavenumbers` raises ValueError."""
    # Arrange
    wavenumbers = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    spectrum = np.array([[10.0, 12.0, 14.0, 16.0, 14.0]])
    scaler = PointScaler(point=2, x_axis=wavenumbers, wavenumbers=wavenumbers)

    # Act
    with pytest.raises(ValueError) as exc_info:
        scaler.fit(spectrum)

    # Assert
    assert "Only one of" in str(exc_info.value)
