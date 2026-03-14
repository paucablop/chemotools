import numpy as np
import pytest
from sklearn.utils.estimator_checks import check_estimator

from chemotools.scale import BandScaler


# Test compliance with scikit-learn
def test_compliance_band_scaler():
    # Arrange
    transformer = BandScaler()
    # Act & Assert
    check_estimator(transformer)


# Test functionality
def test_band_scaler_with_mean():
    """Test that BandScaler correctly scales the spectrum using mean aggregation."""
    # Arrange
    spectra = np.array([[1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0]])

    # The band includes features 2:4
    scaling_factor = spectra[0, 2:4].mean()
    reference_spectra = spectra / scaling_factor

    # Act
    scaler = BandScaler(start=2, end=4)
    spectra_scaled = scaler.fit_transform(spectra)

    # Assert
    assert np.allclose(spectra_scaled, reference_spectra, atol=1e-8)


def test_band_scaler_with_area():
    """Test that BandScaler correctly scales the spectrum using area aggregation."""
    # Arrange
    spectra = np.array([[1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0]])
    x_axis = np.array([100, 200, 300, 400, 500, 600, 700])

    # The band includes features 2:4
    trapz_func = getattr(np, "trapezoid", getattr(np, "trapz", None))
    scaling_factor = trapz_func(spectra[0, 2:4], x=x_axis[2:4], axis=0)
    reference_spectra = spectra / scaling_factor

    # Act
    scaler = BandScaler(start=300, end=500, x_axis=x_axis, aggregation="area")
    spectra_scaled = scaler.fit_transform(spectra)

    # Assert
    assert np.allclose(spectra_scaled, reference_spectra, atol=1e-8)


def test_raises_error_start_larger_than_end():
    """Test that BandScaler raises an error when start is larger than end."""
    # Arrange
    spectra = np.array([[1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0]])

    # Act & Assert
    with pytest.raises(
        ValueError,
        match=r"start_index_ \(4\) must be less than end_index_ \(2\)\.",
    ):
        scaler = BandScaler(start=4, end=2)
        scaler.fit(spectra)


def test_raises_error_area_no_axis():
    """Test BandScaler raises an error when aggregation is area but no x_axis
    is provided."""
    # Arrange
    spectra = np.array([[1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0]])

    # Act & Assert
    with pytest.raises(
        ValueError,
        match=r"x_axis must be provided when aggregation='area'.",
    ):
        scaler = BandScaler(start=300, end=500, aggregation="area")
        scaler.fit(spectra)


def test_raises_warning_zero_band():
    """Test that BandScaler raises a warning when the band has zero mean or area."""
    # Arrange
    spectra = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    x_axis = np.array([100, 200, 300, 400, 500, 600, 700])

    # Act & Assert
    with pytest.warns(
        UserWarning,
        match=r"The scaling factor for sample\(s\) \[0\] is zero\. These samples will "
        "not be scaled\.",
    ):
        scaler = BandScaler(start=300, end=500, x_axis=x_axis, aggregation="area")
        scaler.fit_transform(spectra)
