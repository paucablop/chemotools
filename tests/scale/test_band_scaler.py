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


def test_band_scaler_with_mean_and_baseline_correction():
    """Test that BandScaler correctly scales the spectrum using mean aggregation with
    baseline correction."""
    # Arrange
    spectra = np.array([[1.0, 1.0, 2.0, 3.0, 2.0, 1.0, 1.0]])
    x_axis = np.array([100, 200, 300, 400, 500, 600, 700])

    # The band includes features 1:6
    baseline = np.ones_like(spectra)
    band_y = spectra[0, 1:6] - baseline[0, 1:6]
    scaling_factor = band_y.mean()
    reference_spectra = spectra / scaling_factor

    # Act
    scaler = BandScaler(
        start=200,
        end=700,
        x_axis=x_axis,
        aggregation="mean",
        baseline_correction=True,
    )
    spectra_scaled = scaler.fit_transform(spectra)

    # Assert
    assert np.allclose(spectra_scaled, reference_spectra, atol=1e-8)


def test_band_scaler_with_mean_and_baseline_correction_nonuniform_x_axis():
    """Test that BandScaler baseline correction uses actual x-axis spacing, not
    index-based spacing, so that non-uniform x-axis grids are handled correctly."""
    # Arrange: non-uniformly spaced x-axis with 6 points
    spectra = np.array([[0.0, 1.0, 3.0, 2.0, 4.0, 5.0]])
    x_axis = np.array([0.0, 1.0, 3.0, 4.0, 10.0, 11.0])

    # Band: start=1.0 (start_index_=1), end=10.0 (end_index_=4, exclusive).
    # BandScaler slices
    # X[:, start_index_:end_index_] = X[:, 1:4] → indices [1,2,3] → x=[1,3,4]
    band_y = spectra[0, 1:4]
    band_x = x_axis[1:4]  # [1.0, 3.0, 4.0] — non-uniform spacing

    # Expected baseline: linear in x (not linear in index)
    # t = (x - x[0]) / (x[-1] - x[0]) = [0, 2/3, 1] ≠ linspace(0, 1, 3) = [0, 0.5, 1]
    t = (band_x - band_x[0]) / (band_x[-1] - band_x[0])
    baseline_values = band_y[0] + t * (band_y[-1] - band_y[0])
    band_y_corrected = band_y - baseline_values
    scaling_factor = band_y_corrected.mean()
    reference_spectra = spectra / scaling_factor

    # Act
    scaler = BandScaler(
        start=1.0,
        end=10.0,
        x_axis=x_axis,
        aggregation="mean",
        baseline_correction=True,
    )
    spectra_scaled = scaler.fit_transform(spectra)

    # Assert
    assert np.allclose(spectra_scaled, reference_spectra, atol=1e-8)


def test_band_scaler_with_area_and_baseline_correction():
    """Test that BandScaler correctly scales the spectrum using area aggregation with
    baseline correction."""
    # Arrange
    spectra = np.array([[1.0, 1.0, 2.0, 3.0, 2.0, 1.0, 1.0]])
    x_axis = np.array([100, 200, 300, 400, 500, 600, 700])

    # The band includes features 1:6
    baseline = np.ones_like(spectra)
    band_y = spectra[0, 1:6] - baseline[0, 1:6]
    trapz_func = getattr(np, "trapezoid", getattr(np, "trapz", None))
    scaling_factor = trapz_func(band_y, x=x_axis[1:6], axis=0)
    reference_spectra = spectra / scaling_factor

    # Act
    scaler = BandScaler(
        start=200,
        end=700,
        x_axis=x_axis,
        aggregation="area",
        baseline_correction=True,
    )
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
