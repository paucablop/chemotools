import numpy as np
import pytest
from sklearn.utils.estimator_checks import check_estimator

from chemotools.scale import ParetoScaler


# Test compliance with scikit-learn
def test_compliance_pareto_scaler():
    # Arrange
    transformer = ParetoScaler()
    # Act & Assert
    check_estimator(transformer)


# Test functionality
def test_pareto_scaler_with_mean():
    """Test that ParetoScaler correctly scales the spectrum."""
    # Arrange
    # Create a simple spectrum with known mean and std
    spectra = np.array([[1.0, 2.0, 3.0, 4.0, 5.0], [2.0, 3.0, 4.0, 5.0, 6.0]])
    mean = np.mean(spectra, axis=0)
    scale = np.power(np.std(spectra, axis=0, ddof=0), 0.5)

    # reference spectrum after Pareto scaling: (x - mean) / np.power(std, 0.5)
    reference_spectra = (spectra - mean) / scale

    # Act
    scaler = ParetoScaler()
    spectra_corrected = scaler.fit_transform(spectra)

    # Assert
    assert np.allclose(spectra_corrected, reference_spectra, atol=1e-8)


def test_pareto_scaler_without_mean():
    """Test that ParetoScaler correctly scales the spectrum without centering."""
    # Arrange
    spectra = np.array([[1.0, 2.0, 3.0, 4.0, 5.0], [2.0, 3.0, 4.0, 5.0, 6.0]])
    scale = np.power(np.std(spectra, axis=0, ddof=0), 0.5)

    # reference spectrum after Pareto scaling without mean: x / np.power(std, 0.5)
    reference_spectra = spectra / scale

    # Act
    scaler = ParetoScaler(with_mean=False)
    spectra_corrected = scaler.fit_transform(spectra)

    # Assert
    assert np.allclose(spectra_corrected, reference_spectra, atol=1e-8)


def test_raises_warning_zero_std():
    """Test that ParetoScaler raises a warning when a feature has zero std."""
    # Arrange
    spectra = np.array([[1.0, 2.0, 3.0], [1.0, 3.0, 4.0]])  # First feature has zero std

    # Act & Assert
    with pytest.warns(
        UserWarning,
        match=r"The scale for 1 feature\(s\) is zero \(constant columns\)\.",
    ):
        ParetoScaler().fit(spectra)


def test_inverse_transform():
    """Test that inverse_transform correctly recovers the original data."""
    # Arrange
    spectra = np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
    scaler = ParetoScaler()

    # Act
    spectra_scaled = scaler.fit_transform(spectra)
    spectra_recovered = scaler.inverse_transform(spectra_scaled)

    # Assert
    assert np.allclose(spectra_recovered, spectra, atol=1e-8)
