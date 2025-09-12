import numpy as np
import pytest
from sklearn.utils.estimator_checks import check_estimator

from chemotools.augmentation import AddNoise


# Test compliance with scikit-learn
def test_compliance_add_noise():
    # Arrange
    transformer = AddNoise()
    # Act & Assert
    check_estimator(transformer)


# Test functionality
def test_add_noise_exponential():
    # Arrange
    spectrum = np.ones(10000).reshape(1, -1)
    add_noise = AddNoise(distribution="exponential", scale=0.1, random_state=42)

    # Act
    spectrum_corrected = add_noise.fit_transform(spectrum)

    # Assert
    assert spectrum.shape == spectrum_corrected.shape
    assert np.allclose(np.mean(spectrum_corrected[0]) - 1, 0.1, atol=1e-2)


def test_add_noise_gaussian():
    # Arrange
    spectrum = np.ones(10000).reshape(1, -1)
    add_noise = AddNoise(distribution="gaussian", scale=0.5, random_state=42)

    # Act
    spectrum_corrected = add_noise.fit_transform(spectrum)

    # Assert
    assert spectrum.shape == spectrum_corrected.shape
    assert np.allclose(np.mean(spectrum_corrected[0]) - 1, 0, atol=1e-2)
    assert np.allclose(np.std(spectrum_corrected[0]), 0.5, atol=1e-2)


def test_add_noise_poisson():
    # Arrange
    spectrum = np.ones(10000).reshape(1, -1)
    add_noise = AddNoise(distribution="poisson", scale=0.5, random_state=42)

    # Act
    spectrum_corrected = add_noise.fit_transform(spectrum)

    # Assert
    assert spectrum.shape == spectrum_corrected.shape
    assert np.allclose(np.mean(spectrum_corrected[0]), 1.5011, atol=1e-2)
    assert np.allclose(np.std(spectrum_corrected[0]), 0.5, atol=1e-2)


def test_invalid_noise_distribution():
    # Arrange
    spectrum = np.ones(10000).reshape(1, -1)

    # Act
    add_noise = AddNoise(distribution="invalid", scale=0.5, random_state=42)

    # Assert
    with pytest.raises(ValueError, match="Invalid noise distribution.*"):
        add_noise.fit_transform(spectrum)
