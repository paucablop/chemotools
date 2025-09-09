import numpy as np
from sklearn.utils.estimator_checks import check_estimator

from chemotools.augmentation import SpectrumScale


# Test compliance with scikit-learn
def test_compliance_spectrum_scale():
    # Arrange
    transformer = SpectrumScale()
    # Act & Assert
    check_estimator(transformer)


# Test functionality
def test_spectrum_scale(spectrum):
    # Arrange
    spectrum_scale = SpectrumScale(scale=0.01, random_state=42)

    # Act
    spectrum_corrected = spectrum_scale.fit_transform(spectrum)

    # Assert
    assert np.allclose(spectrum_corrected[0], spectrum[0], atol=0.01)
