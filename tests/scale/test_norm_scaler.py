import numpy as np
from sklearn.utils.estimator_checks import check_estimator

from chemotools.scale import NormScaler


# Test compliance with scikit-learn
def test_compliance_l_norm():
    # Arrange
    transformer = NormScaler()
    # Act & Assert
    check_estimator(transformer)


# Test functionality
def test_l1_norm(spectrum):
    # Arrange
    norm = 1
    l1_norm = NormScaler(l_norm=norm)
    spectrum_norm = np.linalg.norm(spectrum[0], ord=norm)

    # Act
    spectrum_corrected = l1_norm.fit_transform(spectrum)

    # Assert
    assert np.allclose(spectrum_corrected[0], spectrum[0] / spectrum_norm, atol=1e-8)


def test_l2_norm(spectrum):
    # Arrange
    norm = 2
    l1_norm = NormScaler(l_norm=norm)
    spectrum_norm = np.linalg.norm(spectrum[0], ord=norm)

    # Act
    spectrum_corrected = l1_norm.fit_transform(spectrum)

    # Assert
    assert np.allclose(spectrum_corrected[0], spectrum[0] / spectrum_norm, atol=1e-8)
