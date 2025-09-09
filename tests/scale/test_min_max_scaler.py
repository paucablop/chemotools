import numpy as np
from sklearn.utils.estimator_checks import check_estimator

from chemotools.scale import MinMaxScaler


# Test compliance with scikit-learn
def test_compliance_min_max_norm():
    # Arrange
    transformer = MinMaxScaler()
    # Act & Assert
    check_estimator(transformer)


# Test functionality
def test_max_scaler(spectrum):
    # Arrange
    max_norm = MinMaxScaler(use_min=False)

    # Act
    spectrum_corrected = max_norm.fit_transform(spectrum)

    # Assert
    assert np.allclose(
        spectrum_corrected[0], spectrum[0] / np.max(spectrum[0]), atol=1e-8
    )


def test_min_norm(spectrum):
    # Arrange
    min_norm = MinMaxScaler()

    # Act
    spectrum_corrected = min_norm.fit_transform(spectrum)

    # Assert
    assert np.allclose(
        spectrum_corrected[0],
        (spectrum[0] - np.min(spectrum[0]))
        / (np.max(spectrum[0]) - np.min(spectrum[0])),
        atol=1e-8,
    )
