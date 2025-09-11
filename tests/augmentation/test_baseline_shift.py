import numpy as np
from sklearn.utils.estimator_checks import check_estimator

from chemotools.augmentation import BaselineShift


# Test compliance with scikit-learn
def test_compliance_baseline_shift():
    # Arrange
    transformer = BaselineShift()
    # Act & Assert
    check_estimator(transformer)


# Test functionality
def test_baseline_shift():
    # Arrange
    spectrum = np.ones(100).reshape(1, -1)
    baseline_shift = BaselineShift(scale=1, random_state=42)

    # Act
    spectrum_corrected = baseline_shift.fit_transform(spectrum)

    # Assert
    assert spectrum.shape == spectrum_corrected.shape
    assert np.mean(spectrum_corrected[0]) > np.mean(spectrum[0])
    assert np.isclose(np.std(spectrum_corrected[0]), 0.0, atol=1e-8)
    assert np.isclose(
        np.mean(spectrum_corrected[0]) - np.mean(spectrum[0]),
        0.37454011884736227,
        atol=1e-8,
    )
