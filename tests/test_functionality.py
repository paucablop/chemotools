import numpy as np

from chemotools.baseline import AirPls
from chemotools.normalize import LNormalize, MinMaxNormalize
from chemotools.smoothing import WhittakerSmooth
from tests.fixtures import spectrum, reference_airpls, reference_whitakker


def test_air_pls(spectrum, reference_airpls):
    # Arrange
    air_pls = AirPls()

    # Act
    spectrum_corrected = air_pls.fit_transform(spectrum)

    # Assert
    assert np.allclose(spectrum_corrected[0], reference_airpls[0], atol=1e-8)


def test_l1_norm(spectrum):
    # Arrange
    norm = 1
    l1_norm = LNormalize(l_norm=norm)
    spectrum_norm = np.linalg.norm(spectrum[0], ord=norm)

    # Act
    spectrum_corrected = l1_norm.fit_transform(spectrum)

    # Assert
    assert np.allclose(spectrum_corrected[0], spectrum[0] / spectrum_norm, atol=1e-8)


def test_l2_norm(spectrum):
    # Arrange
    norm = 2
    l1_norm = LNormalize(l_norm=norm)
    spectrum_norm = np.linalg.norm(spectrum[0], ord=norm)

    # Act
    spectrum_corrected = l1_norm.fit_transform(spectrum)

    # Assert
    assert np.allclose(spectrum_corrected[0], spectrum[0] / spectrum_norm, atol=1e-8)


def test_max_norm(spectrum):
    # Arrange
    max_norm = MinMaxNormalize(norm='max')

    # Act
    spectrum_corrected = max_norm.fit_transform(spectrum)

    # Assert
    assert np.allclose(spectrum_corrected[0], spectrum[0] / np.max(spectrum[0]), atol=1e-8)


def test_min_norm(spectrum):
    # Arrange
    min_norm = MinMaxNormalize(norm='min')

    # Act
    spectrum_corrected = min_norm.fit_transform(spectrum)

    # Assert
    assert np.allclose(spectrum_corrected[0], spectrum[0] / np.min(spectrum[0]), atol=1e-8)


def test_whitakker_smooth(spectrum, reference_whitakker):
    # Arrange
    whitakker_smooth = WhittakerSmooth()

    # Act
    spectrum_corrected = whitakker_smooth.fit_transform(spectrum)

    # Assert
    assert np.allclose(spectrum_corrected[0], reference_whitakker[0], atol=1e-2)
