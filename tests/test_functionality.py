import numpy as np

from chemotools.baseline import AirPls, LinearCorrection, NonNegative
from chemotools.derivative import NorrisWilliams, SavitzkyGolay
from chemotools.normalize import LNormalize, MinMaxNormalize
from chemotools.scattering import MultiplicativeScatterCorrection, StandardNormalVariate
from chemotools.smoothing import MeanFilter, MedianFilter, WhittakerSmooth
from tests.fixtures import (
    spectrum,
    reference_airpls,
    reference_msc_mean,
    reference_msc_median,
    reference_sg_15_2,
    reference_snv,
    reference_whitakker,
)


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


def test_linear_correction(spectrum):
    # Arrange
    linear_correction = LinearCorrection()

    # Act
    spectrum_corrected = linear_correction.fit_transform(spectrum)

    # Assert
    assert spectrum_corrected[0][0] == 0
    assert spectrum_corrected[-1][0] == 0


def test_max_norm(spectrum):
    # Arrange
    max_norm = MinMaxNormalize(norm="max")

    # Act
    spectrum_corrected = max_norm.fit_transform(spectrum)

    # Assert
    assert np.allclose(
        spectrum_corrected[0], spectrum[0] / np.max(spectrum[0]), atol=1e-8
    )


def test_mean_filter():
    # Arrange
    array = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
    mean_filter = MeanFilter(window_size=2)

    # Act
    array_corrected = mean_filter.fit_transform(array)

    # Assert
    assert np.allclose(array_corrected[0], [1, 1.5, 2.5, 3.5, 4.5], atol=1e-8)


def test_median_filter():
    # Arrange
    array = np.array([[1.0, 2.0, 30.0, 4.0, 5.0]])
    mean_filter = MedianFilter(window_size=3)

    # Act
    array_corrected = mean_filter.fit_transform(array)

    # Assert
    assert np.allclose(array_corrected[0], [1, 2.0, 4.0, 5.0, 5.0], atol=1e-8)


def test_min_norm(spectrum):
    # Arrange
    min_norm = MinMaxNormalize(norm="min")

    # Act
    spectrum_corrected = min_norm.fit_transform(spectrum)

    # Assert
    assert np.allclose(
        spectrum_corrected[0], spectrum[0] / np.min(spectrum[0]), atol=1e-8
    )


def test_multiplicative_scatter_correction_mean(spectrum, reference_msc_mean):
    # Arrange
    msc = MultiplicativeScatterCorrection()

    # Act
    spectrum_corrected = msc.fit_transform(spectrum)

    # Assert
    assert np.allclose(spectrum_corrected[0], reference_msc_mean[0], atol=1e-8)


def test_multiplicative_scatter_correction_with_reference(spectrum, reference_msc_mean):
    # Arrange
    msc = MultiplicativeScatterCorrection(reference=reference_msc_mean)

    # Act
    spectrum_corrected = msc.fit_transform(spectrum)

    # Assert
    assert np.allclose(spectrum_corrected[0], reference_msc_mean[0], atol=1e-8)


def test_multiplicative_scatter_correction_median(spectrum, reference_msc_median):
    # Arrange
    msc = MultiplicativeScatterCorrection(use_median=True)

    # Act
    spectrum_corrected = msc.fit_transform(spectrum)

    # Assert
    assert np.allclose(spectrum_corrected[0], reference_msc_median[0], atol=1e-8)


def test_multiplicative_scatter_correction_with_reference_median(
    spectrum, reference_msc_median
):
    # Arrange
    msc = MultiplicativeScatterCorrection(
        reference=reference_msc_median, use_median=True
    )

    # Act
    spectrum_corrected = msc.fit_transform(spectrum)

    # Assert
    assert np.allclose(spectrum_corrected[0], reference_msc_median[0], atol=1e-8)


def test_non_negative_zeroes():
    # Arrange
    spectrum = np.array([[-1, 0, 1]])
    non_negative = NonNegative(mode="zero")

    # Act
    spectrum_corrected = non_negative.fit_transform(spectrum)

    # Assert
    assert np.allclose(spectrum_corrected[0], [0, 0, 1], atol=1e-8)


def test_non_negative_absolute():
    # Arrange
    spectrum = np.array([[-1, 0, 1]])
    non_negative = NonNegative(mode="abs")

    # Act
    spectrum_corrected = non_negative.fit_transform(spectrum)

    # Assert
    assert np.allclose(spectrum_corrected[0], [1, 0, 1], atol=1e-8)

def test_norris_williams_filter_1():
    # Arrange
    norris_williams_filter = NorrisWilliams()
    array = np.ones((1, 10)).reshape(1, -1)

    # Act
    spectrum_corrected = norris_williams_filter.fit_transform(array)

    # Assert
    assert np.allclose(spectrum_corrected[0], np.zeros((1, 10)), atol=1e-2)


def test_savizky_golay_filter_1(spectrum, reference_sg_15_2):
    # Arrange
    savitzky_golay_filter = SavitzkyGolay(
        window_size=15, polynomial_order=2, derivate_order=1, mode="interp"
    )

    # Act
    spectrum_corrected = savitzky_golay_filter.fit_transform(spectrum)

    # Assert
    assert np.allclose(spectrum_corrected[0], reference_sg_15_2[0], atol=1e-2)

def test_saviszky_golay_filter_2():
    # Arrange
    savitzky_golay_filter = SavitzkyGolay(
        window_size=3, polynomial_order=2, derivate_order=1, mode="interp"
    )

    array = np.ones((1, 10)).reshape(1, -1)

    # Act
    spectrum_corrected = savitzky_golay_filter.fit_transform(array)

    # Assert
    assert np.allclose(spectrum_corrected[0], np.zeros((1, 10)), atol=1e-2)

def test_saviszky_golay_filter_3():
    # Arrange
    savitzky_golay_filter = SavitzkyGolay(
        window_size=3, polynomial_order=2, derivate_order=1, mode="interp"
    )

    array = np.array([0., 1., 2., 3., 4., 5., 6., 7., 8., 9.]).reshape(1, -1)

    # Act
    spectrum_corrected = savitzky_golay_filter.fit_transform(array)

    # Assert
    assert np.allclose(spectrum_corrected[0], np.ones((1, 10)), atol=1e-2)

def test_standard_normal_variate(spectrum, reference_snv):
    # Arrange
    snv = StandardNormalVariate()

    # Act
    spectrum_corrected = snv.fit_transform(spectrum)

    # Assert
    assert np.allclose(spectrum_corrected[0], reference_snv[0], atol=1e-2)


def test_whitakker_smooth(spectrum, reference_whitakker):
    # Arrange
    whitakker_smooth = WhittakerSmooth()

    # Act
    spectrum_corrected = whitakker_smooth.fit_transform(spectrum)

    # Assert
    assert np.allclose(spectrum_corrected[0], reference_whitakker[0], atol=1e-2)
