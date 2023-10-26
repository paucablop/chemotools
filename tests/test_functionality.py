import numpy as np
import pytest

from chemotools.augmenation import (
    BaselineShift,
    ExponentialNoise, 
    IndexShift,
    NormalNoise, 
    SpectrumScale,
    UniformNoise,
)

from chemotools.baseline import (
    AirPls,
    ArPls,
    ConstantBaselineCorrection,
    LinearCorrection,
    NonNegative,
    SubtractReference,
)
from chemotools.derivative import NorrisWilliams, SavitzkyGolay
from chemotools.scale import MinMaxScaler, NormScaler, PointScaler
from chemotools.scatter import (
    ExtendedMultiplicativeScatterCorrection,
    MultiplicativeScatterCorrection,
    RobustNormalVariate,
    StandardNormalVariate,
)
from chemotools.smooth import MeanFilter, MedianFilter, WhittakerSmooth
from chemotools.variable_selection import RangeCut, SelectFeatures
from tests.fixtures import (
    spectrum,
    spectrum_arpls,
    reference_airpls,
    reference_arpls,
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
    assert np.allclose(spectrum_corrected[0], reference_airpls[0], atol=1e-7)


def test_ar_pls(spectrum_arpls, reference_arpls):
    # Arrange
    arpls = ArPls(1e2, 0.0001)
    reference = np.array(spectrum_arpls) - np.array(reference_arpls)

    # Act
    spectrum_corrected = arpls.fit_transform(spectrum_arpls)

    # Assert
    assert np.allclose(spectrum_corrected[0], reference[0], atol=1e-4)


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
    assert np.isclose(np.mean(spectrum_corrected[0]) - np.mean(spectrum[0]), 0.77395605, atol=1e-8)


def test_constant_baseline_correction():
    # Arrange
    spectrum = np.array([1, 1, 1, 1, 1, 1, 1, 2, 2, 1]).reshape(1, -1)
    constant_baseline_correction = ConstantBaselineCorrection(start=7, end=8)

    # Act
    spectrum_corrected = constant_baseline_correction.fit_transform(spectrum)

    # Assert
    expected = np.array([-1, -1, -1, -1, -1, -1, -1, 0, 0, -1])
    assert np.allclose(spectrum_corrected[0], expected, atol=1e-8)


def test_constant_baseline_correction_with_wavenumbers():
    # Arrange
    spectrum = np.array([1, 1, 1, 1, 1, 1, 1, 2, 2, 1]).reshape(1, -1)
    wavenumbers = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    constant_baseline_correction = ConstantBaselineCorrection(
        start=9, end=10, wavenumbers=wavenumbers
    )

    # Act
    spectrum_corrected = constant_baseline_correction.fit_transform(spectrum)

    # Assert
    expected = np.array([-1, -1, -1, -1, -1, -1, -1, 0, 0, -1])
    assert np.allclose(spectrum_corrected[0], expected, atol=1e-8)


def test_exponential_noise():
    # Arrange
    spectrum = np.ones(10000).reshape(1, -1)
    exponential_noise = ExponentialNoise(scale=0.1, random_state=42)

    # Act
    spectrum_corrected = exponential_noise.fit_transform(spectrum)

    # Assert
    assert spectrum.shape == spectrum_corrected.shape
    assert np.allclose(np.mean(spectrum_corrected[0])-1, 0.1, atol=1e-2)



def test_extended_baseline_correction():
    # Arrange
    spectrum = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]).reshape(
        1, -1
    )
    reference = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    emsc = ExtendedMultiplicativeScatterCorrection(reference=reference)

    # Act
    spectrum_emsc = emsc.fit_transform(spectrum)

    # Assert
    assert np.allclose(spectrum_emsc[0], reference, atol=1e-8)


def test_extended_baseline_correction_with_weights():
    # Arrange
    spectrum = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]).reshape(
        1, -1
    )
    reference = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    emsc = ExtendedMultiplicativeScatterCorrection(reference=reference, weights=weights)

    # Act
    spectrum_emsc = emsc.fit_transform(spectrum)

    # Assert
    assert np.allclose(spectrum_emsc[0], reference, atol=1e-8)


def test_extended_baseline_correction_with_no_reference():
    # Arrange
    spectrum = np.array([1.0, 2.0, 3.0, 4.0, 5.0]).reshape(1, -1)

    # Act
    emsc = ExtendedMultiplicativeScatterCorrection(use_mean=False, use_median=False)

    # Assert
    with pytest.raises(ValueError):
        emsc.fit_transform(spectrum)



def test_extended_baseline_correction_with_wrong_reference():
    # Arrange
    spectrum = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]).reshape(
        1, -1
    )
    reference = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    # Act
    emsc = ExtendedMultiplicativeScatterCorrection(reference=reference)

    # Assert
    with pytest.raises(ValueError):
        emsc.fit_transform(spectrum)


def test_extended_baseline_correction_with_wrong_weights():
    # Arrange
    spectrum = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]).reshape(
        1, -1
    )
    weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    # Act
    emsc = ExtendedMultiplicativeScatterCorrection(weights=weights)

    # Assert
    with pytest.raises(ValueError):
        emsc.fit_transform(spectrum)


def test_extended_baseline_correction_with_noreference_no_median_no_mean():
    # Arrange
    emsc = ExtendedMultiplicativeScatterCorrection(use_mean=False)

    # Act & Assert
    with pytest.raises(ValueError):
        emsc.fit_transform(spectrum)


def test_extended_baseline_correction_through_msc(spectrum):
    # EMSC of 0 order should be equivalient to MSC
    # Arrange
    msc = MultiplicativeScatterCorrection()
    emsc = ExtendedMultiplicativeScatterCorrection(order=0)

    # Act
    spectrum_msc = msc.fit_transform(spectrum)
    spectrum_emsc = emsc.fit_transform(spectrum)

    # Assert
    assert np.allclose(spectrum_emsc[0], spectrum_msc, atol=1e-8)


def test_extended_baseline_correction_through_msc_median(spectrum):
    # EMSC of 0 order should be equivalient to MSC
    # Arrange
    msc = MultiplicativeScatterCorrection(use_median=True)
    emsc = ExtendedMultiplicativeScatterCorrection(order=0, use_median=True)

    # Act
    spectrum_msc = msc.fit_transform(spectrum)
    spectrum_emsc = emsc.fit_transform(spectrum)

    # Assert
    assert np.allclose(spectrum_emsc[0], spectrum_msc, atol=1e-8)


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
    max_norm = MinMaxScaler(use_min=False)

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


def test_multiplicative_scatter_correction_mean(spectrum, reference_msc_mean):
    # Arrange
    msc = MultiplicativeScatterCorrection()

    # Act
    spectrum_corrected = msc.fit_transform(spectrum)

    # Assert
    assert np.allclose(spectrum_corrected[0], reference_msc_mean[0], atol=1e-8)


def test_multiplicative_scatter_correction_with_reference(spectrum, reference_msc_mean):
    # Arrange
    msc = MultiplicativeScatterCorrection(reference=reference_msc_mean[0])

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
        reference=reference_msc_median[0], use_median=True
    )

    # Act
    spectrum_corrected = msc.fit_transform(spectrum)

    # Assert
    assert np.allclose(spectrum_corrected[0], reference_msc_median[0], atol=1e-8)


def test_multiplicative_scatter_correction_with_weights(spectrum, reference_msc_mean):
    # Arrange
    weights = np.ones(len(spectrum[0]))

    msc = MultiplicativeScatterCorrection(weights=weights)

    # Act
    spectrum_corrected = msc.fit_transform(spectrum)

    # Assert
    assert np.allclose(spectrum_corrected[0], reference_msc_mean[0], atol=1e-8)


def test_multiplicative_scatter_correction_with_wrong_weights(
    spectrum, reference_msc_mean
):
    # Arrange
    weights = np.ones(10)
    msc = MultiplicativeScatterCorrection(weights=weights)

    # Act & Assert
    with pytest.raises(ValueError):
        msc.fit_transform(spectrum)


def test_multiplicative_scatter_correction_with_wrong_reference(spectrum):
    # Arrange
    reference = np.ones(10)
    msc = MultiplicativeScatterCorrection(reference=reference)

    # Act & Assert
    with pytest.raises(ValueError):
        msc.fit_transform(spectrum)


def test_multiplicative_scatter_correction_no_mean_no_median_no_reference(spectrum):
    # Arrange
    reference = np.ones(10)
    msc = MultiplicativeScatterCorrection(use_mean=False)

    # Act & Assert
    with pytest.raises(ValueError):
        msc.fit_transform(spectrum)


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


def test_normal_noise():
    # Arrange
    spectrum = np.ones(10000).reshape(1, -1)
    normal_noise = NormalNoise(scale=0.5, random_state=42)

    # Act
    spectrum_corrected = normal_noise.fit_transform(spectrum)

    # Assert
    assert spectrum.shape == spectrum_corrected.shape
    assert np.allclose(np.mean(spectrum_corrected[0])-1, 0, atol=1e-2)
    assert np.allclose(np.std(spectrum_corrected[0]), 0.5, atol=1e-2)


def test_norris_williams_filter_1():
    # Arrange
    norris_williams_filter = NorrisWilliams()
    array = np.ones((1, 10)).reshape(1, -1)

    # Act
    spectrum_corrected = norris_williams_filter.fit_transform(array)

    # Assert
    assert np.allclose(spectrum_corrected[0], np.zeros((1, 10)), atol=1e-2)


def test_norris_williams_filter_2():
    # Arrange
    norris_williams_filter = NorrisWilliams(derivative_order=2)
    array = np.ones((1, 10)).reshape(1, -1)

    # Act
    spectrum_corrected = norris_williams_filter.fit_transform(array)

    # Assert
    assert np.allclose(spectrum_corrected[0], np.zeros((1, 10)), atol=1e-2)


def test_norris_williams_wrong_filter():
    # Arrange
    norris_williams_filter = NorrisWilliams(derivative_order=5)
    array = np.ones((1, 10)).reshape(1, -1)

    # Act & Assert

    with pytest.raises(ValueError):
        norris_williams_filter.fit_transform(array)


def test_point_scaler(spectrum):
    # Arrange
    index_scaler = PointScaler(point=0)
    reference_spectrum = [value / spectrum[0][0] for value in spectrum[0]]

    # Act
    spectrum_corrected = index_scaler.fit_transform(spectrum)

    # Assert
    assert np.allclose(spectrum_corrected[0], reference_spectrum, atol=1e-8)


def test_point_scaler_with_wavenumbers():
    # Arrange
    wavenumbers = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    spectrum = np.array([[10.0, 12.0, 14.0, 16.0, 14.0, 12.0, 10.0, 12.0, 14.0, 16.0]])

    # Act
    index_scaler = PointScaler(point=4, wavenumbers=wavenumbers)
    spectrum_corrected = index_scaler.fit_transform(spectrum)

    # Assert
    assert np.allclose(spectrum_corrected[0], spectrum[0] / spectrum[0][3], atol=1e-8)


def test_range_cut_by_index(spectrum):
    # Arrange
    range_cut = RangeCut(start=0, end=10)

    # Act
    spectrum_corrected = range_cut.fit_transform(spectrum)

    # Assert
    assert np.allclose(spectrum_corrected[0], spectrum[0][:10], atol=1e-8)


def test_range_cut_by_wavenumber():
    # Arrange
    wavenumbers = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    spectrum = np.array([[10, 12, 14, 16, 14, 12, 10, 12, 14, 16]])
    range_cut = RangeCut(start=2.5, end=7.9, wavenumbers=wavenumbers)

    # Act
    spectrum_corrected = range_cut.fit_transform(spectrum)

    # Assert
    assert np.allclose(spectrum_corrected[0], spectrum[0][1:7], atol=1e-8)


def test_range_cut_by_wavenumber_2():
    # Arrange
    wavenumbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    spectrum = np.array([[10, 12, 14, 16, 14, 12, 10, 12, 14, 16]])
    range_cut = RangeCut(start=2.5, end=7.9, wavenumbers=wavenumbers)

    # Act
    spectrum_corrected = range_cut.fit_transform(spectrum)

    # Assert
    assert np.allclose(spectrum_corrected[0], spectrum[0][1:7], atol=1e-8)


def test_robust_normal_variate():
    # Arrange
    spectrum = np.array([2, 3.5, 5, 27, 8, 9]).reshape(1, -1)
    reference = np.array([-2.5, -0.5, 1.5, 30.833333, 5.5, 6.83333333])
    rnv = RobustNormalVariate()

    # Act
    spectrum_corrected = rnv.fit_transform(spectrum)

    # Assert
    assert np.allclose(spectrum_corrected[0], reference, atol=1e-8)


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

    array = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]).reshape(1, -1)

    # Act
    spectrum_corrected = savitzky_golay_filter.fit_transform(array)

    # Assert
    assert np.allclose(spectrum_corrected[0], np.ones((1, 10)), atol=1e-2)


def test_select_features():
    # Arrange
    spectrum = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])

    # Act
    select_features = SelectFeatures()
    spectrum_corrected = select_features.fit_transform(spectrum)

    # Assert
    assert np.allclose(spectrum_corrected[0], spectrum[0], atol=1e-8)


def test_select_features_with_index():
    # Arrange
    spectrum = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    expected = np.array([[1, 2, 3, 8, 9, 10]])

    # Act
    select_features = SelectFeatures(features=np.array([0, 1, 2, 7, 8, 9]))
    spectrum_corrected = select_features.fit_transform(spectrum)

    # Assert
    assert np.allclose(spectrum_corrected[0], expected, atol=1e-8)


def test_select_features_with_wavenumbers():
    # Arrange
    wavenumbers = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    spectrum = np.array([[1.0, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0, 34.0, 55.0, 89.0]])
    expected = np.array([[1.0, 2.0, 3.0, 34.0, 55.0, 89.0]])

    # Act
    select_features = SelectFeatures(
        features=np.array([1, 2, 3, 8, 9, 10]), wavenumbers=wavenumbers
    )
    spectrum_corrected = select_features.fit_transform(spectrum)

    # Assert
    assert np.allclose(spectrum_corrected[0], expected, atol=1e-8)


def test_index_shift():
    # Arrange
    spectrum = np.array([[1, 1, 1, 1, 1, 2, 1, 1, 1, 1]])
    spectrum_shift = IndexShift(shift=1, random_state=42)

    # Act
    spectrum_corrected = spectrum_shift.fit_transform(spectrum)

    # Assert
    assert spectrum_corrected[0][4] == 2


def test_spectrum_scale(spectrum):
    # Arrange
    spectrum_scale = SpectrumScale(scale=0.01, random_state=42)

    # Act
    spectrum_corrected = spectrum_scale.fit_transform(spectrum)

    # Assert
    assert np.allclose(spectrum_corrected[0], spectrum[0], atol=0.01)


def test_standard_normal_variate(spectrum, reference_snv):
    # Arrange
    snv = StandardNormalVariate()

    # Act
    spectrum_corrected = snv.fit_transform(spectrum)

    # Assert
    assert np.allclose(spectrum_corrected[0], reference_snv[0], atol=1e-2)


def test_subtract_reference(spectrum):
    # Arrange
    baseline = SubtractReference(reference=spectrum)

    # Act
    spectrum_corrected = baseline.fit_transform(spectrum)

    # Assert
    assert np.allclose(spectrum_corrected[0], np.zeros(len(spectrum)), atol=1e-8)


def test_subtract_reference_without_reference(spectrum):
    # Arrange
    baseline = SubtractReference()

    # Act
    spectrum_corrected = baseline.fit_transform(spectrum)

    # Assert
    assert np.allclose(spectrum_corrected[0], spectrum, atol=1e-8)


def test_uniform_noise():
    # Arrange
    spectrum = np.ones(10000).reshape(1, -1)
    uniform_noise = UniformNoise(low=-1, high=1, random_state=42)

    # Act
    spectrum_corrected = uniform_noise.fit_transform(spectrum)

    # Assert
    assert spectrum.shape == spectrum_corrected.shape
    assert np.allclose(np.mean(spectrum_corrected[0])-1, 0, atol=1e-2)
    assert np.allclose(np.std(spectrum_corrected[0]), np.sqrt(1/3), atol=1e-2)


def test_whitakker_smooth(spectrum, reference_whitakker):
    # Arrange
    whitakker_smooth = WhittakerSmooth()

    # Act
    spectrum_corrected = whitakker_smooth.fit_transform(spectrum)

    # Assert
    assert np.allclose(spectrum_corrected[0], reference_whitakker[0], atol=1e-8)
