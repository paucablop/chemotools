from sklearn.utils.estimator_checks import check_estimator

from chemotools.baseline import AirPls, ConstantBaselineCorrection, CubicSplineCorrection, LinearCorrection, NonNegative, PolynomialCorrection, SubtractReference
from chemotools.derivative import NorrisWilliams, SavitzkyGolay
from chemotools.scale import IndexScaler, MinMaxScaler, NormScaler
from chemotools.scatter import MultiplicativeScatterCorrection, StandardNormalVariate
from chemotools.smooth import MeanFilter, MedianFilter, SavitzkyGolayFilter, WhittakerSmooth
from chemotools.variable_selection import RangeCut

from tests.fixtures import spectrum


# AirPls
def test_compliance_air_pls():
    # Arrange
    transformer = AirPls()
    # Act & Assert
    check_estimator(transformer)

# ConstantBaselineCorrection
def test_compliance_constant_baseline_correction():
    # Arrange
    transformer = ConstantBaselineCorrection()
    # Act & Assert
    check_estimator(transformer)

# CubicSplineCorrection
def test_compliance_cubic_spline_correction():
    # Arrange
    transformer = CubicSplineCorrection()
    # Act & Assert
    check_estimator(transformer)

# IndexScaler
def test_compliance_index_scaler():
    # Arrange
    transformer = IndexScaler()
    # Act & Assert
    check_estimator(transformer)

# LinearCorrection
def test_compliance_linear_correction():
    # Arrange
    transformer = LinearCorrection()
    # Act & Assert
    check_estimator(transformer)

# LNormalize
def test_compliance_l_norm():
    # Arrange
    transformer = NormScaler()
    # Act & Assert
    check_estimator(transformer)

# MeanFilter
def test_compliance_mean_filter():
    # Arrange
    transformer = MeanFilter()
    # Act & Assert
    check_estimator(transformer)

# MedianFilter
def test_compliance_median_filter():
    # Arrange
    transformer = MedianFilter()
    # Act & Assert
    check_estimator(transformer)

# MinMaxNormalize
def test_compliance_min_max_norm():
    # Arrange
    transformer = MinMaxScaler()
    # Act & Assert
    check_estimator(transformer)

# MultiplicativeScatterCorrection
def test_compliance_multiplicative_scatter_correction():
    # Arrange
    transformer = MultiplicativeScatterCorrection()
    # Act & Assert
    check_estimator(transformer)

# NonNegative
def test_compliance_non_negative():
    # Arrange
    transformer = NonNegative()
    # Act & Assert
    check_estimator(transformer)

# NorrisWilliams
def test_compliance_norris_williams():
    # Arrange
    transformer = NorrisWilliams()
    # Act & Assert
    check_estimator(transformer)

# NorrisWilliams
def test_compliance_norris_williams_2():
    # Arrange
    transformer = NorrisWilliams(derivative_order=2)
    # Act & Assert
    check_estimator(transformer)


# PolynomialCorrection
def test_compliance_polynomial_correction():
    # Arrange
    transformer = PolynomialCorrection()
    # Act & Assert
    check_estimator(transformer)

# SavitzkyGolay
def test_compliance_savitzky_golay():
    # Arrange
    transformer = SavitzkyGolay()
    # Act & Assert
    check_estimator(transformer)

# SavitzkyGolayFilter
def test_compliance_savitzky_golay_filter():
    # Arrange
    transformer = SavitzkyGolayFilter()
    # Act & Assert
    check_estimator(transformer)

# StandardNormalVariate
def test_compliance_standard_normal_variate():
    # Arrange
    transformer = StandardNormalVariate()
    # Act & Assert
    check_estimator(transformer)

# RangeCut
def test_compliance_range_cut():
    # Arrange
    transformer = RangeCut()
    # Act & Assert
    check_estimator(transformer)

# SubtractReference
def test_compliance_subtract_reference():
    # Arrange
    transformer = SubtractReference()
    # Act & Assert
    check_estimator(transformer)

# WhittakerSmooth
def test_compliance_whittaker_smooth():
    # Arrange
    transformer = WhittakerSmooth()
    # Act & Assert
    check_estimator(transformer)
