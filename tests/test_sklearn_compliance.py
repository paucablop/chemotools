from sklearn.utils.estimator_checks import check_estimator

from chemotools.baseline.air_pls import AirPls
from chemotools.derivative.savitzky_golay import SavitzkyGolay
from chemotools.normalize.max_norm import MaxNorm
from chemotools.scattering import MultiplicativeScatterCorrection, StandardNormalVariate
from chemotools.smoothing import SavitzkyGolayFilter, WhittakerSmooth


# AirPls
def test_compliance_air_pls():
    # Arrange
    transformer = AirPls()
    # Act & Assert
    check_estimator(transformer)


def test_compliance_max_norm():
    # Arrange
    transformer = MaxNorm()
    # Act & Assert
    check_estimator(transformer)

# MultiplicativeScatterCorrection
def test_compliance_multiplicative_scatter_correction():
    # Arrange
    transformer = MultiplicativeScatterCorrection()
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


# WhittakerSmooth
def test_compliance_whittaker_smooth():
    # Arrange
    transformer = WhittakerSmooth()
    # Act & Assert
    check_estimator(transformer)
