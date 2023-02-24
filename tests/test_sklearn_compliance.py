from sklearn.utils.estimator_checks import check_estimator
from chemotools.derivative.savitzky_golay import SavitzkyGolay
from chemotools.scattering import MultiplicativeSignalCorrection, StandardNormalVariate
from chemotools.smoothing import SavitzkyGolayFilter


# MultiplicativeSignalCorrection
def test_compliance_multiplicative_scatter_correction():
    # Arrange
    transformer = MultiplicativeSignalCorrection()
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
