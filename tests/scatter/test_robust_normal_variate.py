import numpy as np
import pytest
from sklearn.utils.estimator_checks import check_estimator

from chemotools.scatter import RobustNormalVariate


# Test compliance with scikit-learn
def test_compliance_robust_normal_variate():
    # Arrange
    transformer = RobustNormalVariate()
    # Act & Assert
    check_estimator(transformer)


# Test functionality
def test_robust_normal_variate():
    # Arrange
    spectrum = np.array([2, 3.5, 5, 27, 8, 9]).reshape(1, -1)
    reference = np.array([-2.5, -0.5, 1.5, 30.833333, 5.5, 6.83333333])
    rnv = RobustNormalVariate()

    # Act
    spectrum_corrected = rnv.fit_transform(spectrum)

    # Assert
    assert np.allclose(spectrum_corrected[0], reference, atol=1e-8)


def test_rnv_zero_denom_warns():
    # Arrange
    X = np.array([[1.0, 1.0, 1.0, 2.0]])
    rnv = RobustNormalVariate(percentile=50).fit(X)

    # Act & Assert
    with pytest.warns(
        UserWarning, match="Denominator is zero in RNV. Adding epsilon to avoid NaNs."
    ):
        rnv.transform(X)
