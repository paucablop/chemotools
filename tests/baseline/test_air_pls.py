import numpy as np
from sklearn.utils.estimator_checks import check_estimator

from chemotools.baseline import AirPls


# Test compliance with scikit-learn
def test_compliance_air_pls():
    # Arrange
    transformer = AirPls()
    # Act & Assert
    check_estimator(transformer)


# Test functionality
def test_air_pls(spectrum, reference_airpls):
    # Arrange
    air_pls = AirPls()

    # Act
    spectrum_corrected = air_pls.fit_transform(spectrum)

    # Assert
    assert np.allclose(spectrum_corrected[0], reference_airpls[0], atol=1e-7)
