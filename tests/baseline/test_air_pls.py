import numpy as np
from sklearn.utils.estimator_checks import check_estimator

from chemotools.baseline import AirPls


# Test compliance with scikit-learn
def test_compliance_air_pls():
    # Arrange
    transformer = AirPls()
    # Act & Assert
    check_estimator(transformer)


def test_compliance_air_pls_sparse():
    # Arrange
    transformer = AirPls(solver_type="sparse")
    # Act & Assert
    check_estimator(transformer)


# Test functionality
def test_air_pls_banded(spectrum, reference_airpls):
    # Arrange
    air_pls = AirPls(lam=100, nr_iterations=15, solver_type="banded")

    # Act
    spectrum_corrected = air_pls.fit_transform(spectrum)

    # Assert
    assert np.allclose(spectrum_corrected[0], reference_airpls[0], atol=1e-3)


def test_air_pls_sparse(spectrum, reference_airpls):
    # Arrange
    air_pls = AirPls(lam=100, nr_iterations=15, solver_type="sparse")

    # Act
    spectrum_corrected = air_pls.fit_transform(spectrum)

    # Assert
    assert np.allclose(spectrum_corrected[0], reference_airpls[0], atol=1e-3)
