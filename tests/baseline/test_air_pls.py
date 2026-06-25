import numpy as np
import pytest
from sklearn.utils._param_validation import InvalidParameterError
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


def test_air_pls_parallel_matches_serial():
    # Arrange
    rng = np.random.default_rng(17)
    X = rng.normal(size=(14, 100)) + 2.0
    serial = AirPls(lam=100, nr_iterations=10, n_jobs=1)
    parallel = AirPls(lam=100, nr_iterations=10, n_jobs=2)

    # Act
    y_serial = serial.fit_transform(X)
    y_parallel = parallel.fit_transform(X)

    # Assert
    np.testing.assert_allclose(y_parallel, y_serial, rtol=0.0, atol=1e-10)


def test_air_pls_n_jobs_minus_one_runs():
    # Arrange
    X = np.array([[1.0, 2.0, 5.0, 2.0, 1.0]], dtype=np.float64)
    air_pls = AirPls(n_jobs=-1)

    # Act
    y = air_pls.fit_transform(X)

    # Assert
    assert y.shape == X.shape


def test_air_pls_invalid_n_jobs_zero_rejected():
    # Arrange
    X = np.array([[1.0, 2.0, 5.0, 2.0, 1.0]], dtype=np.float64)
    air_pls = AirPls(n_jobs=0)

    # Act & Assert
    with pytest.raises((InvalidParameterError, ValueError), match="n_jobs"):
        air_pls.fit(X)


def test_air_pls_legacy_state_without_n_jobs():
    # Arrange: simulate a pickle that pre-dates the n_jobs attribute
    legacy = AirPls()
    legacy_state = {k: v for k, v in legacy.__dict__.items() if k != "n_jobs"}
    restored = AirPls()

    # Act
    restored.__setstate__(legacy_state)

    # Assert
    assert restored.n_jobs == 1
    assert restored.get_params(deep=False)["n_jobs"] == 1
