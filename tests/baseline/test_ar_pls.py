import numpy as np
import pytest
from sklearn.utils._param_validation import InvalidParameterError
from sklearn.utils.estimator_checks import check_estimator

from chemotools.baseline import ArPls


# Test compliance with scikit-learn
def test_compliance_ar_pls():
    # Arrange
    transformer = ArPls()
    # Act & Assert
    check_estimator(transformer)


def test_compliance_ar_pls_sparse():
    # Arrange
    transformer = ArPls(solver_type="sparse")
    # Act & Assert
    check_estimator(transformer)


# Test functionality
def test_ar_pls_banded(spectrum_arpls, reference_arpls):
    # Arrange
    arpls = ArPls(1e2, 0.0001, solver_type="banded")
    reference = np.array(spectrum_arpls) - np.array(reference_arpls)

    # Act
    spectrum_corrected = arpls.fit_transform(spectrum_arpls)

    # Assert
    assert np.allclose(spectrum_corrected[0], reference[0], atol=1e-4)


def test_ar_pls_sparse(spectrum_arpls, reference_arpls):
    # Arrange
    arpls = ArPls(1e2, 0.0001, solver_type="sparse")
    reference = np.array(spectrum_arpls) - np.array(reference_arpls)

    # Act
    spectrum_corrected = arpls.fit_transform(spectrum_arpls)

    # Assert
    assert np.allclose(spectrum_corrected[0], reference[0], atol=1e-4)


def test_ar_pls_parallel_matches_serial():
    # Arrange
    rng = np.random.default_rng(23)
    X = rng.normal(size=(12, 120)) + 3.0
    serial = ArPls(1e2, 1e-4, n_jobs=1)
    parallel = ArPls(1e2, 1e-4, n_jobs=2)

    # Act
    y_serial = serial.fit_transform(X)
    y_parallel = parallel.fit_transform(X)

    # Assert
    np.testing.assert_allclose(y_parallel, y_serial, rtol=0.0, atol=1e-10)


def test_ar_pls_n_jobs_minus_one_runs():
    # Arrange
    X = np.array([[1.0, 2.0, 5.0, 2.0, 1.0]], dtype=np.float64)
    arpls = ArPls(n_jobs=-1)

    # Act
    y = arpls.fit_transform(X)

    # Assert
    assert y.shape == X.shape


def test_ar_pls_invalid_n_jobs_zero_rejected():
    # Arrange
    X = np.array([[1.0, 2.0, 5.0, 2.0, 1.0]], dtype=np.float64)
    arpls = ArPls(n_jobs=0)

    # Act & Assert
    with pytest.raises((InvalidParameterError, ValueError), match="n_jobs"):
        arpls.fit(X)


def test_ar_pls_legacy_state_without_n_jobs():
    # Arrange: simulate a pickle that pre-dates the n_jobs attribute
    legacy = ArPls()
    legacy_state = {k: v for k, v in legacy.__dict__.items() if k != "n_jobs"}
    restored = ArPls()

    # Act
    restored.__setstate__(legacy_state)

    # Assert
    assert restored.n_jobs == 1
    assert restored.get_params(deep=False)["n_jobs"] == 1
