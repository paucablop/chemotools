import numpy as np
import pytest
from sklearn.utils._param_validation import InvalidParameterError
from sklearn.utils.estimator_checks import check_estimator

from chemotools.baseline import AsLs


# Test compliance with scikit-learn
def test_compliance_as_ls():
    # Arrange
    transformer = AsLs()
    # Act & Assert
    check_estimator(transformer)


def test_compliance_as_ls_sparse():
    # Arrange
    transformer = AsLs(solver_type="sparse")
    # Act & Assert
    check_estimator(transformer)


# Test functionality
def test_as_ls_banded(spectrum, reference_asls):
    # Arrange
    as_ls = AsLs(solver_type="banded")

    # Act
    spectrum_corrected = as_ls.fit_transform(spectrum)

    # Assert
    assert np.allclose(spectrum_corrected[0], reference_asls[0], atol=1e-4)


def test_as_ls_sparse(spectrum, reference_asls):
    # Arrange
    as_ls = AsLs(solver_type="sparse")

    # Act
    spectrum_corrected = as_ls.fit_transform(spectrum)

    # Assert
    assert np.allclose(spectrum_corrected[0], reference_asls[0], atol=1e-4)


def test_as_ls_parallel_matches_serial():
    # Arrange
    rng = np.random.default_rng(31)
    X = rng.normal(size=(12, 100)) + 2.0
    serial = AsLs(n_jobs=1)
    parallel = AsLs(n_jobs=2)

    # Act
    y_serial = serial.fit_transform(X)
    y_parallel = parallel.fit_transform(X)

    # Assert
    np.testing.assert_allclose(y_parallel, y_serial, rtol=0.0, atol=1e-10)


def test_as_ls_n_jobs_minus_one_runs():
    # Arrange
    X = np.array([[1.0, 2.0, 5.0, 2.0, 1.0]], dtype=np.float64)
    as_ls = AsLs(n_jobs=-1)

    # Act
    y = as_ls.fit_transform(X)

    # Assert
    assert y.shape == X.shape


def test_as_ls_invalid_n_jobs_zero_rejected():
    # Arrange
    X = np.array([[1.0, 2.0, 5.0, 2.0, 1.0]], dtype=np.float64)
    as_ls = AsLs(n_jobs=0)

    # Act & Assert
    with pytest.raises((InvalidParameterError, ValueError), match="n_jobs"):
        as_ls.fit(X)


def test_as_ls_legacy_state_without_n_jobs():
    # Arrange: reproduce what pickle.loads does for an old pickle that pre-dates
    # the n_jobs attribute — object.__new__ creates the instance without calling
    # __init__, then __setstate__ receives the old (n_jobs-free) state dict.
    legacy = AsLs()
    legacy_state = {k: v for k, v in legacy.__dict__.items() if k != "n_jobs"}
    restored = object.__new__(AsLs)

    # Act
    restored.__setstate__(legacy_state)

    # Assert
    assert restored.n_jobs == 1
    assert restored.get_params(deep=False)["n_jobs"] == 1
