import numpy as np
import pytest
from sklearn.utils._param_validation import InvalidParameterError
from sklearn.utils.estimator_checks import check_estimator

from chemotools.smooth import WhittakerSmooth


# Test compliance with scikit-learn
def test_compliance_whittaker_smooth():
    # Arrange
    transformer = WhittakerSmooth()
    # Act & Assert
    check_estimator(transformer)


def test_compliance_whittaker_smooth_sparse():
    # Arrange
    transformer = WhittakerSmooth(solver_type="sparse")
    # Act & Assert
    check_estimator(transformer)


# Test functionality
def test_whittaker_smooth_banded(spectrum, reference_whittaker):
    # Arrange
    whittaker_smooth = WhittakerSmooth()

    # Act
    spectrum_corrected = whittaker_smooth.fit_transform(spectrum)

    # Assert
    assert np.allclose(spectrum_corrected[0], reference_whittaker[0], atol=1e-8)


def test_whittaker_smooth_sparse(spectrum, reference_whittaker):
    # Arrange
    whittaker_smooth = WhittakerSmooth(solver_type="sparse")

    # Act
    spectrum_corrected = whittaker_smooth.fit_transform(spectrum)

    # Assert
    assert np.allclose(spectrum_corrected[0], reference_whittaker[0], atol=1e-8)


def test_whittaker_smooth_banded_multi_row_matches_single_row():
    # Arrange: stack multiple independent copies of the same spectrum with distinct
    # row values so that a row/column transposition would produce wrong results
    rng = np.random.default_rng(42)
    n_samples, n_features = 20, 200
    X = rng.normal(size=(n_samples, n_features))

    ws = WhittakerSmooth()
    ws.fit(X)

    # Act: transform all rows at once (batch path)
    X_batch = ws.transform(X)

    # Reference: transform each row independently (single-row path)
    X_sequential = np.vstack([ws.transform(X[[i]]) for i in range(n_samples)])

    # Assert: every row must match the single-row result
    assert np.allclose(X_batch, X_sequential, atol=1e-10)


def test_whittaker_smooth_parallel_matches_serial():
    # Arrange
    rng = np.random.default_rng(5)
    X = rng.normal(size=(18, 120))
    serial = WhittakerSmooth(n_jobs=1)
    parallel = WhittakerSmooth(n_jobs=2)

    # Act
    y_serial = serial.fit_transform(X)
    y_parallel = parallel.fit_transform(X)

    # Assert
    np.testing.assert_allclose(y_parallel, y_serial, rtol=0.0, atol=1e-12)


def test_whittaker_smooth_sparse_parallel_matches_serial():
    # Arrange
    rng = np.random.default_rng(7)
    X = rng.normal(size=(14, 90))
    serial = WhittakerSmooth(solver_type="sparse", n_jobs=1)
    parallel = WhittakerSmooth(solver_type="sparse", n_jobs=2)

    # Act
    y_serial = serial.fit_transform(X)
    y_parallel = parallel.fit_transform(X)

    # Assert
    np.testing.assert_allclose(y_parallel, y_serial, rtol=0.0, atol=1e-12)


def test_whittaker_smooth_n_jobs_minus_one_runs():
    # Arrange
    X = np.array([[1.0, 2.0, 3.0, 2.0, 1.0]], dtype=np.float64)
    transformer = WhittakerSmooth(n_jobs=-1)

    # Act
    y = transformer.fit_transform(X)

    # Assert
    assert y.shape == X.shape


def test_whittaker_smooth_invalid_n_jobs_zero_rejected():
    # Arrange
    X = np.array([[1.0, 2.0, 5.0, 2.0, 1.0]], dtype=np.float64)
    zero_jobs = WhittakerSmooth(n_jobs=0)

    # Act & Assert
    with pytest.raises((InvalidParameterError, ValueError), match="n_jobs"):
        zero_jobs.fit(X)


def test_whittaker_smooth_legacy_state_without_n_jobs():
    # Arrange: simulate a pickle that pre-dates the n_jobs attribute
    legacy = WhittakerSmooth()
    legacy_state = {k: v for k, v in legacy.__dict__.items() if k != "n_jobs"}
    restored = WhittakerSmooth()

    # Act
    restored.__setstate__(legacy_state)

    # Assert
    assert restored.n_jobs == 1
    assert restored.get_params(deep=False)["n_jobs"] == 1
