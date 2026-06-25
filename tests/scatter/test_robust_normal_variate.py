import numpy as np
import pytest
from sklearn.utils._param_validation import InvalidParameterError
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


def test_robust_normal_variate_snapshot_current_behavior():
    """Snapshot test to lock current transform output for default settings."""
    # Arrange
    rnv = RobustNormalVariate()
    X = np.array(
        [
            [2.0, 3.5, 5.0, 27.0, 8.0, 9.0],
            [9.0, 8.0, 27.0, 5.0, 3.5, 2.0],
            [1.0, 2.0, 2.0, 3.0, 8.0, 13.0],
        ],
        dtype=np.float64,
    )
    expected = np.array(
        [
            [-2.5, -0.5, 1.5, 30.833333333333, 5.5, 6.833333333333],
            [6.833333333333, 5.5, 30.833333333333, 1.5, -0.5, -2.5],
            [
                -2.12132034356,
                0.0,
                0.0,
                2.12132034356,
                12.727922061358,
                23.334523779157,
            ],
        ],
        dtype=np.float64,
    )

    # Act
    observed = rnv.fit_transform(X)

    # Assert
    np.testing.assert_allclose(observed, expected, rtol=0.0, atol=1e-8)


def test_rnv_zero_denom_warns():
    # Arrange
    X = np.array([[1.0, 1.0, 1.0, 2.0]])
    rnv = RobustNormalVariate(percentile=50).fit(X)

    # Act & Assert
    with pytest.warns(
        UserWarning, match="Denominator is zero in RNV. Adding epsilon to avoid NaNs."
    ):
        rnv.transform(X)


def test_rnv_percentile_above_100_rejected():
    # Arrange
    X = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)
    rnv = RobustNormalVariate(percentile=101)

    # Act & Assert
    with pytest.raises(Exception, match="percentile"):
        rnv.fit(X)


def test_rnv_parallel_matches_serial():
    # Arrange
    rng = np.random.default_rng(42)
    X = rng.normal(size=(24, 64))
    serial = RobustNormalVariate(n_jobs=1)
    parallel = RobustNormalVariate(n_jobs=2)

    # Act
    y_serial = serial.fit_transform(X)
    y_parallel = parallel.fit_transform(X)

    # Assert
    np.testing.assert_allclose(y_parallel, y_serial, rtol=0.0, atol=1e-12)


def test_rnv_n_jobs_minus_one_runs():
    # Arrange
    X = np.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]], dtype=np.float64)
    rnv = RobustNormalVariate(n_jobs=-1)

    # Act
    y = rnv.fit_transform(X)

    # Assert
    assert y.shape == X.shape


def test_rnv_invalid_n_jobs_zero_rejected():
    # Arrange
    X = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)
    rnv = RobustNormalVariate(n_jobs=0)

    # Act & Assert
    with pytest.raises((InvalidParameterError, ValueError), match="n_jobs"):
        rnv.fit(X)


def test_rnv_legacy_state_without_n_jobs():
    # Arrange: simulate a pickle that pre-dates the n_jobs attribute
    legacy = RobustNormalVariate()
    legacy_state = {k: v for k, v in legacy.__dict__.items() if k != "n_jobs"}
    restored = RobustNormalVariate()

    # Act
    restored.__setstate__(legacy_state)

    # Assert
    assert restored.n_jobs == 1
    assert restored.get_params(deep=False)["n_jobs"] == 1
