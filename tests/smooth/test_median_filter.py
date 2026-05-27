import numpy as np
import pytest
from sklearn.utils.estimator_checks import check_estimator

from chemotools.smooth import MedianFilter


# Test compliance with scikit-learn
def test_compliance_median_filter():
    # Arrange
    transformer = MedianFilter()
    # Act & Assert
    check_estimator(transformer)


# Test functionality
def test_median_filter():
    # Arrange
    array = np.array([[1.0, 2.0, 30.0, 4.0, 5.0]])
    median_filter = MedianFilter(window_length=3)

    # Act
    array_corrected = median_filter.fit_transform(array)

    # Assert
    assert np.allclose(array_corrected[0], [1, 2.0, 4.0, 5.0, 5.0], atol=1e-8)


def test_median_filter_snapshot_current_behavior():
    """Snapshot test to lock current transform output for nearest mode."""
    # Arrange
    mf = MedianFilter(window_length=3, mode="nearest")
    X = np.array(
        [
            [0.0, 10.0, 2.0, 8.0, 4.0, 6.0],
            [6.0, 4.0, 8.0, 2.0, 10.0, 0.0],
        ],
        dtype=np.float64,
    )
    expected = np.array(
        [
            [0.0, 2.0, 8.0, 4.0, 6.0, 6.0],
            [6.0, 6.0, 4.0, 8.0, 2.0, 0.0],
        ],
        dtype=np.float64,
    )

    # Act
    observed = mf.fit_transform(X)

    # Assert
    np.testing.assert_allclose(observed, expected, rtol=0.0, atol=1e-12)


def test_median_filter_repeated_transform_is_stable():
    # Arrange
    mf = MedianFilter(window_length=3, mode="nearest")
    X = np.array(
        [
            [1.0, 9.0, 2.0, 8.0, 3.0, 7.0],
            [7.0, 3.0, 8.0, 2.0, 9.0, 1.0],
        ],
        dtype=np.float64,
    )
    mf.fit(X)

    # Act
    observed_first = mf.transform(X)
    observed_second = mf.transform(X)

    # Assert
    np.testing.assert_allclose(observed_first, observed_second, rtol=0.0, atol=1e-12)


def test_median_filter_does_not_mutate_input():
    # Arrange
    mf = MedianFilter(window_length=3, mode="nearest")
    X = np.array([[0.0, 10.0, 2.0, 8.0, 4.0, 6.0]], dtype=np.float64)
    X_original = X.copy()

    # Act
    _ = mf.fit_transform(X)

    # Assert
    np.testing.assert_array_equal(X, X_original)


def test_median_filter_parallel_matches_serial():
    # Arrange
    X = np.array(
        [
            [0.0, 10.0, 2.0, 8.0, 4.0, 6.0],
            [6.0, 4.0, 8.0, 2.0, 10.0, 0.0],
            [5.0, 1.0, 9.0, 3.0, 7.0, 2.0],
        ],
        dtype=np.float64,
    )
    serial = MedianFilter(window_length=3, mode="nearest", n_jobs=1)
    parallel = MedianFilter(window_length=3, mode="nearest", n_jobs=2)

    # Act
    y_serial = serial.fit_transform(X)
    y_parallel = parallel.fit_transform(X)

    # Assert
    np.testing.assert_allclose(y_parallel, y_serial, rtol=0.0, atol=1e-12)


def test_median_filter_invalid_n_jobs_zero():
    # Arrange
    X = np.array([[1.0, 2.0, 30.0, 4.0, 5.0]], dtype=np.float64)
    mf = MedianFilter(window_length=3, n_jobs=0)

    # Act
    with pytest.raises(ValueError, match="n_jobs"):
        mf.fit(X)


# --- Deprecation tests ---
def test_median_filter_window_size_deprecated():
    """Using the old `window_size` parameter emits a FutureWarning."""
    # Arrange
    array = np.array([[1.0, 2.0, 30.0, 4.0, 5.0]])
    mf = MedianFilter(window_size=3)

    # Act
    with pytest.warns(FutureWarning, match="window_size"):
        mf.fit(array)

    # Assert
    assert mf.window_length_ == 3


def test_median_filter_window_size_conflict():
    """Passing both `window_length` and `window_size` raises ValueError."""
    # Arrange
    array = np.array([[1.0, 2.0, 30.0, 4.0, 5.0]])
    mf = MedianFilter(window_length=5, window_size=5)

    # Act
    with pytest.raises(ValueError) as exc_info:
        mf.fit(array)

    # Assert
    assert "Only one of" in str(exc_info.value)
