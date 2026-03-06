import numpy as np
import pytest
from sklearn.utils.estimator_checks import check_estimator

from chemotools.derivative import NorrisWilliams


# Test compliance with scikit-learn
def test_compliance_norris_williams_der_1():
    # Arrange
    transformer = NorrisWilliams()
    # Act & Assert
    check_estimator(transformer)


def test_compliance_norris_williams_der_2():
    # Arrange
    transformer = NorrisWilliams(deriv=2)
    # Act & Assert
    check_estimator(transformer)


# Test functionality
def test_norris_williams_filter_1():
    # Arrange
    norris_williams_filter = NorrisWilliams()
    array = np.ones((1, 10)).reshape(1, -1)

    # Act
    spectrum_corrected = norris_williams_filter.fit_transform(array)

    # Assert
    assert np.allclose(spectrum_corrected[0], np.zeros((1, 10)), atol=1e-2)


def test_norris_williams_filter_2():
    # Arrange
    norris_williams_filter = NorrisWilliams(deriv=2)
    array = np.ones((1, 10)).reshape(1, -1)

    # Act
    spectrum_corrected = norris_williams_filter.fit_transform(array)

    # Assert
    assert np.allclose(spectrum_corrected[0], np.zeros((1, 10)), atol=1e-2)


def test_norris_williams_wrong_filter():
    # Arrange
    norris_williams_filter = NorrisWilliams(deriv=5)
    array = np.ones((1, 10)).reshape(1, -1)

    # Act & Assert
    with pytest.raises(ValueError):
        norris_williams_filter.fit_transform(array)


# --- Deprecation tests ---
def test_norris_williams_window_size_deprecated():
    """Using the old `window_size` parameter emits a FutureWarning."""
    # Arrange
    array = np.ones((1, 10))
    nw = NorrisWilliams(window_size=5)

    # Act
    with pytest.warns(FutureWarning, match="window_size"):
        nw.fit(array)

    # Assert
    assert nw.window_length_ == 5


def test_norris_williams_derivative_order_deprecated():
    """Using the old `derivative_order` parameter emits a FutureWarning."""
    # Arrange
    array = np.ones((1, 10))
    nw = NorrisWilliams(derivative_order=2)

    # Act
    with pytest.warns(FutureWarning, match="derivative_order"):
        nw.fit(array)

    # Assert
    assert nw.deriv_ == 2


def test_norris_williams_window_size_conflict():
    """Passing both `window_length` and `window_size` raises ValueError."""
    # Arrange
    array = np.ones((1, 10))
    nw = NorrisWilliams(window_length=7, window_size=7)

    # Act
    with pytest.raises(ValueError) as exc_info:
        nw.fit(array)

    # Assert
    assert "Only one of" in str(exc_info.value)


def test_norris_williams_derivative_order_conflict():
    """Passing both `deriv` and `derivative_order` raises ValueError."""
    # Arrange
    array = np.ones((1, 10))
    nw = NorrisWilliams(deriv=2, derivative_order=2)

    # Act
    with pytest.raises(ValueError) as exc_info:
        nw.fit(array)

    # Assert
    assert "Only one of" in str(exc_info.value)
