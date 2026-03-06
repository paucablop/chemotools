import numpy as np
import pytest
from sklearn.utils.estimator_checks import check_estimator

from chemotools.smooth import MeanFilter


# Test compliance with scikit-learn
def test_compliance_mean_filter():
    # Arrange
    transformer = MeanFilter()
    # Act & Assert
    check_estimator(transformer)


# Test functionality
def test_mean_filter():
    # Arrange
    array = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
    mean_filter = MeanFilter(window_length=2)

    # Act
    array_corrected = mean_filter.fit_transform(array)

    # Assert
    assert np.allclose(array_corrected[0], [1, 1.5, 2.5, 3.5, 4.5], atol=1e-8)


# --- Deprecation tests ---
def test_mean_filter_window_size_deprecated():
    """Using the old `window_size` parameter emits a FutureWarning."""
    # Arrange
    array = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
    mf = MeanFilter(window_size=3)

    # Act
    with pytest.warns(FutureWarning, match="window_size"):
        mf.fit(array)

    # Assert
    assert mf.window_length_ == 3


def test_mean_filter_window_size_conflict():
    """Passing both `window_length` and `window_size` raises ValueError."""
    # Arrange
    array = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
    mf = MeanFilter(window_length=5, window_size=5)

    # Act
    with pytest.raises(ValueError) as exc_info:
        mf.fit(array)

    # Assert
    assert "Only one of" in str(exc_info.value)
