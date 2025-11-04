import numpy as np
import pytest
from scipy.signal import savgol_filter
from sklearn.utils.estimator_checks import check_estimator

from chemotools.smooth import SavitzkyGolayFilter


# Test compliance with scikit-learn
def test_compliance_savitzky_golay_filter():
    # Arrange
    transformer = SavitzkyGolayFilter()
    # Act & Assert
    check_estimator(transformer)


# Test functionality against scipy using all different modes
@pytest.mark.parametrize("mode", ["mirror", "constant", "nearest", "wrap", "interp"])
@pytest.mark.parametrize(
    "x_name, x",
    [
        ("sine_wave", np.sin(np.linspace(0, 2 * np.pi, 100), dtype=np.float64)),
        ("random_noise", np.random.random(50).astype(np.float64)),
        ("linear_ramp", np.linspace(0, 10, 75, dtype=np.float64)),
        ("exponential_decay", np.exp(-np.linspace(0, 5, 80)).astype(np.float64)),
        (
            "step_function",
            np.concatenate([np.ones(25), np.zeros(25), np.ones(25)]).astype(np.float64),
        ),
    ],
)
def test_functionality_savitzky_golay_filter(mode, x_name, x):
    # Arrange
    # x is now passed as a parameter

    # Act
    expected = savgol_filter(x, window_length=3, polyorder=1, mode=mode)
    smoothed = SavitzkyGolayFilter(
        window_size=3, polynomial_order=1, mode=mode
    ).fit_transform(x.reshape(1, -1))[0]

    # Assert
    assert np.allclose(smoothed, expected), "Expected smoothed output did not match"
