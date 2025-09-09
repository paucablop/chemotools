import numpy as np
from sklearn.cross_decomposition import PLSRegression

from chemotools.outliers import Leverage


def test_leverage_calculation(dummy_data_loader):
    # Arrange
    X, y = dummy_data_loader
    pls = PLSRegression(n_components=1).fit(X, y)

    # Act
    leverage = Leverage(pls, confidence=0.95).fit(X)
    leverages = leverage.predict_residuals(X)

    # Assert
    assert np.all(leverages >= 0), "Leverage values should be positive"
    assert np.sum(leverages) == 1, "Sum of leverage values should be 1"
    assert np.isclose(
        np.mean(leverages), 1 / len(X)
    ), "Mean of leverage values should be 1/n_samples"
    assert np.isclose(leverages[0], 0.02940591986082612), "Leverage value mismatch"
    assert np.isclose(leverages[-1], 0.02936313351948305), "Leverage value mismatch"
