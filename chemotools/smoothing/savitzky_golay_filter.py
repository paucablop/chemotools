import numpy as np
from scipy.signal import savgol_filter
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from chemotools.utils.check_inputs import check_input


class SavitzkyGolayFilter(BaseEstimator, TransformerMixin):
    def __init__(
        self, window_size: int = 3, polynomial_order: int = 1, mode: str = "nearest"
    ) -> None:
        self.window_size = window_size
        self.polynomial_order = polynomial_order
        self.mode = mode

    def fit(self, X: np.ndarray, y=None) -> "SavitzkyGolayFilter":
        # Check that X is a 2D array and has only finite values
        X = check_input(X)

        # Set the number of features
        self.n_features_in_ = X.shape[1]

        # Set the fitted attribute to True
        self._is_fitted = True

        return self

    def transform(self, X: np.ndarray, y=None) -> np.ndarray:
        # Check that the estimator is fitted
        check_is_fitted(self, "_is_fitted")

        # Check that X is a 2D array and has only finite values
        X = check_input(X)
        X_ = X.copy()

        if X_.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features but got {X_.shape[1]}"
            )

        # Calculate the standard normal variate
        for i, x in enumerate(X_):
            X_[i] = self._calculate_smoothing(x)

        return X_.reshape(-1, 1) if X_.ndim == 1 else X_

    def _calculate_smoothing(self, x) -> np.ndarray:
        return savgol_filter(
            x,
            self.window_size,
            self.polynomial_order,
            deriv=0,
            axis=0,
            mode=self.mode,
        )
