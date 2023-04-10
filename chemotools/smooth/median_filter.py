import numpy as np
from scipy.ndimage import median_filter
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from chemotools.utils.check_inputs import check_input


class MedianFilter(BaseEstimator, TransformerMixin):
    def __init__(self, window_size: int = 3, mode: str = 'nearest') -> None:
        self.window_size = window_size
        self.mode = mode

    def fit(self, X: np.ndarray, y=None) -> "MedianFilter":
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

        # Mean filter the data
        for i, x in enumerate(X_):
            X_[i] = median_filter(x, size=self.window_size, mode=self.mode)
        return X_.reshape(-1, 1) if X_.ndim == 1 else X_
