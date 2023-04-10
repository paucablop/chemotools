import numpy as np
from scipy.interpolate import CubicSpline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from chemotools.utils.check_inputs import check_input

class CubicSplineCorrection(BaseEstimator, TransformerMixin):
    def __init__(self, indices: tuple = None) -> None:
        self.indices = indices

    def fit(self, X: np.ndarray, y=None) -> "CubicSplineCorrection":
        # Check that X is a 2D array and has only finite values
        X = check_input(X)

        # Set the number of features
        self.n_features_in_ = X.shape[1]

        # Set the fitted attribute to True
        self._is_fitted = True

        return self

    def transform(self, X: np.ndarray, y=None, copy=True):
        # Check that the estimator is fitted
        check_is_fitted(self, "_is_fitted")

        # Check that X is a 2D array and has only finite values
        X = check_input(X)
        X_ = X.copy()

        # Check that the number of features is the same as the fitted data
        if X_.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features but got {X_.shape[1]}")

        # Calculate spline baseline correction
        for i, x in enumerate(X_):
            X_[i] = self._spline_baseline_correct(x)
        return X_.reshape(-1, 1) if X_.ndim == 1 else X_

    def _spline_baseline_correct(self, x: np.ndarray) -> np.ndarray:
        if self.indices is None:
            indices = [0, len(x) - 1]
        else:
            indices = self.indices

        intensity = x[indices]  
        spl = CubicSpline(indices, intensity)
        baseline = spl(range(len(x)))      
        return x - baseline