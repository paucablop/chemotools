import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from chemotools.utils.check_inputs import check_input

class PolynomialCorrection(BaseEstimator, TransformerMixin):
    def __init__(self, order: int = 1, indices: list = None) -> None:
        self.order = order
        self.indices = indices

    def fit(self, X: np.ndarray, y=None) -> "PolynomialCorrection":
        # Check that X is a 2D array and has only finite values
        X = check_input(X)

        # Set the number of features
        self.n_features_in_ = X.shape[1]

        # Set the fitted attribute to True
        self._is_fitted = True

        if self.indices is None:
            self.indices_ = range(0, len(X[0]))
        else:
            self.indices_ = self.indices

        return self
    
    def transform(self, X: np.ndarray, y=0, copy=True) -> np.ndarray:
        # Check that the estimator is fitted
        check_is_fitted(self, "_is_fitted")

        # Check that X is a 2D array and has only finite values
        X = check_input(X)
        X_ = X.copy()

        # Check that the number of features is the same as the fitted data
        if X_.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features but got {X_.shape[1]}")

        # Calculate polynomial baseline correction
        for i, x in enumerate(X_):
            X_[i] = self._baseline_correct_spectrum(x)
        return X_.reshape(-1, 1) if X_.ndim == 1 else X_
    
    def _baseline_correct_spectrum(self, x: np.ndarray) -> np.ndarray:
        intensity = x[self.indices_]
        poly = np.polyfit(self.indices_, intensity, self.order)
        baseline = [np.polyval(poly, i) for i in range(0, len(x))]      
        return x - baseline