import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from chemotools.utils.check_inputs import check_input


class WhittakerSmooth(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        lam: float = 1e2,
        differences: int = 1,
    ):
        self.lam = lam
        self.differences = differences

    def fit(self, X: np.ndarray, y=None) -> "WhittakerSmooth":
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

        # Check that the number of features is the same as the fitted data
        if X_.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features but got {X_.shape[1]}")

        # Calculate the whittaker smooth
        for i, x in enumerate(X_):
            X_[i] = self._calculate_whittaker_smooth(x)
            
        return X_.reshape(-1, 1) if X_.ndim == 1 else X_

    def _calculate_whittaker_smooth(self, x):
        x = np.asarray(x)
        n = len(x)
        D = np.diff(np.eye(n), self.differences)
        w = np.ones(n)
        for i in range(self.differences+1):
            W = np.diag(w) + 1e-8*np.eye(n)
            Z = W + self.lam * np.dot(D, D.T)
            z = np.linalg.solve(Z, w * x)
            w = np.sqrt(np.maximum(z, 0))
        return z
