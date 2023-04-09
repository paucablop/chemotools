import logging
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from chemotools.utils.check_inputs import check_input

logger = logging.getLogger(__name__)

class AirPls(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        nr_iterations: int = 15,
        lam: int = 1e2,
        polynomial_order: int = 1,
    ):
        self.nr_iterations = nr_iterations
        self.lam = lam
        self.polynomial_order = polynomial_order

    def fit(self, X: np.ndarray, y=None) -> "AirPls":
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

        # Calculate the air pls smooth
        for i, x in enumerate(X_):
            X_[i] = x - self._calculate_air_pls(x)

        return X_.reshape(-1, 1) if X_.ndim == 1 else X_

    def _calculate_whittaker_smooth(self, x, w):
        x = np.asarray(x)
        n = len(x)
        D = np.diff(np.eye(n), self.polynomial_order)
        for i in range(self.polynomial_order):
            W = np.diag(w) + 1e-8*np.eye(n)
            Z = W + self.lam * np.dot(D, D.T)
            z = np.linalg.solve(Z, w * x)
            w = np.sqrt(np.maximum(z, 0))
        return z

    def _calculate_air_pls(self, x):
        m = x.shape[0]
        w = np.ones(m)

        for i in range(1, self.nr_iterations):
            z = self._calculate_whittaker_smooth(x, w)
            d = x - z
            dssn = np.abs(d[d<0].sum()) 

            if dssn < 0.001 * np.abs(x).sum(): 
                break
                
            if i == self.nr_iterations - 1:
                break

            w[d>=0]=0
            w[d<0] =  np.exp(i * np.abs(d[d < 0]) / dssn)

            negative_d = d[d < 0]
            if negative_d.size > 0:
                w[0] = np.exp(i * negative_d.max() / dssn)

            w[-1] = w[0]

        return z


