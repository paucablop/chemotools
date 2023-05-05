import logging
import numpy as np
from scipy.sparse import spdiags, csc_matrix
from scipy.sparse.linalg import splu

from sklearn.base import BaseEstimator, TransformerMixin, OneToOneFeatureMixin
from sklearn.utils.validation import check_is_fitted

from chemotools.utils.check_inputs import check_input

logger = logging.getLogger(__name__)

# This code is adapted from the following source:
# Sung-June Baek a, Aaron Park *a, Young-Jin Ahn a and Jaebum Choo
# Baseline correction using asymmetrically reweighted penalized least squares smoothing


class ArPls(OneToOneFeatureMixin, BaseEstimator, TransformerMixin):
    def __init__(
        self,
        lam: int = 1e4,
        ratio: int = 0.01,
        nr_iterations: int = 100,
    ):
        self.lam = lam
        self.ratio = ratio
        self.nr_iterations = nr_iterations

    def fit(self, X: np.ndarray, y=None) -> "ArPls":
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
            raise ValueError(
                f"Expected {self.n_features_in_} features but got {X_.shape[1]}"
            )

        # Calculate the ar pls baseline
        for i, x in enumerate(X_):
            X_[i] = x - self._calculate_ar_pls(x)

        return X_.reshape(-1, 1) if X_.ndim == 1 else X_

    def _calculate_ar_pls(self, x):
        N = len(x)
        D = np.diff(np.eye(N), 2)
        H = self.lam * np.dot(D, D.T)
        w = np.ones(N)
        iteration = 0
        while iteration < self.nr_iterations:
            W = spdiags(w, 0, N, N)
            C = csc_matrix(W + H)
            z = splu(C).solve(w*x)
            d = x - z
            dn = d[d<0]
            if len(dn) == 0:
                break
            m = np.mean(dn) 
            s = np.std(dn)   
            exponent = np.clip(2* (d-(2*s-m))/s, -709, 709)
            wt = 1.0 / (1.0 + np.exp(exponent))
            if np.linalg.norm(w-wt)/np.linalg.norm(w) < self.ratio:
                break
            w = wt
            iteration += 1
        return z
