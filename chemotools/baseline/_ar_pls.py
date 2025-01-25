import logging
import numpy as np
import scipy.sparse as sp
from scipy.sparse import spdiags, csc_matrix
from scipy.sparse.linalg import splu

from sklearn.base import BaseEstimator, TransformerMixin, OneToOneFeatureMixin
from sklearn.utils.validation import check_is_fitted, validate_data

logger = logging.getLogger(__name__)


class ArPls(TransformerMixin, OneToOneFeatureMixin, BaseEstimator):
    """
    This class implements the Assymmetrically Reweighted Penalized Least Squares (ArPls) is a baseline
    correction method for spectroscopy data. It uses an iterative process
    to estimate and remove the baseline from the spectra.

    Parameters
    ----------
    lam : float, optional (default=1e4)
        The penalty parameter for the difference matrix in the objective function.

    ratio : float, optional (default=0.01)
        The convergence threshold for the weight updating scheme.

    nr_iterations : int, optional (default=100)
        The maximum number of iterations for the weight updating scheme.


    Methods
    -------
    fit(X, y=None)
        Fit the estimator to the data.

    transform(X, y=None)
        Transform the data by removing the baseline.

    _calculate_diff(N)
        Calculate the difference matrix for a given size.

    _calculate_ar_pls(x)
        Calculate the baseline for a given spectrum.

    References
    ----------
    - Sung-June Baek, Aaron Park, Young-Jin Ahn, Jaebum Choo
    Baseline correction using asymmetrically reweighted penalized
    least squares smoothing
    """

    def __init__(
        self,
        lam: float = 1e4,
        ratio: float = 0.01,
        nr_iterations: int = 100,
    ):
        self.lam = lam
        self.ratio = ratio
        self.nr_iterations = nr_iterations

    def fit(self, X: np.ndarray, y=None) -> "ArPls":
        """Fit the estimator to the data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data.

        y : array-like of shape (n_samples,), optional (default=None)
            The target values.

        Returns
        -------
        self : ArPls
            Returns the instance itself.
        """

        # Check that X is a 2D array and has only finite values
        X = validate_data(
            self, X, y="no_validation", ensure_2d=True, reset=True, dtype=np.float64
        )

        return self

    def transform(self, X: np.ndarray, y=None) -> np.ndarray:
        """Transform the data by removing the baseline.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data.

        y : array-like of shape (n_samples,), optional (default=None)
            The target values.

        Returns
        -------
        X_ : array-like of shape (n_samples, n_features)
            The transformed data with the baseline removed.
        """

        # Check that the estimator is fitted
        check_is_fitted(self, "n_features_in_")

        # Check that X is a 2D array and has only finite values
        X_ = validate_data(
            self,
            X,
            y="no_validation",
            ensure_2d=True,
            copy=True,
            reset=False,
        )

        # Calculate the ar pls baseline
        for i, x in enumerate(X_):
            X_[i] = x - self._calculate_ar_pls(x)

        return X_.reshape(-1, 1) if X_.ndim == 1 else X_

    def _calculate_diff(self, N):
        identity_matrix = sp.eye(N, format="csc")
        D2 = sp.diags([1, -2, 1], [0, 1, 2], shape=(N - 2, N), format="csc")
        return D2.dot(identity_matrix).T

    def _calculate_ar_pls(self, x):
        N = len(x)
        D = self._calculate_diff(N)
        H = self.lam * D.dot(D.T)
        w = np.ones(N)
        iteration = 0
        while iteration < self.nr_iterations:
            W = spdiags(w, 0, N, N)
            C = csc_matrix(W + H)
            z = splu(C).solve(w * x)
            d = x - z
            dn = d[d < 0]
            if len(dn) == 0:
                break
            m = np.mean(dn)
            s = np.std(dn)
            exponent = np.clip(2 * (d - (2 * s - m)) / s, -709, 709)
            wt = 1.0 / (1.0 + np.exp(exponent))
            if np.linalg.norm(w - wt) / np.linalg.norm(w) < self.ratio:
                break
            w = wt
            iteration += 1
        return z
