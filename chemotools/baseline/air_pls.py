import logging
import numpy as np
from scipy.sparse import csc_matrix, eye, diags
from scipy.sparse.linalg import spsolve
from sklearn.base import BaseEstimator, TransformerMixin, OneToOneFeatureMixin
from sklearn.utils.validation import check_is_fitted

from chemotools.utils.check_inputs import check_input

logger = logging.getLogger(__name__)


class AirPls(OneToOneFeatureMixin, BaseEstimator, TransformerMixin):
    """
    This class implements the AirPLS (Adaptive Iteratively Reweighted Penalized Least Squares) algorithm for baseline
    correction of spectra data. AirPLS is a common approach for removing the baseline from spectra, which can be useful
    in various applications such as spectroscopy and chromatography.

    Parameters
    ----------
    lam : float, optional default=1e2
        The lambda parameter controls the smoothness of the baseline. Increasing the value of lambda results in
        a smoother baseline.

    polynomial_order : int, optional default=1
        The polynomial order determines the degree of the polynomial used to fit the baseline. A value of 1 corresponds
        to a linear fit, while higher values correspond to higher-order polynomials.

    nr_iterations : int, optional default=15
        The number of iterations used to calculate the baseline. Increasing the number of iterations can improve the
        accuracy of the baseline correction, but also increases the computation time.

    Attributes
    ----------
    n_features_in_ : int
        The number of features in the input data.

    _is_fitted : bool
        A flag indicating whether the estimator has been fitted to data.

    Methods
    -------
    fit(X, y=None)
        Fit the estimator to the input data.

    transform(X, y=None)
        Transform the input data by subtracting the baseline.

    _calculate_whittaker_smooth(x, w)
        Calculate the Whittaker smooth of a given input vector x, with weights w.
        
    _calculate_air_pls(x)
        Calculate the AirPLS baseline of a given input vector x.

    References
    ----------
    - Z.-M. Zhang, S. Chen, and Y.-Z. Liang, Baseline correction using adaptive iteratively reweighted penalized least
      squares. Analyst 135 (5), 1138-1146 (2010).
    """

    def __init__(
        self,
        lam: int = 100,
        polynomial_order: int = 1,
        nr_iterations: int = 15,
    ):
        self.lam = lam
        self.polynomial_order = polynomial_order
        self.nr_iterations = nr_iterations

    def fit(self, X: np.ndarray, y=None) -> "AirPls":
        """Fit the AirPls baseline correction estimator to the input data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data.

        y : array-like of shape (n_samples,), optional (default=None)
            The target values.

        Returns
        -------
        self : AirPls
            Returns the instance itself.
        """
        # Check that X is a 2D array and has only finite values
        X = check_input(X)

        # Set the number of features
        self.n_features_in_ = X.shape[1]

        # Set the fitted attribute to True
        self._is_fitted = True

        return self

    def transform(self, X: np.ndarray, y=None) -> np.ndarray:
        """Correct the baseline in the input data using the fitted AirPls estimator.

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
        check_is_fitted(self, "_is_fitted")

        # Check that X is a 2D array and has only finite values
        X = check_input(X)
        X_ = X.copy()

        # Check that the number of features is the same as the fitted data
        if X_.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features but got {X_.shape[1]}"
            )

        # Calculate the air pls smooth
        for i, x in enumerate(X_):
            X_[i] = x - self._calculate_air_pls(x)

        return X_.reshape(-1, 1) if X_.ndim == 1 else X_

    def _calculate_whittaker_smooth(self, x, w):
        X = np.matrix(x)
        m = X.size
        E = eye(m, format="csc")
        for i in range(self.polynomial_order):
            E = E[1:] - E[:-1]
        W = diags(w, 0, shape=(m, m))
        A = csc_matrix(W + (self.lam * E.T * E))
        B = csc_matrix(W * X.T)
        background = spsolve(A, B)
        return np.array(background)

    def _calculate_air_pls(self, x):
        m = x.shape[0]
        w = np.ones(m)

        for i in range(1, self.nr_iterations):
            z = self._calculate_whittaker_smooth(x, w)
            d = x - z
            dssn = np.abs(d[d < 0].sum())

            if dssn < 0.001 * np.abs(x).sum():
                break

            if i == self.nr_iterations - 1:
                break

            w[d >= 0] = 0
            w[d < 0] = np.exp(i * np.abs(d[d < 0]) / dssn)

            negative_d = d[d < 0]
            if negative_d.size > 0:
                w[0] = np.exp(i * negative_d.max() / dssn)

            w[-1] = w[0]

        return z
