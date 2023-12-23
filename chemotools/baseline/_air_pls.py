import logging

import numpy as np
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from chemotools.utils.check_inputs import check_input
from chemotools.utils.whittaker_base import WhittakerLikeSolver

logger = logging.getLogger(__name__)


class AirPls(
    OneToOneFeatureMixin, BaseEstimator, TransformerMixin, WhittakerLikeSolver
):
    """
    This class implements the AirPLS (Adaptive Iteratively Reweighted Penalized Least Squares) algorithm for baseline
    correction of spectra data. AirPLS is a common approach for removing the baseline from spectra, which can be useful
    in various applications such as spectroscopy and chromatography.

    Parameters
    ----------
    lam : float or int, optional default=1e2
        The lambda parameter controls the smoothness of the baseline. Increasing the value of lambda results in
        a smoother baseline.

    polynomial_order : int, optional default=1
        The polynomial order determines the degree of the polynomial used to fit the baseline. A value of 1 corresponds
        to a linear fit, while higher values correspond to higher-order polynomials.

    nr_iterations : int, optional default=15
        The number of iterations used to calculate the baseline. Increasing the number of iterations can improve the
        accuracy of the baseline correction, but also increases the computation time.

    rcond : float, default=1e-15
        The relative condition number which is used to keep all matrices involved
        positive definite. This is not actively used at the moment.
        It works in the same way as the ``rcond`` parameter of SciPy's ``linalg.pinvh``.

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

    # TODO: polynomial order is actually differences
    def __init__(
        self,
        lam: int | float = 100,
        polynomial_order: int = 1,
        nr_iterations: int = 15,
        rcond: float = 1e-15,
    ):
        self.lam: int | float = lam
        self.polynomial_order: int = polynomial_order
        self.nr_iterations: int = nr_iterations
        self.rcond: float = rcond

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
        X = BaseEstimator._validate_data(self, X, reset=True)  # type: ignore

        # the internal solver is set up
        self._setup_for_fit(
            series_size=X.shape[1],
            lam=self.lam,
            differences=self.polynomial_order,
            rcond=self.rcond,
        )

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
        check_is_fitted(self, "n_features_in_")

        # Check that X is a 2D array and has only finite values
        X = check_input(X)
        X_ = X.copy()

        # Check that the number of features is the same as the fitted data
        # NOTE: ``n_features_in_`` is set in ``BaseEstimator._validate_data`` when
        #       ``reset`` is True
        if X_.shape[1] != self.n_features_in_:  # type: ignore
            raise ValueError(
                f"Expected {self.n_features_in_} features but got {X_.shape[1]}"  # type: ignore # noqa: E501
            )

        # Calculate the air pls smooth
        for i, x in enumerate(X_):
            X_[i] = x - self._calculate_air_pls(x)

        # FIXME: can this even happen because X is ensured to be 2D?
        if X_.ndim == 1:
            # FIXME: shouldn't this be a row and not a column vector because
            #        Scikit-Learn works with shape (n_samples, n_features), i.e.,
            #        (1, n_features) for a single sample?
            return X_.reshape((-1, 1))
        else:
            return X_

    def _calculate_air_pls(self, x):
        # FIXME: this initial weighting strategy might not yield the best results
        w = np.ones_like(x)
        z = np.zeros_like(x)
        dssn_thresh = 1e-3 * np.abs(x).sum()

        # FIXME: work on full Arrays and use internal loop of ``whittaker_solve``
        for i in range(0, self.nr_iterations - 1):
            # the baseline is fitted using the Whittaker smoother framework
            z = self._whittaker_solve(X=x, w=w, use_same_w_for_all=True)[0]
            d = x - z
            dssn = np.abs(d[d < 0].sum())

            # the algorithm is stopped if the threshold is reached
            if dssn < dssn_thresh:
                break

            # the weights are updated
            below_base_indics = d < 0
            w[~below_base_indics] = 0.0
            exp_mult = i + 1
            w[below_base_indics] = np.exp(exp_mult * np.abs(d[d < 0]) / dssn)

            d_negative = d[below_base_indics]
            if d_negative.size > 0:
                # FIXME: this might easily yield a weight of 1 if the maximum of the
                #        negative_d is very close to zero
                w[0] = np.exp(exp_mult * d_negative.max() / dssn)

            w[-1] = w[0]

        return z
