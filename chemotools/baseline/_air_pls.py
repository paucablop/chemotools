"""
This module contains the ``AirPLS`` transformer, which performs baseline correction on
data according to the Whittaker-Henderson formulation of Penalized Least Squares which
was modified by the introduction of weights that are updated iteratively to improve the
baseline identification.

References
----------
It's based on the algorithms described in [1]_ and [2]_ where an implementational
adaption of [2]_ was required to make it numerically stable ([3]_).

.. [1] Z.-M. Zhang, S. Chen, and Y.-Z. Liang, "Baseline correction using adaptive
   iteratively reweighted penalized least squares", Analyst 135 (5), 1138-1146 (2010)
.. [2] G. Biessy, "Revisiting Whittaker-Henderson smoothing", arXiv:2306.06932 (2023)
.. [3] https://math.stackexchange.com/q/4819039/1261538

"""

import logging
from typing import Union

import numpy as np
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from chemotools.utils.check_inputs import check_input
from chemotools.utils.whittaker_base import WhittakerLikeSolver

logger = logging.getLogger(__name__)


# TODO: is polynomial_order actually differences and if so, is the description correct?
class AirPls(
    OneToOneFeatureMixin, BaseEstimator, TransformerMixin, WhittakerLikeSolver
):
    """
    This class implements the Adaptive Iteratively Reweighted Penalized Least Squares
    a.k.a AirPLS algorithm for baseline correction of spectra data. AirPLS is a common
    approach for removing the baseline from spectra, which can be useful in various
    applications such as spectroscopy and chromatography.

    Parameters
    ----------
    lam : float or int, optional default=1e2
        The lambda parameter that controls the smoothness of the baseline. Higher values
        will result in a smoother baseline.

    polynomial_order : int, optional default=1
        The degree of the polynomial used to fit the baseline. A value of 1 corresponds
        to a linear fit, while higher values correspond to higher-order polynomials.

    nr_iterations : int, optional default=15
        The number of iterations used to calculate the baseline. Increasing the number
        of iterations can improve the accuracy of the baseline correction at the cost of
        computation time.

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
    It's based on the algorithms described in [1]_ and [2]_ where an implementational
    adaption of [2]_ was required to make it numerically stable ([3]_).

    .. [1] Z.-M. Zhang, S. Chen, and Y.-Z. Liang, "Baseline correction using adaptive
       iteratively reweighted penalized least squares", Analyst 135 (5), 1138-1146
       (2010)
    .. [2] G. Biessy, "Revisiting Whittaker-Henderson smoothing", arXiv:2306.06932
       (2023)
    .. [3] https://math.stackexchange.com/q/4819039/1261538

    """

    # TODO: polynomial order is actually differences
    def __init__(
        self,
        lam: Union[float, int] = 100,
        polynomial_order: int = 1,
        nr_iterations: int = 15,
    ):
        self.lam: Union[float, int] = lam
        self.polynomial_order: int = polynomial_order
        self.nr_iterations: int = nr_iterations

    def fit(self, X: np.ndarray, y=None) -> "AirPls":
        """Fit the AirPls baseline correction estimator to the input data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data. It is internally promoted to ``np.float64`` to avoid loss of
            precision.

        y : array-like of shape (n_samples,), optional (default=None)
            The target values.

        Returns
        -------
        self : AirPls
            Returns the instance itself.

        """
        # Check that X is a 2D array and has only finite values
        X = BaseEstimator._validate_data(  # type: ignore
            self,
            X,
            reset=True,
            ensure_2d=True,
            force_all_finite=True,
            dtype=WhittakerLikeSolver._WhittakerLikeSolver__dtype,  # type: ignore
        )

        # the internal solver is set up
        self._setup_for_fit(
            n_data=X.shape[1],
            differences=self.polynomial_order,
            lam=self.lam,
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
        X = check_input(
            X,
            dtype=WhittakerLikeSolver._WhittakerLikeSolver__dtype,  # type: ignore
        )
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

        return X_

    def _calculate_air_pls(self, x):
        # FIXME: this initial weighting strategy might not yield the best results
        w = np.ones_like(x)
        # FIXME: this initialisation will will fail for many signals and produce a
        #        zero-baseline
        z = np.zeros_like(x)
        dssn_thresh = max(1e-3 * np.abs(x).sum(), 1e-308)  # to avoid 0 equalities

        # FIXME: work on full Arrays and use internal loop of ``whittaker_solve``
        for i in range(0, self.nr_iterations - 1):
            # the baseline is fitted using the Whittaker smoother framework
            z, _ = self._solve_single_b_fixed_lam(b=x, w=w)
            d = x - z
            dssn = np.abs(d[d < 0].sum())

            # the algorithm is stopped if the threshold is reached
            if dssn <= dssn_thresh:
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
