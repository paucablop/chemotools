"""
This module contains the ``ArPLS`` transformer, which performs baseline correction on
data according to the Whittaker-Henderson formulation of Penalized Least Squares which
was modified by the introduction of weights that are updated iteratively to improve the
baseline identification. It simultaneously estimates the baseline as well as the
baseline noise.

References
----------
It's based on the algorithms described in [1]_ and [2]_ where an implementational
adaption of [2]_ was required to make it numerically stable ([3]_).

.. [1] S.-J. Baek, A. Park, Y.-J. Ahn, J. Choo, "Baseline correction using
   asymmetrically reweighted penalized least squares smoothing", Analyst, 140, 250–257
   (2015)
.. [2] G. Biessy, "Revisiting Whittaker-Henderson smoothing", arXiv:2306.06932 (2023)
.. [3] https://math.stackexchange.com/q/4819039/1261538

"""

import logging
from numbers import Integral

import numpy as np
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin
from sklearn.utils.validation import check_is_fitted, check_scalar

from chemotools.utils.check_inputs import check_input
from chemotools.utils.whittaker_base import WhittakerLikeSolver

logger = logging.getLogger(__name__)


class ArPls(OneToOneFeatureMixin, BaseEstimator, TransformerMixin, WhittakerLikeSolver):
    """
    This class implements the Asymmetrically Reweighted Penalized Least Squares a.k.a
    ArPLS which is a baseline correction method for spectroscopy data. It uses an
    iterative process that simultaneously estimates the baseline as well as the baseline
    noise.

    Parameters
    ----------
    lam : float or int, default=1e4
        The lambda parameter that controls the smoothness of the baseline. Higher values
        will result in a smoother baseline.

    ratio : float, default=0.01
        The convergence threshold for the weight updating scheme. Lower values will
        result in a more accurate baseline at the cost of computation time and even
        convergence.

    nr_iterations : int, optional (default=100)
        The maximum number of iterations for the weight updating scheme.

    rcond : float, default=1e-15
        The relative condition number which is used to keep all matrices involved
        positive definite. This is not actively used at the moment.
        It works in the same way as the ``rcond`` parameter of SciPy's ``linalg.pinvh``.

    Methods
    -------
    fit(X, y=None)
        Fit the estimator to the data.

    transform(X, y=None)
        Transform the data by removing the baseline.

    _calculate_ar_pls(x)
        Calculate the baseline for a given spectrum.

    References
    ----------
    .. [1] S.-J. Baek, A. Park, Y.-J. Ahn, J. Choo, "Baseline correction using
       asymmetrically reweighted penalized least squares smoothing", Analyst, 140,
       250–257 (2015)
    .. [2] G. Biessy, "Revisiting Whittaker-Henderson smoothing", arXiv:2306.06932
       (2023)
    .. [3] https://math.stackexchange.com/q/4819039/1261538

    """

    def __init__(
        self,
        lam: float | int = 1e4,
        differences: int = 2,
        ratio: float = 0.01,
        nr_iterations: int = 100,
        rcond: float = 1e-15,
    ):
        self.lam: float | int = lam
        self.differences: int = differences
        self.ratio: float = ratio
        self.nr_iterations: int = nr_iterations
        self.rcond: float = rcond

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

        # the constructor parameters are checked
        check_scalar(
            x=self.ratio,
            name="ratio",
            target_type=float,
            min_val=1e-15,
        )
        check_scalar(
            x=self.nr_iterations, name="nr_iterations", target_type=Integral, min_val=1
        )

        # Check that X is a 2D array and has only finite values
        X = BaseEstimator._validate_data(self, X, reset=True)  # type: ignore

        # the internal solver is setup
        self._setup_for_fit(
            series_size=X.shape[1],
            lam=self.lam,
            differences=self.differences,
            rcond=self.rcond,
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
        X = check_input(X)
        X_ = X.copy()

        # Check that the number of features is the same as the fitted data
        # NOTE: ``n_features_in_`` is set in ``BaseEstimator._validate_data`` when
        #       ``reset`` is True
        if X_.shape[1] != self.n_features_in_:  # type: ignore
            raise ValueError(
                f"Expected {self.n_features_in_} features but got {X_.shape[1]}"  # type: ignore # noqa: E501
            )

        # Calculate the ar pls baseline
        for i, x in enumerate(X_):
            X_[i] = x - self._calculate_ar_pls(x)

        # FIXME: can this even happen because X is ensured to be 2D?
        if X_.ndim == 1:
            # FIXME: shouldn't this be a row and not a column vector because
            #        Scikit-Learn works with shape (n_samples, n_features), i.e.,
            #        (1, n_features) for a single sample?
            return X_.reshape((-1, 1))
        else:
            return X_

    def _calculate_ar_pls(self, x):
        # FIXME: this initial weighting strategy might not yield the best results
        w = np.ones_like(x)
        z = np.zeros_like(x)
        # FIXME: work on full Arrays and use internal loop of ``whittaker_solve``
        for _ in range(self.nr_iterations):
            # the baseline is fitted using the Whittaker smoother framework
            z = self._whittaker_solve(X=x, w=w, use_same_w_for_all=True)[0]
            d = x - z

            # if there is no data point below the baseline, the baseline is considered
            # to be fitted
            d_negative = d[d < 0]
            if len(d_negative) == 0:
                break
            m = np.mean(d_negative)
            s = np.std(d_negative)
            exponent = np.clip(2.0 * (d - (2.0 * s - m)) / s, -709, 709)  # type: ignore
            wt = 1.0 / (1.0 + np.exp(exponent))
            if np.linalg.norm(w - wt) / np.linalg.norm(w) < self.ratio:  # type: ignore
                break
            w = wt
        return z
