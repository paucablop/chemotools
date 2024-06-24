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

# Authors:
# Pau Cabaneros
# Niklas Zell <nik.zoe@web.de>


### Imports ###

import logging
from numbers import Integral
from typing import Union

import numpy as np
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin
from sklearn.utils.validation import check_is_fitted, check_scalar

from chemotools.utils._whittaker_base import WhittakerLikeSolver
from chemotools.utils.check_inputs import check_input

logger = logging.getLogger(__name__)

### Main Class ###


class ArPls(
    OneToOneFeatureMixin,
    BaseEstimator,
    TransformerMixin,
    WhittakerLikeSolver,
):
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

    differences : int, default=2
        The order of the differences used for the penalty terms that enforces smoothness
        of the baseline.
        Higher values will result in a smoother baseline.
        Currently, values ``>= 3`` are highly discouraged due to numerical instability
        that might obscure the smoothing effect.

    ratio : float, default=0.01
        The convergence threshold for the weight updating scheme. Lower values will
        result in a more accurate baseline at the cost of computation time and even
        convergence.

    nr_iterations : int, optional (default=100)
        The maximum number of iterations for the weight updating scheme.

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
        lam: Union[float, int] = 1e4,
        differences: int = 2,
        ratio: float = 0.01,
        nr_iterations: int = 100,
    ):
        self.lam: Union[float, int] = lam
        self.differences: int = differences
        self.ratio: float = ratio
        self.nr_iterations: int = nr_iterations

    def fit(self, X: np.ndarray, y=None) -> "ArPls":
        """Fit the estimator to the data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data. It is internally promoted to ``np.float64`` to avoid loss of
            precision.

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
        X = BaseEstimator._validate_data(  # type: ignore
            self,
            X,
            reset=True,
            ensure_2d=True,
            force_all_finite=True,
            dtype=WhittakerLikeSolver._WhittakerLikeSolver__dtype,  # type: ignore
        )

        # the internal solver is setup
        self._setup_for_fit(
            num_data=X.shape[1],
            differences=self.differences,
            lam=self.lam,
            child_class_name=self.__class__.__name__,
        )

        return self

    def transform(self, X: np.ndarray, y=None) -> np.ndarray:
        """Transform the data by removing the baseline.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data. It is internally promoted to ``np.float64`` to avoid loss of
            precision.

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

        # Calculate the ar pls baseline
        for i, x in enumerate(X_):
            X_[i] = x - self._calculate_ar_pls(x)

        return X_

    def _calculate_ar_pls(self, x):
        # FIXME: this initial weighting strategy might not yield the best results
        w = np.ones_like(x)
        # FIXME: this initialisation will will fail for many signals and produce a
        #        zero-baseline
        z = np.zeros_like(x)
        # FIXME: work on full Arrays and use internal loop of ``whittaker_solve``
        for _ in range(self.nr_iterations):
            # the baseline is fitted using the Whittaker smoother framework
            z, _ = self._solve_single_b_fixed_lam(rhs_b=x, weights=w)
            d = x - z

            # if there is no data point below the baseline, the baseline is considered
            # to be fitted
            d_negative = d[np.where(d < 0)[0]]
            if len(d_negative) == 0:
                break
            m = d_negative.mean()
            s = d_negative.std()
            exponent = np.clip(2.0 * (d - (2.0 * s - m)) / s, -709, 709)  # type: ignore
            wt = 1.0 / (1.0 + np.exp(exponent))
            if np.linalg.norm(w - wt) / np.linalg.norm(w) < self.ratio:  # type: ignore
                break
            w = wt

        return z
