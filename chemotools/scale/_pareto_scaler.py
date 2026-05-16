"""
The :mod:`chemotools.scale._pareto_scaler` module implements a Pareto Scaler
transformer.
"""

# Authors: Pau Cabaneros
# License: MIT

import warnings
from numbers import Real

import numpy as np
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin
from sklearn.utils._param_validation import Interval
from sklearn.utils.validation import check_is_fitted, validate_data

from chemotools._doc_mixin import DocLinkMixin


class ParetoScaler(DocLinkMixin, TransformerMixin, OneToOneFeatureMixin, BaseEstimator):
    """
    This transformer scales data using a generalized power of the standard deviation,
    as described by [1]_. It acts as a bridge between Mean Centering (P=0) and
    Autoscaling (P=1).

    Parameters
    ----------
    p : float, default=0.5
        The exponent to use in the scaling. Must be a non-negative float between 0
        and 1.
        - p=0.0: No scaling (Mean Centering only).
        - p=0.5: Standard Pareto Scaling.
        - p=1.0: Autoscaling (Unit Variance scaling).

    with_mean : bool, default=True
        If True, center the data before scaling. If False, no centering is performed.

    copy : bool, default=True
        If True, a copy of the input data will be made. If False, the input data will
        be modified in place.

    Attributes
    ----------
    mean_ : np.ndarray of shape (n_features,)
        The mean value for each feature, calculated during fitting.

    scale_ : np.ndarray of shape (n_features,)
        The scale factor for each feature, calculated as the standard deviation raised
        to the power of p.

    n_features_in_ : int
        The number of features in the input data.

    References
    ----------
    .. [1] Kurt Varmuza and Peter Filzmoser (2024),
        Adjusted Pareto Scaling for Multivariate Calibration Models,
        Journal of Chemometrics,
        Volume 38, Issue 11,
        https://doi.org/10.1002/cem.3588.

    Examples
    --------
    >>> from chemotools.datasets import load_fermentation_train
    >>> from chemotools.scale import ParetoScaler
    >>> # Load sample data
    >>> X, _ = load_fermentation_train()
    >>> scaler = ParetoScaler(p=0.3)
    ParetoScaler(p=0.3)
    >>> # Fit and transform the data
    >>> X_scaled = scaler.fit_transform(X)

    Notes
    -----
    In spectroscopic applications, standard Pareto scaling (:math:`P=0.5`) is often used
    to reduce the dominance of large peaks (e.g., solvent or high-abundance
    metabolites) without inflating baseline noise as severely as autoscaling.

    According to Varmuza & Filzmoser, :math:`P` should be treated as a tunable
    hyperparameter. For datasets where relevant information is buried in
    low-intensity signals but the noise floor is high, an "Adjusted" :math:`P` (e.g.,
    0.3 or 0.7) may provide a superior balance of signal-to-noise ratio and
    model interpretability compared to fixed Pareto scaling.

    See also
    --------
    sklearn.preprocessing.StandardScaler : Standardize features by removing the mean
    and scaling to unit variance.
    """

    _parameter_constraints: dict = {
        "p": [Interval(Real, 0, 1, closed="both")],
        "with_mean": ["boolean"],
        "copy": ["boolean"],
    }

    def __init__(
        self,
        p: float = 0.5,
        with_mean: bool = True,
        copy: bool = True,
    ):
        self.p = p
        self.with_mean = with_mean
        self.copy = copy

    def fit(self, X: np.ndarray, y=None) -> "ParetoScaler":
        """
        Fit the transformer to the input data.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            The input data to fit the transformer to.

        y : None
            Ignored to align with API.

        Returns
        -------
        self : ParetoScaler
            The fitted transformer.
        """
        # Validate input parameters
        self._validate_params()

        # Validate input data
        X = validate_data(
            self, X, y="no_validation", ensure_2d=True, reset=True, dtype=np.float64
        )

        # 1. Calculate Mean
        if self.with_mean:
            self.mean_ = np.mean(X, axis=0)
        else:
            self.mean_ = np.zeros(X.shape[1])

        # 2. Calculate Standard Deviation (The base for Pareto)
        # Using ddof=0 for population standard deviation (ddof set to 0 to align with
        # StandardScaler)
        std = np.std(X, axis=0, ddof=0)

        # 3. Calculate Scale: std^p
        # Note: If p=0.5, it's sqrt(std), which is Pareto.
        self.scale_ = np.power(std, self.p)

        # 4. Handle Zero Variance / Constant Features
        # Catch cases where scale_ is 0 (which happens if std is 0 and p > 0)
        zero_scale_mask = np.isclose(self.scale_, 0)

        if np.any(zero_scale_mask):
            num_zeros = np.sum(zero_scale_mask)
            warnings.warn(
                f"The scale for {num_zeros} feature(s) is zero (constant columns). "
                "To avoid division by zero, these features will not be scaled.",
                UserWarning,
            )
            # Set scale to 1.0 so that (X - mean) / 1.0 = (X - mean)
            self.scale_[zero_scale_mask] = 1.0

        return self

    def transform(self, X: np.ndarray, y=None) -> np.ndarray:
        """
        Transform the input data.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            The input data to transform.

        y : None
            Ignored to align with API.

        Returns
        -------
        X_transformed : np.ndarray of shape (n_samples, n_features)
            The transformed data.
        """
        # Check that the estimator is fitted
        check_is_fitted(self, "scale_")

        # Check that X is a 2D array and has only finite values
        X_ = validate_data(
            self,
            X,
            y="no_validation",
            ensure_2d=True,
            copy=self.copy,
            reset=False,
            dtype=np.float64,
        )

        # Return the scaled data
        return (X_ - self.mean_) / self.scale_

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Inverse transform the data back to the original space.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            The data to inverse transform.

        Returns
        -------
        X_original : np.ndarray of shape (n_samples, n_features)
            The data transformed back to the original space.
        """
        # Check that the estimator is fitted
        check_is_fitted(self, "scale_")

        # Check that X is a 2D array and has only finite values
        X_ = validate_data(
            self,
            X,
            y="no_validation",
            ensure_2d=True,
            copy=self.copy,
            reset=False,
            dtype=np.float64,
        )

        return X_ * self.scale_ + self.mean_
