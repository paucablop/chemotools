"""
The :mod:`chemotools.scale._norm_scaler` module implements a L-norm Scaler transformer.
"""

# Authors: Pau Cabaneros
# License: MIT

from numbers import Integral
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, OneToOneFeatureMixin
from sklearn.utils.validation import check_is_fitted, validate_data
from sklearn.utils._param_validation import Interval


class NormScaler(TransformerMixin, OneToOneFeatureMixin, BaseEstimator):
    """
    A transformer that scales the input data by the L-norm of the spectrum.

    Parameters
    ----------
    l_norm : int, optional, default=2
        The L-norm to use. Default is 2.

    Attributes
    ----------
    n_features_in_ : int
        The number of features in the input data.

    Examples
    --------
    >>> from chemotools.datasets import load_fermentation_train
    >>> from chemotools.scale import NormScaler
    >>> # Load sample data
    >>> X, _ = load_fermentation_train()
    >>> # Initialize NormScaler
    >>> scaler = NormScaler(l_norm=2)
    NormScaler()
    >>> # Fit and transform the data
    >>> X_scaled = scaler.fit_transform(X)
    """

    _parameter_constraints: dict = {
        "l_norm": [Interval(Integral, 1, None, closed="left")],
    }

    def __init__(self, l_norm: int = 2):
        self.l_norm = l_norm

    def fit(self, X: np.ndarray, y=None) -> "NormScaler":
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
        self : NormScaler
            The fitted transformer.
        """
        # Check that X is a 2D array and has only finite values
        X = validate_data(
            self, X, y="no_validation", ensure_2d=True, reset=True, dtype=np.float64
        )
        return self

    def transform(self, X: np.ndarray, y=None) -> np.ndarray:
        """
        Transform the input data by scaling by the L-norm.

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
        check_is_fitted(self, "n_features_in_")

        # Check that X is a 2D array and has only finite values
        X_ = validate_data(
            self,
            X,
            y="no_validation",
            ensure_2d=True,
            copy=True,
            reset=False,
            dtype=np.float64,
        )

        # Normalize the data by the maximum value
        for i, x in enumerate(X_):
            X_[i] = x / np.linalg.norm(x, ord=self.l_norm)

        return X_.reshape(-1, 1) if X_.ndim == 1 else X_
