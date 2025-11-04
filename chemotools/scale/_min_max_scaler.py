"""
The :mod:`chemotools.scale._min_max_scaler` module implements a Min-Max Scaler transformer.
"""

# Authors: Pau Cabaneros
# License: MIT

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, OneToOneFeatureMixin
from sklearn.utils.validation import check_is_fitted, validate_data


class MinMaxScaler(TransformerMixin, OneToOneFeatureMixin, BaseEstimator):
    """
    A transformer that scales the input data by subtracting the minimum and dividing by
    the difference between the maximum and the minimum. When the use_min parameter is False,
    the data is scaled by the maximum.

    Parameters
    ----------
    use_min : bool, default=True
        The normalization to use. If True, the data is subtracted by the minimum and
        scaled by the maximum. If False, the data is scaled by the maximum.

    Attributes
    ----------
    n_features_in_ : int
        The number of features in the input data.

    Examples
    --------
    >>> from chemotools.datasets import load_fermentation_train
    >>> from chemotools.scale import MinMaxScaler
    >>> # Load sample data
    >>> X, _ = load_fermentation_train()
    >>> # Initialize MinMaxScaler
    >>> scaler = MinMaxScaler()
    MinMaxScaler()
    >>> # Fit and transform the data
    >>> X_scaled = scaler.fit_transform(X)
    """

    _parameter_constraints: dict = {
        "use_min": ["boolean"],
    }

    def __init__(self, use_min: bool = True):
        self.use_min = use_min

    def fit(self, X: np.ndarray, y=None) -> "MinMaxScaler":
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
        self : MinMaxScaler
            The fitted transformer.
        """
        # Check that X is a 2D array and has only finite values
        X = validate_data(
            self, X, y="no_validation", ensure_2d=True, reset=True, dtype=np.float64
        )
        return self

    def transform(self, X: np.ndarray, y=None) -> np.ndarray:
        """
        Transform the input data by scaling it.

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
        if self.use_min:
            X_ = (X_ - np.min(X_, axis=1, keepdims=True)) / (
                np.max(X_, axis=1, keepdims=True) - np.min(X_, axis=1, keepdims=True)
            )

        else:
            X_ = X_ / np.max(X_, axis=1, keepdims=True)

        return X_.reshape(-1, 1) if X_.ndim == 1 else X_
