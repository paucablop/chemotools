"""
The :mod:`chemotools.baseline._non_negative` module implements
a non-negative transformer.
"""

# Author: Pau Cabaneros
# License: MIT

from typing import Literal
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, OneToOneFeatureMixin
from sklearn.utils.validation import check_is_fitted, validate_data


class NonNegative(TransformerMixin, OneToOneFeatureMixin, BaseEstimator):
    """
    A transformer that sets all negative values to zero or to abs.

    Parameters
    ----------
    mode : Literal["zero", "abs"], optional, default="zero"
        The mode to use for the non-negative values. Can be:
        - *zero*: set all negative values to zero.
        - *abs*: set all negative values to their absolute value.

    Attributes
    ----------
    n_features_in_ : int
        The number of features in the input data.

    Examples
    --------
    >>> from chemotools.baseline import NonNegative
    >>> from chemotools.datasets import load_fermentation_train
    >>> # Load sample data
    >>> X, _ = load_fermentation_train()
    >>> # Instantiate the transformer
    >>> transformer = NonNegative(mode="zero")
    NonNegative(mode="zero")
    >>> transformer.fit(X)
    >>> # Generate non-negative data
    >>> X_non_negative = transformer.transform(X)
    """

    def __init__(self, mode: Literal["zero", "abs"] = "zero"):
        self.mode = mode

    def fit(self, X: np.ndarray, y=None) -> "NonNegative":
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
        self : NonNegative
            The fitted transformer.
        """
        # Check that X is a 2D array and has only finite values
        X = validate_data(
            self, X, y="no_validation", ensure_2d=True, reset=True, dtype=np.float64
        )
        return self

    def transform(self, X: np.ndarray, y=None) -> np.ndarray:
        """
        Transform the input data to non-negative values.

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

        # Calculate non-negative values
        for i, x in enumerate(X_):
            if self.mode == "zero":
                X_[i] = np.clip(x, a_min=0, a_max=np.inf)

            if self.mode == "abs":
                X_[i] = np.abs(x)

        return X_.reshape(-1, 1) if X_.ndim == 1 else X_
