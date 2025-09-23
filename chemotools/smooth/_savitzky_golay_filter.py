"""
The :mod:`chemotools.smooth._median_filter` module implements the Median Filter (MD) transformation.
"""

# Authors: Pau Cabaneros
# License: MIT

from typing import Literal

import numpy as np
from scipy.signal import savgol_filter
from sklearn.base import BaseEstimator, TransformerMixin, OneToOneFeatureMixin
from sklearn.utils.validation import check_is_fitted, validate_data


class SavitzkyGolayFilter(TransformerMixin, OneToOneFeatureMixin, BaseEstimator):
    """
    A transformer that calculates the Savitzky-Golay filter of the input data.

    Parameters
    ----------
    window_size : int, optional
        The size of the window to use for the Savitzky-Golay filter. Must be odd. Default
        is 3.

    polynomial_order : int, optional
        The order of the polynomial to use for the Savitzky-Golay filter. Must be less
        than window_size. Default is 1.

    mode : str, optional
        The mode to use for the Savitzky-Golay filter. Can be "nearest", "constant",
        "reflect", "wrap", "mirror" or "interp". Default is "nearest".

    Attributes
    ----------
    n_features_in_ : int
        The number of features in the training data.

    Examples
    --------
    >>> from chemotools.datasets import load_fermentation_train
    >>> from chemotools.smooth import SavitzkyGolayFilter
    >>> # Load sample data
    >>> X, _ = load_fermentation_train()
    >>> # Initialize SavitzkyGolayFilter
    >>> sgf = SavitzkyGolayFilter()
    SavitzkyGolayFilter()
    >>> # Fit and transform the data
    >>> X_smoothed = sgf.fit_transform(X)

    """

    def __init__(
        self,
        window_size: int = 3,
        polynomial_order: int = 1,
        mode: Literal["mirror", "constant", "nearest", "wrap", "interp"] = "nearest",
    ) -> None:
        self.window_size = window_size
        self.polynomial_order = polynomial_order
        self.mode = mode

    def fit(self, X: np.ndarray, y=None) -> "SavitzkyGolayFilter":
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
        self : SavitzkyGolayFilter
            The fitted transformer.
        """
        # Check that X is a 2D array and has only finite values
        X = validate_data(
            self, X, y="no_validation", ensure_2d=True, reset=True, dtype=np.float64
        )
        return self

    def transform(self, X: np.ndarray, y=None) -> np.ndarray:
        """
        Transform the input data by calculating the Savitzky-Golay filter.

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

        # Calculate the standard normal variate
        for i, x in enumerate(X_):
            X_[i] = self._calculate_smoothing(x)

        return X_.reshape(-1, 1) if X_.ndim == 1 else X_

    def _calculate_smoothing(self, x) -> np.ndarray:
        return savgol_filter(
            x,
            self.window_size,
            self.polynomial_order,
            deriv=0,
            axis=0,
            mode=self.mode,
        )
