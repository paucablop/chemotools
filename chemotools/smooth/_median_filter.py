"""
The :mod:`chemotools.smooth._median_filter` module implements the Median Filter (MD) transformation.
"""

# Authors: Pau Cabaneros
# License: MIT

from typing import Literal
from numbers import Integral

import numpy as np
from scipy.ndimage import median_filter
from sklearn.base import BaseEstimator, TransformerMixin, OneToOneFeatureMixin
from sklearn.utils.validation import check_is_fitted, validate_data
from sklearn.utils._param_validation import Interval, StrOptions


class MedianFilter(TransformerMixin, OneToOneFeatureMixin, BaseEstimator):
    """
    A smoothing transformer that calculates the median filter of the input data.

    Parameters
    ----------
    window_size : int, optional, default=3
        The size of the window to use for the median filter. Must be odd. Default is 3.

    mode : str, optional, default="nearest"
        The mode to use for the median filter. Can be "nearest", "constant", "reflect",
        "wrap", "mirror" or "interp". Default is "nearest".

    Attributes
    ----------
    n_features_in_ : int
        The number of features in the training data.

    Examples
    --------
    >>> from chemotools.datasets import load_fermentation_train
    >>> from chemotools.smooth import MedianFilter
    >>> # Load sample data
    >>> X, _ = load_fermentation_train()
    >>> # Initialize MedianFilter
    >>> md = MedianFilter()
    MedianFilter()
    >>> # Fit and transform the data
    >>> X_smoothed = md.fit_transform(X)
    """

    _parameter_constraints: dict = {
        "window_size": [Interval(Integral, 3, None, closed="left")],
        "mode": [
            StrOptions({"nearest", "constant", "reflect", "wrap", "mirror", "interp"})
        ],
    }

    def __init__(
        self,
        window_size: int = 3,
        mode: Literal[
            "reflect",
            "constant",
            "nearest",
            "mirror",
            "wrap",
            "grid-constant",
            "grid-mirror",
            "grid-wrap",
        ] = "nearest",
    ) -> None:
        self.window_size = window_size
        self.mode = mode

    def fit(self, X: np.ndarray, y=None) -> "MedianFilter":
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
        self : MedianFilter
            The fitted transformer.
        """
        # Check that X is a 2D array and has only finite values
        X = validate_data(
            self, X, y="no_validation", ensure_2d=True, reset=True, dtype=np.float64
        )
        return self

    def transform(self, X: np.ndarray, y=None) -> np.ndarray:
        """
        Transform the input data by calculating the median filter.

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

        # Mean filter the data
        for i, x in enumerate(X_):
            X_[i] = median_filter(x, size=self.window_size, mode=self.mode)
        return X_.reshape(-1, 1) if X_.ndim == 1 else X_
