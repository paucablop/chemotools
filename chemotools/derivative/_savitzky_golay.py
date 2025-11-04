"""
The :mod:`chemotools.derivative._savitzky_golay` module implements the Savitzky-Golay
transformer to calculate the Savitzky-Golay derivative of spectral data.
"""

# Author: Pau Cabaneros
# License: MIT

from typing import Literal
from numbers import Integral

import numpy as np
from scipy.signal import savgol_filter
from sklearn.base import BaseEstimator, TransformerMixin, OneToOneFeatureMixin
from sklearn.utils.validation import check_is_fitted, validate_data
from sklearn.utils._param_validation import Interval, StrOptions


class SavitzkyGolay(TransformerMixin, OneToOneFeatureMixin, BaseEstimator):
    """
    A transformer that calculates the Savitzky-Golay derivative of the input data.

    Parameters
    ----------
    window_size : int, optional, default=3
        The size of the window to use for the derivative calculation. Must be odd. Default
        is 3.

    polynomial_order : int, optional, default=1
        The order of the polynomial to use for the derivative calculation. Must be less
        than window_size. Default is 1.

    derivative_order : int, optional, default=1
        The order of the derivative to calculate. Default is 1.

    mode : str, optional, default="nearest"
        The mode to use for the derivative calculation. Can be "nearest", "constant",
        "reflect", "wrap", "mirror" or "interp". Default is "nearest".

    Attributes
    ----------
    n_features_in_ : int
        The number of features in the input data.

    References
    ----------
    [1] Åsmund Rinnan, Frans van den Berg, Søren Balling Engelsen,
        "Review of the most common pre-processing techniques for near-infrared spectra,"
        TrAC Trends in Analytical Chemistry 28 (10) 1201-1222 (2009).

    Examples
    --------
    >>> from chemotools.derivative import SavitzkyGolay
    >>> from chemotools.datasets import load_fermentation_train
    >>> # Load sample data
    >>> X, _ = load_fermentation_train()
    >>> # Instantiate the transformer
    >>> transformer = SavitzkyGolay(window_size=3, polynomial_order=1)
    SavitzkyGolay()
    >>> transformer.fit(X)
    >>> # Calculate Savitzky-Golay derivative
    >>> X_corrected = transformer.transform(X)
    """

    _parameter_constraints: dict = {
        "window_size": [Interval(Integral, 3, None, closed="left")],
        "polynomial_order": [Interval(Integral, 0, None, closed="left")],
        "derivative_order": [Interval(Integral, 0, None, closed="left")],
        "mode": [
            StrOptions({"nearest", "constant", "reflect", "wrap", "mirror", "interp"})
        ],
    }

    def __init__(
        self,
        window_size: int = 3,
        polynomial_order: int = 1,
        derivate_order: int = 1,
        mode: Literal["mirror", "constant", "nearest", "wrap", "interp"] = "nearest",
    ) -> None:
        self.window_size = window_size
        self.polynomial_order = polynomial_order
        self.derivate_order = derivate_order
        self.mode = mode

    def fit(self, X: np.ndarray, y=None) -> "SavitzkyGolay":
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
        self : SavitzkyGolay
            The fitted transformer.
        """
        # Check that X is a 2D array and has only finite values
        X = validate_data(
            self, X, y="no_validation", ensure_2d=True, reset=True, dtype=np.float64
        )
        return self

    def transform(self, X: np.ndarray, y=None) -> np.ndarray:
        """
        Transform the input data by calculating the Savitzky-Golay derivative.

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
            X_[i] = savgol_filter(
                x,
                self.window_size,
                self.polynomial_order,
                deriv=self.derivate_order,
                axis=0,
                mode=self.mode,
            )

        return X_.reshape(-1, 1) if X_.ndim == 1 else X_
