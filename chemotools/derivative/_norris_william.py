"""
The :mod:`chemotools.derivative._norris_william` module implements the Norris-Williams
transformer to calculate the Norris-Williams derivative of spectral data.
"""

# Author: Pau Cabaneros
# License: MIT

from numbers import Integral
from typing import Literal

import numpy as np
from scipy.ndimage import convolve1d
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.validation import check_is_fitted, validate_data

from chemotools._deprecation import (
    DEPRECATED_PARAMETER,
    deprecated_parameter_constraint,
    resolve_renamed_parameter,
)


class NorrisWilliams(TransformerMixin, OneToOneFeatureMixin, BaseEstimator):
    """
    A transformer that calculates the Norris-Williams derivative of the input data.

    Parameters
    ----------
    window_length : int, optional, default=5
        The size of the window to use for the derivative
        calculation. Must be odd. Default is 5.

    gap_size : int, optional, default=3
        The size of the gap to use for the derivative
        calculation. Must be odd. Default is 3.

    deriv : int, optional, default=1
        The order of the derivative to calculate. Can be 1 or 2. Default is 1.

    mode : str, optional, default="nearest"
        The mode to use for the derivative calculation. Can be "nearest", "constant",
        "reflect", "wrap", "mirror" or "interp". Default is "nearest".

    window_size : int, optional
        Deprecated alias for ``window_length``.

    derivative_order : int, optional
        Deprecated alias for ``deriv``.

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
    >>> from chemotools.derivative import NorrisWilliams
    >>> from chemotools.datasets import load_fermentation_train
    >>> # Load sample data
    >>> X, _ = load_fermentation_train()
    >>> # Instantiate the transformer
    >>> transformer = NorrisWilliams(window_size=5, gap_size=3)
    NorrisWilliams()
    >>> transformer.fit(X)
    >>> # Calculate Norris-Williams derivative
    >>> X_corrected = transformer.transform(X)
    """

    _parameter_constraints: dict = {
        "window_length": [Interval(Integral, 3, None, closed="left")],
        "gap_size": [Interval(Integral, 1, None, closed="left")],
        "deriv": [Interval(Integral, 1, 2, closed="both")],
        "mode": [StrOptions({"nearest", "constant", "reflect", "wrap", "mirror"})],
        "window_size": [
            Interval(Integral, 3, None, closed="left"),
            deprecated_parameter_constraint(),
        ],
        "derivative_order": [
            Interval(Integral, 1, 2, closed="both"),
            deprecated_parameter_constraint(),
        ],
    }

    def __init__(
        self,
        window_length: int = 5,
        gap_size: int = 3,
        deriv: int = 1,
        mode: Literal["nearest", "constant", "reflect", "wrap", "mirror"] = "nearest",
        window_size=DEPRECATED_PARAMETER,
        derivative_order=DEPRECATED_PARAMETER,
    ):
        self.window_length = window_length
        self.gap_size = gap_size
        self.deriv = deriv
        self.mode = mode
        self.window_size = window_size
        self.derivative_order = derivative_order

    def fit(self, X: np.ndarray, y=None) -> "NorrisWilliams":
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
        self : NorrisWilliams
            The fitted transformer.
        """
        # Check that X is a 2D array and has only finite values
        X = validate_data(
            self, X, y="no_validation", ensure_2d=True, reset=True, dtype=np.float64
        )

        self.window_length_ = resolve_renamed_parameter(
            new_name="window_length",
            new_value=self.window_length,
            new_default=5,
            old_name="window_size",
            old_value=self.window_size,
        )

        self.deriv_ = resolve_renamed_parameter(
            new_name="deriv",
            new_value=self.deriv,
            new_default=1,
            old_name="derivative_order",
            old_value=self.derivative_order,
        )

        return self

    def transform(self, X: np.ndarray, y=None):
        """
        Transform the input data by calculating the Norris-Williams derivative.

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

        if self.deriv_ == 1:
            for i, x in enumerate(X_):
                derivative = self._spectrum_first_derivative(x)
                X_[i] = derivative
            return X_.reshape(-1, 1) if X_.ndim == 1 else X_

        if self.deriv_ == 2:
            for i, x in enumerate(X_):
                derivative = self._spectrum_second_derivative(x)
                X_[i] = derivative
            return X_.reshape(-1, 1) if X_.ndim == 1 else X_

        raise ValueError(f"Expected deriv to be 1 or 2 but got {self.deriv_}")

    def _smoothing_kernel(self):
        return np.ones(self.window_length_) / self.window_length_

    def _first_derivative_kernel(self):
        array = np.zeros(self.gap_size)
        array[0] = 1 / (self.gap_size)
        array[-1] = -1 / (self.gap_size)
        return array

    def _second_derivative_kernel(self):
        array = np.zeros(self.gap_size)
        array[0] = 1 / (self.gap_size)
        array[-1] = 1 / (self.gap_size)
        array[int((self.gap_size - 1) / 2)] = -2 / self.gap_size
        return array

    def _spectrum_first_derivative(self, X):
        # Apply filter of data
        smoothing_kernel = self._smoothing_kernel()
        first_derivative_kernel = self._first_derivative_kernel()
        smoothed = convolve1d(X, smoothing_kernel, mode=self.mode)
        derivative = convolve1d(smoothed, first_derivative_kernel, mode=self.mode)
        return derivative

    def _spectrum_second_derivative(self, X):
        # Apply filter of data
        smoothing_kernel = self._smoothing_kernel()
        second_derivative_kernel = self._second_derivative_kernel()
        smoothed = convolve1d(X, smoothing_kernel, mode=self.mode)
        derivative = convolve1d(smoothed, second_derivative_kernel, mode=self.mode)
        return derivative
