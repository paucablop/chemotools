"""
The :mod:`chemotools.derivative._savitzky_golay` module implements the Savitzky-Golay
transformer to calculate the Savitzky-Golay derivative of spectral data.
"""

# Author: Pau Cabaneros
# License: MIT

from numbers import Integral
from typing import Literal

import numpy as np
from scipy.signal import savgol_filter
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.validation import check_is_fitted, validate_data

from chemotools._deprecation import (
    DEPRECATED_PARAMETER,
    deprecated_parameter_constraint,
    resolve_renamed_parameter,
)
from chemotools._doc_mixin import DocLinkMixin


class SavitzkyGolay(
    DocLinkMixin, TransformerMixin, OneToOneFeatureMixin, BaseEstimator
):
    """
    A transformer that calculates the Savitzky-Golay derivative of the input data.

    Parameters
    ----------
    window_length : int, optional, default=3
        The size of the window to use for the derivative
        calculation. Must be odd. Default is 3.

    polyorder : int, optional, default=1
        The order of the polynomial to use for the derivative calculation. Must be less
        than ``window_length``. Default is 1.

    deriv : int, optional, default=1
        The order of the derivative to calculate. Default is 1.

    window_size : int, optional
        Deprecated alias for ``window_length``.

    polynomial_order : int, optional
        Deprecated alias for ``polyorder``.

    derivate_order : int, optional
        Deprecated alias for ``deriv``.

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
    >>> transformer = SavitzkyGolay(window_length=3, polyorder=1)
    SavitzkyGolay()
    >>> transformer.fit(X)
    >>> # Calculate Savitzky-Golay derivative
    >>> X_corrected = transformer.transform(X)
    """

    _parameter_constraints: dict = {
        "window_length": [Interval(Integral, 3, None, closed="left")],
        "polyorder": [Interval(Integral, 0, None, closed="left")],
        "deriv": [Interval(Integral, 0, None, closed="left")],
        "window_size": [
            Interval(Integral, 3, None, closed="left"),
            deprecated_parameter_constraint(),
        ],
        "polynomial_order": [
            Interval(Integral, 0, None, closed="left"),
            deprecated_parameter_constraint(),
        ],
        "derivate_order": [
            Interval(Integral, 0, None, closed="left"),
            deprecated_parameter_constraint(),
        ],
        "mode": [StrOptions({"nearest", "constant", "wrap", "mirror", "interp"})],
    }

    def __init__(
        self,
        window_length: int = 3,
        polyorder: int = 1,
        deriv: int = 1,
        mode: Literal["mirror", "constant", "nearest", "wrap", "interp"] = "nearest",
        window_size=DEPRECATED_PARAMETER,
        polynomial_order=DEPRECATED_PARAMETER,
        derivate_order=DEPRECATED_PARAMETER,
    ) -> None:
        self.window_length = window_length
        self.polyorder = polyorder
        self.deriv = deriv
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
        # Validate the input parameters
        self._validate_params()

        # Check that X is a 2D array and has only finite values
        X = validate_data(
            self, X, y="no_validation", ensure_2d=True, reset=True, dtype=np.float64
        )

        self.window_length_ = resolve_renamed_parameter(
            new_name="window_length",
            new_value=self.window_length,
            new_default=3,
            old_name="window_size",
            old_value=self.window_size,
        )
        self.polyorder_ = resolve_renamed_parameter(
            new_name="polyorder",
            new_value=self.polyorder,
            new_default=1,
            old_name="polynomial_order",
            old_value=self.polynomial_order,
        )
        self.deriv_ = resolve_renamed_parameter(
            new_name="deriv",
            new_value=self.deriv,
            new_default=1,
            old_name="derivate_order",
            old_value=self.derivate_order,
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
                self.window_length_,
                self.polyorder_,
                deriv=self.deriv_,
                axis=0,
                mode=self.mode,
            )

        return X_.reshape(-1, 1) if X_.ndim == 1 else X_
