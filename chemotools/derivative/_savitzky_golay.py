from typing import Literal

import numpy as np
from scipy.signal import savgol_filter
from sklearn.base import BaseEstimator, TransformerMixin, OneToOneFeatureMixin
from sklearn.utils.validation import check_is_fitted, validate_data


class SavitzkyGolay(TransformerMixin, OneToOneFeatureMixin, BaseEstimator):
    """
    A transformer that calculates the Savitzky-Golay derivative of the input data.

    Parameters
    ----------
    window_size : int, optional
        The size of the window to use for the derivative calculation. Must be odd. Default
        is 3.

    polynomial_order : int, optional
        The order of the polynomial to use for the derivative calculation. Must be less
        than window_size. Default is 1.

    derivate_order : int, optional
        The order of the derivative to calculate. Default is 1.

    mode : str, optional
        The mode to use for the derivative calculation. Can be "nearest", "constant",
        "reflect", "wrap", "mirror" or "interp". Default is "nearest".

    Methods
    -------
    fit(X, y=None)
        Fit the transformer to the input data.

    transform(X, y=0, copy=True)
        Transform the input data by calculating the Savitzky-Golay derivative.
    """

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
            Ignored.

        Returns
        -------
        self : NorrisWilliams
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
            Ignored.

        Returns
        -------
        X_ : np.ndarray of shape (n_samples, n_features)
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

        if X_.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features but got {X_.shape[1]}"
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
