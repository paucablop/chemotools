import numpy as np
from scipy.ndimage import convolve1d
from sklearn.base import BaseEstimator, TransformerMixin, OneToOneFeatureMixin
from sklearn.utils.validation import check_is_fitted

from chemotools.utils.check_inputs import check_input


class NorrisWilliams(OneToOneFeatureMixin, BaseEstimator, TransformerMixin):
    """
    A transformer that calculates the Norris-Williams derivative of the input data.

    Parameters
    ----------
    window_size : int, optional
        The size of the window to use for the derivative calculation. Must be odd. Default is 5.

    gap_size : int, optional
        The size of the gap to use for the derivative calculation. Must be odd. Default is 3.

    derivative_order : int, optional
        The order of the derivative to calculate. Can be 1 or 2. Default is 1.

    mode : str, optional
        The mode to use for the derivative calculation. Can be "nearest", "constant", 
        "reflect", "wrap", "mirror" or "interp". Default is "nearest".

    Attributes
    ----------
    n_features_in_ : int
        The number of features in the input data.

    _is_fitted : bool
        Whether the transformer has been fitted to data.

    Methods
    -------
    fit(X, y=None)
        Fit the transformer to the input data.

    transform(X, y=0, copy=True)
        Transform the input data by calculating the Norris-Williams derivative.
    """
    def __init__(
        self,
        window_size: int = 5,
        gap_size: int = 3,
        derivative_order: int = 1,
        mode="nearest",
    ):
        self.window_size = window_size
        self.gap_size = gap_size
        self.derivative_order = derivative_order
        self.mode = mode

    def fit(self, X: np.ndarray, y=None) -> "NorrisWilliams":
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
        X = check_input(X)

        # Set the number of features
        self.n_features_in_ = X.shape[1]

        # Set the fitted attribute to True
        self._is_fitted = True

        return self

    def transform(self, X: np.ndarray, y=None):
        """
        Transform the input data by calculating the Norris-Williams derivative.

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
        check_is_fitted(self, "_is_fitted")

        # Check that X is a 2D array and has only finite values
        X = check_input(X)
        X_ = X.copy()

        if X_.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features but got {X_.shape[1]}"
            )

        if self.derivative_order == 1:
            for i, x in enumerate(X_):
                derivative = self._spectrum_first_derivative(x)
                X_[i] = derivative
            return X_.reshape(-1, 1) if X_.ndim == 1 else X_

        if self.derivative_order == 2:
            for i, x in enumerate(X_):
                derivative = self._spectrum_second_derivative(x)
                X_[i] = derivative
            return X_.reshape(-1, 1) if X_.ndim == 1 else X_

        raise ValueError(
            f"Expected derivative_order to be 1 or 2 but got {self.derivative_order}"
        )

    def _smoothing_kernel(self):
        return np.ones(self.window_size) / self.window_size

    def _first_derivaive_kernel(self):
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
        first_derivative_kenel = self._first_derivaive_kernel()
        smoothed = convolve1d(X, smoothing_kernel, mode=self.mode)
        derivative = convolve1d(smoothed, first_derivative_kenel, mode=self.mode)
        return derivative

    def _spectrum_second_derivative(self, X):
        # Apply filter of data
        smoothing_kernel = self._smoothing_kernel()
        second_derivative_kernel = self._second_derivative_kernel()
        smoothed = convolve1d(X, smoothing_kernel, mode=self.mode)
        derivative = convolve1d(smoothed, second_derivative_kernel, mode=self.mode)
        return derivative
