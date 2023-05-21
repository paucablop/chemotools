import numpy as np
from scipy.signal import savgol_filter
from sklearn.base import BaseEstimator, TransformerMixin, OneToOneFeatureMixin
from sklearn.utils.validation import check_is_fitted

from chemotools.utils.check_inputs import check_input


class SavitzkyGolayFilter(OneToOneFeatureMixin, BaseEstimator, TransformerMixin):
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
        The number of features in the input data.

    _is_fitted : bool
        Whether the transformer has been fitted to data.

    Methods
    -------
    fit(X, y=None)
        Fit the transformer to the input data.

    transform(X, y=0, copy=True)
        Transform the input data by calculating the Savitzky-Golay filter.
    """
    def __init__(
        self, window_size: int = 3, polynomial_order: int = 1, mode: str = "nearest"
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
            Ignored.

        Returns
        -------
        self : SavitzkyGolayFilter
            The fitted transformer.
        """
        # Check that X is a 2D array and has only finite values
        X = check_input(X)

        # Set the number of features
        self.n_features_in_ = X.shape[1]

        # Set the fitted attribute to True
        self._is_fitted = True

        return self

    def transform(self, X: np.ndarray, y=None) -> np.ndarray:
        """
        Transform the input data by calculating the Savitzky-Golay filter.

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
