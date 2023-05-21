import numpy as np
from scipy.ndimage import median_filter
from sklearn.base import BaseEstimator, TransformerMixin, OneToOneFeatureMixin
from sklearn.utils.validation import check_is_fitted

from chemotools.utils.check_inputs import check_input


class MedianFilter(OneToOneFeatureMixin, BaseEstimator, TransformerMixin):
    """
    A transformer that calculates the median filter of the input data.

    Parameters
    ----------
    window_size : int, optional
        The size of the window to use for the median filter. Must be odd. Default is 3.

    mode : str, optional
        The mode to use for the median filter. Can be "nearest", "constant", "reflect",
        "wrap", "mirror" or "interp". Default is "nearest".

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
        Transform the input data by calculating the median filter.
    """
    def __init__(self, window_size: int = 3, mode: str = 'nearest') -> None:
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
            Ignored.

        Returns
        -------
        self : MedianFilter
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
        Transform the input data by calculating the median filter.

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

        # Mean filter the data
        for i, x in enumerate(X_):
            X_[i] = median_filter(x, size=self.window_size, mode=self.mode)
        return X_.reshape(-1, 1) if X_.ndim == 1 else X_
