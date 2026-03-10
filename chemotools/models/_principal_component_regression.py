"""
The :mod:`chemotools.scale._min_max_scaler` module implements a Min-Max Scaler transformer.
"""

# Authors: Ruggero Guerrini
# License: MIT

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.utils.validation import check_X_y, check_array
from sklearn.utils.validation import check_is_fitted
from numbers import Integral
from sklearn.utils._param_validation import Interval, RealNotInt, StrOptions
import numpy as np


class PCR(BaseEstimator, RegressorMixin):
    """
    Description

    Parameters
    ----------
    use_min : bool, default=True
        The normalization to use. If True, the data is subtracted by the minimum and
        scaled by the maximum. If False, the data is scaled by the maximum.

    Attributes
    ----------
    n_features_in_ : int
        The number of features in the input data.

    Examples
    --------
    >>> from chemotools.datasets import load_fermentation_train
    >>> from chemotools.scale import MinMaxScaler
    >>> # Load sample data
    >>> X, _ = load_fermentation_train()
    >>> # Initialize MinMaxScaler
    >>> scaler = MinMaxScaler()
    MinMaxScaler()
    >>> # Fit and transform the data
    >>> X_scaled = scaler.fit_transform(X)
    """

    _parameter_constraints: dict = {
        "n_components": [
            Interval(Integral, 0, None, closed="left"),
            Interval(RealNotInt, 0, 1, closed="neither"),
            StrOptions({"mle"}),
            None,
        ],
    }

    def __init__(self, n_components: int = 2):
        self.n_components = n_components

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PCR":
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
        self : MinMaxScaler
            The fitted transformer.
        """
        X, y = check_X_y(X, y)
        if not 1 <= self.n_components <= min(X.shape):
            raise ValueError("n_components must be between 1 and min(X.shape)")
        self.pca_ = PCA(n_components=self.n_components).fit(X)
        T = self.pca_.transform(X)
        self.lr_ = LinearRegression().fit(T, y)
        # expose pca and linear regression fitting attributes

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Transform the input data by scaling it.

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
        check_is_fitted(self, ["pca_", "lr_"])
        X = check_array(X)
        T = self.pca_.transform(X)
        return self.lr_.predict(T)
