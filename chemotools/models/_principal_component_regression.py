"""
The :mod:`chemotools.models._principal_component_regression` module implements a PCR model.
"""

# Authors: Ruggero Guerrini
# License: MIT

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.utils.validation import validate_data, check_is_fitted
from numbers import Integral
from sklearn.utils._param_validation import Interval, RealNotInt, StrOptions
import numpy as np


class PrincipalComponentRegression(RegressorMixin, BaseEstimator):
    """
    Description

    Parameters
    ----------
    n_components : int, default = 2
        The number of components used to calculate the PCA model
        # add comments on parameter constraints

    Attributes
    ----------
    pca_ : PCA objects from sklearn
        Fitted PCA model

    lr_ : Linear Regression objects from sklearn
        Fitted Linear Regression model

    Examples
    --------
    >>> from chemotools.decomposition import PrincipalComponentRegression
    >>> # Generate sample data
    >>> X = np.random.randn(100, 50)
    >>> X_test = np.random.randn(10, 50)
    >>> y = X[:, 0] + 2*X[:, 1] + np.random.randn(100)*0.1
    >>> # Fit model
    >>> pcr = PrincipalComponentRegression(n_components=2)
    >>> pcr.fit(X, y)
    >>> y_hat = pcr.predict(X_test)
    """

    _parameter_constraints: dict = {
        "n_components": [
            Interval(Integral, 0, None, closed="left"),
            Interval(RealNotInt, 0, 1, closed="neither"),
            StrOptions({"mle"}),
            None,
        ],
        "copy": ["boolean"],
    }

    def __init__(self, n_components: int | None = None, copy: bool = True):
        self.n_components = n_components
        self.copy = copy

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PrincipalComponentRegression":
        """
        Fit the model to the input data.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            The input data to fit the transformer to.

        y : np.ndarray of shape (n_samples, )
            The properties to be predicted.

        Returns
        -------
        self : PrincipalComponentRegression
            The regression model.
        """
        # Validate input data
        self._validate_params()
        X, y = validate_data(
            self,
            X,
            y,
            ensure_2d=True,
            reset=True,
            copy=self.copy,
            dtype=np.float64,
            multi_output=True,
        )

        # Train PCA model
        self.pca_ = PCA(n_components=self.n_components).fit(X)
        T = self.pca_.transform(X)

        # Train linear regression model
        self.lr_ = LinearRegression().fit(T, y)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict new data

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            The input data to predict.

        Returns
        -------
        y_hat : np.ndarray of shape (n_samples,)
            The predicted value.
        """
        # Validate input data
        check_is_fitted(self, ["pca_", "lr_"])
        X = validate_data(
            self,
            X,
            ensure_2d=True,
            reset=False,
            copy=self.copy,
            dtype=np.float64,
        )

        # Predict
        T = self.pca_.transform(X)
        return self.lr_.predict(T)
