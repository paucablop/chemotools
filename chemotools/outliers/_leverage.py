"""
The :mod:`chemotools.outliers._leverage` module implements the Leverage
outlier detection algorithm.
"""

# Authors: Pau Cabaneros
# License: MIT

from typing import Optional, Union
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.utils.validation import validate_data, check_is_fitted
from sklearn.utils._param_validation import Interval, Real


from ._base import _ModelResidualsBase, ModelTypes


class Leverage(_ModelResidualsBase):
    """
    Calculate the leverage of the training samples on  the latent space of a PLS model.
    This method allows to detect datapoints with high leverage in the model.

    Parameters
    ----------
    model : Union[ModelType, Pipeline]
        A fitted PLSRegression model or Pipeline ending with such a model

    confidence : float, default=0.95
        Confidence level for statistical calculations (between 0 and 1)

    Attributes
    ----------
    estimator_ : ModelType
        The fitted model of type _PLS

    transformer_ : Optional[Pipeline]
        Preprocessing steps before the model

    n_features_in_ : int
        Number of features in the input data

    n_components_ : int
        Number of components in the model

    n_samples_ : int
        Number of samples used to train the model

    critical_value_ : float
        The calculated critical value for outlier detection

    References
    ----------
    [1] Kim H. Esbensen,
        "Multivariate Data Analysis - In Practice", 5th Edition, 2002.

    Examples
    --------
    >>> from sklearn.cross_decomposition import PLSRegression
    >>> from chemotools.outliers import Leverage
    >>> X = np.random.rand(100, 10)
    >>> y = np.random.rand(100)
    >>> pls = PLSRegression(n_components=3).fit(X, y)
    >>> # Initialize Leverage with the fitted PLS model
    >>> leverage = Leverage(pls, confidence=0.95)
    Leverage(model=PLSRegression(n_components=3), confidence=0.95)
    >>> leverage.fit(X, y)
    >>> # Predict outliers in the dataset
    >>> outliers = leverage.predict(X)
    >>> # Get the leverage of the samples
    >>> residuals = leverage.predict_residuals(X)
    """

    _parameter_constraints: dict = {
        "model": [Pipeline, ModelTypes],
        "confidence": [Interval(Real, 0, 1, closed="both")],
    }

    def __init__(
        self, model: Union[ModelTypes, Pipeline], confidence: float = 0.95
    ) -> None:
        model, confidence = model, confidence
        super().__init__(model, confidence)

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "Leverage":
        """
        Fit the model to the input data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data

        y : array-like of shape (n_samples,), default=None
            Target data

        Returns
        -------
        self : Leverage
            Fitted estimator with the critical threshold computed
        """
        X = validate_data(
            self, X, y="no_validation", ensure_2d=True, reset=True, dtype=np.float64
        )

        if self.transformer_:
            X = self.transformer_.fit_transform(X)

        # Compute the critical threshold
        self.critical_value_ = self._calculate_critical_value(X)

        return self

    def predict(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """Calculate Leverage for training data on the model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data

        Returns
        -------
        ndarray of shape (n_samples,)
            Bool with samples with a leverage above the critical value
        """
        return super().predict(X, y)

    def predict_residuals(
        self, X: np.ndarray, y: Optional[np.ndarray] = None, validate: bool = True
    ) -> np.ndarray:
        """Calculate the leverage of the samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data

        Returns
        -------
        np.ndarray
            Leverage of the samples
        """
        # Check the estimator has been fitted
        check_is_fitted(self, ["critical_value_"])

        # Validate the input data
        if validate:
            X = validate_data(self, X, ensure_2d=True, dtype=np.float64)

        # Apply preprocessing if available
        if self.transformer_:
            X = self.transformer_.transform(X)

        # Calculate the leverage
        return calculate_leverage(X, self.estimator_)

    def _calculate_critical_value(self, X: np.ndarray) -> float:
        """Calculate the critical value for outlier detection using the percentile outlier method."""

        # Calculate the leverage of the samples
        leverage = calculate_leverage(X, self.estimator_)

        # Calculate the critical value
        return np.percentile(leverage, self.confidence * 100)


def calculate_leverage(X: np.ndarray, model: ModelTypes) -> np.ndarray:
    """
    Calculate the leverage of the training samples in a PLS/PCA-like model.

    Parameters
    ----------
    model : Union[_BasePCA, _PLS]
        A fitted PCA/PLS model

    X : np.ndarray
        Preprocessed input data

    Returns
    -------
    np.ndarray
        Leverage of the samples
    """

    X_transformed = model.transform(X)

    X_hat = (
        X_transformed @ np.linalg.inv(X_transformed.T @ X_transformed) @ X_transformed.T
    )

    return np.diag(X_hat)
