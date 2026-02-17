"""
The :mod:`chemotools.outliers._leverage` module implements the Leverage
outlier detection algorithm.
"""

# Authors: Pau Cabaneros
# License: MIT

from typing import Optional, Union

import numpy as np
from sklearn.pipeline import Pipeline

from ._base import ModelTypes, _ModelResidualsBase


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

    def __init__(
        self, model: Union[ModelTypes, Pipeline], confidence: float = 0.95
    ) -> None:
        super().__init__(model, confidence)

    def _fit_residuals(self, X: np.ndarray, y: Optional[np.ndarray]) -> None:
        """Calculate the critical value for leverage using the percentile method."""
        leverage = calculate_leverage(X, self.estimator_)
        self.critical_value_ = np.percentile(leverage, self.confidence_ * 100)

    def _compute_residuals(self, X: np.ndarray, y: Optional[np.ndarray]) -> np.ndarray:
        """Calculate the leverage of the samples."""
        return calculate_leverage(X, self.estimator_)


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
