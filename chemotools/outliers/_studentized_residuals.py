"""
The :mod:`chemotools.outliers._studentized_residuals` module
implements the Studentized Residuals outlier detection algorithm.
"""

# Authors: Pau Cabaneros
# License: MIT

from typing import Optional, Union

import numpy as np
from sklearn.cross_decomposition._pls import _PLS
from sklearn.pipeline import Pipeline

from ._base import _ModelResidualsBase
from ._leverage import calculate_leverage


class StudentizedResiduals(_ModelResidualsBase):
    """
    Calculate the Studentized Residuals on a _PLS model preditions.

    Parameters
    ----------
    model : Union[ModelType, Pipeline]
        A fitted _PLS model or Pipeline ending with such a model

    confidence : float, default=0.95
        Confidence level for statistical calculations (between 0 and 1)

    Attributes
    ----------
    estimator_ : ModelType
        The fitted model of type _BasePCA or _PLS

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

    Methods
    -------
    fit(X, y=None)
        Fit the Studentized Residuals model by computing
        residuals from the training set. Calculates the critical
        threshold based on the chosen method.

    predict(X, y=None)
        Identify outliers in the input data based on Studentized Residuals threshold.

    predict_residuals(X, y=None, validate=True)
        Calculate Studentized Residuals for input data.

    _calculate_critical_value(X)
        Calculate the critical value for outlier detection using the specified method.

    Examples
    --------
    >>> from chemotools.datasets import load_fermentation_train
    >>> from chemotools.outliers import StudentizedResiduals
    >>> from sklearn.cross_decomposition import PLSRegression
    >>> # Load sample data
    >>> X, y = load_fermentation_train()
    >>> y = y.values
    >>> # Instantiate the PLS model
    >>> pls = PLSRegression(n_components=3).fit(X, y)
    >>> # Initialize StudentizedResiduals with the fitted PLS model
    >>> studentized_residuals = StudentizedResiduals(model=pls, confidence=0.95)
    StudentizedResiduals(model=PLSRegression(n_components=3), confidence=0.95)
    >>> studentized_residuals.fit(X, y)
    >>> # Predict outliers in the dataset
    >>> outliers = studentized_residuals.predict(X, y)
    >>> # Calculate Studentized residuals
    >>> studentized_residuals_stats = studentized_residuals.predict_residuals(X, y)

    References
    ----------
    [1] Kim H. Esbensen,
        "Multivariate Data Analysis - In Practice", 5th Edition, 2002.
    """

    estimator_: _PLS

    def __init__(self, model: Union[_PLS, Pipeline], confidence=0.95) -> None:
        super().__init__(model, confidence)

    def _fit_residuals(self, X: np.ndarray, y: Optional[np.ndarray]) -> None:
        """Compute studentized residuals from training data and set critical value."""
        y_residuals = self._prepare_y_residuals(X, y)
        studentized_residuals = calculate_studentized_residuals(
            self.estimator_, X, y_residuals
        )
        self.critical_value_ = np.percentile(
            studentized_residuals, self.confidence_ * 100
        )

    def _compute_residuals(self, X: np.ndarray, y: Optional[np.ndarray]) -> np.ndarray:
        """Calculate the studentized residuals of the model predictions."""
        y_residuals = self._prepare_y_residuals(X, y)
        return calculate_studentized_residuals(self.estimator_, X, y_residuals)

    def _prepare_y_residuals(
        self, X: np.ndarray, y: Optional[np.ndarray]
    ) -> np.ndarray:
        """Compute prediction residuals from y, raising if y is None."""
        if y is None:
            raise ValueError("y cannot be None for studentized residuals")

        y_arr = np.asarray(y)
        if y_arr.ndim == 1:
            y_arr = y_arr.reshape(-1, 1)

        predictions = np.asarray(self.estimator_.predict(X))
        if predictions.ndim == 1:
            predictions = predictions.reshape(-1, 1)

        y_residuals = y_arr - predictions
        return y_residuals.reshape(-1, 1) if y_residuals.ndim == 1 else y_residuals


def calculate_studentized_residuals(
    model: _PLS, X: np.ndarray, y_residuals: np.ndarray
) -> np.ndarray:
    """Calculate the studentized residuals of the model predictions.

    Parameters
    ----------
    model : _PLS
        A fitted model

    X : array-like of shape (n_samples, n_features)
        Input data

    y : array-like of shape (n_samples,)
        Target values

    Returns
    -------
    ndarray of shape (n_samples,)
        Studentized residuals of the model predictions
    """

    # Calculate the leverage of the samples
    leverage = calculate_leverage(X, model)

    # Calculate the standard deviation of the residuals
    std = np.sqrt(np.sum(y_residuals**2, axis=0) / (X.shape[0] - model.n_components))

    return (y_residuals / (std * np.sqrt(1 - leverage.reshape(-1, 1)))).flatten()
