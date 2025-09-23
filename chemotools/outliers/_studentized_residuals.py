"""
The :mod:`chemotools.outliers._studentized_residuals` module implements the Studentized Residuals
outlier detection algorithm.
"""

# Authors: Pau Cabaneros
# License: MIT

from typing import Optional, Union
import numpy as np

from sklearn.cross_decomposition._pls import _PLS
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import validate_data, check_is_fitted
from sklearn.utils._param_validation import Interval, Real

from ._base import _ModelResidualsBase, ModelTypes
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
        Fit the Studentized Residuals model by computing residuals from the training set.
        Calculates the critical threshold based on the chosen method.

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

    _parameter_constraints: dict = {
        "model": [Pipeline, ModelTypes],
        "confidence": [Interval(Real, 0, 1, closed="both")],
    }

    def __init__(self, model: Union[_PLS, Pipeline], confidence=0.95) -> None:
        super().__init__(model, confidence)

    def fit(self, X: np.ndarray, y: Optional[np.ndarray]) -> "StudentizedResiduals":
        """
        Fit the model to the input data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data

        y : array-like of shape (n_samples,)
            Target data

        Returns
        -------
        self : StudentizedResiduals
            Fitted estimator with the critical threshold computed
        """
        # Validate the input data
        X = validate_data(
            self, X, y="no_validation", ensure_2d=True, reset=True, dtype=np.float64
        )

        # Preprocess the data
        if self.transformer_:
            X = self.transformer_.transform(X)

        # Calculate y residuals
        y_residuals = y - self.estimator_.predict(X)
        y_residuals = (
            y_residuals.reshape(-1, 1) if len(y_residuals.shape) == 1 else y_residuals
        )

        # Calculate the studentized residuals
        studentized_residuals = calculate_studentized_residuals(
            self.estimator_, X, y_residuals
        )

        # Calculate the critical threshold
        self.critical_value_ = self._calculate_critical_value(studentized_residuals)

        return self

    def predict(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """Calculate studentized residuals in the model predictions. and return a boolean array indicating outliers.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data

        y : array-like of shape (n_samples,)
            Target data

        Returns
        -------
        ndarray of shape (n_samples,)
            Studentized residuals of the predictions
        """
        return super().predict(X, y)

    def predict_residuals(
        self, X: np.ndarray, y: Optional[np.ndarray], validate: bool = True
    ) -> np.ndarray:
        """Calculate the studentized residuals of the model predictions.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data

        y : array-like of shape (n_samples,)
            Target values

        Returns
        -------
        ndarray of shape (n_samples,)
            Studentized residuals of the model predictions
        """
        # Check the estimator has been fitted
        check_is_fitted(self, ["critical_value_"])

        # Validate the input data
        if validate:
            X = validate_data(self, X, ensure_2d=True, dtype=np.float64)

        # Apply preprocessing if available
        if self.transformer_:
            X = self.transformer_.transform(X)

        # Calculate y residuals
        y_residuals = y - self.estimator_.predict(X)
        y_residuals = (
            y_residuals.reshape(-1, 1) if len(y_residuals.shape) == 1 else y_residuals
        )

        return calculate_studentized_residuals(self.estimator_, X, y_residuals)

    def _calculate_critical_value(self, X: np.ndarray) -> float:
        """Calculate the critical value for outlier detection.

        Parameters
        ----------
        X : array-like of shape (n_samples,)
            Studentized residuals

        Returns
        -------
        float
            The calculated critical value for outlier detection
        """

        return np.percentile(X, self.confidence * 100) if X is not None else 0.0


def calculate_studentized_residuals(
    model: ModelTypes, X: np.ndarray, y_residuals: np.ndarray
) -> np.ndarray:
    """Calculate the studentized residuals of the model predictions.

    Parameters
    ----------
    model : ModelTypes
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
