from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from sklearn.base import BaseEstimator, OutlierMixin

from chemotools._types import EstimatorType, ModelInput
from chemotools._validation import get_model_parameters, validate_and_extract_model

# Backward-compatible alias – existing consumers import this name.
ModelTypes = EstimatorType


class _ModelResidualsBase(ABC, BaseEstimator, OutlierMixin):
    """Base class for model outlier calculations.

    Implements statistical calculations for outlier detection in dimensionality
    reduction models like PCA and PLS.

    Parameters
    ----------
    model : Union[ModelTypes, Pipeline]
        A fitted _BasePCA or _PLS models or Pipeline ending with such a model
    confidence : float
        Confidence level for statistical calculations (between 0 and 1)

    Attributes
    ----------
    estimator_ : ModelTypes
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
    """

    def __init__(
        self,
        model: ModelInput,
        confidence: float,
    ) -> None:
        self.estimator_, self.transformer_ = validate_and_extract_model(model)
        self.n_features_in_, self.n_components_, self.n_samples_ = get_model_parameters(
            self.estimator_
        )
        self.confidence = _validate_confidence(confidence)

    def predict(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """Identify outliers in the input data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data

        y : array-like of shape (n_samples,), default=None
            Target values

        Returns
        -------
        ndarray of shape (n_samples,)
            Returns -1 for outliers and 1 for inliers
        """
        residuals = self.predict_residuals(X, y, validate=True)
        return np.where(residuals > self.critical_value_, -1, 1)

    def fit_predict_residuals(
        self, X: np.ndarray, y: Optional[np.ndarray] = None, validate: bool = True
    ) -> np.ndarray:
        """Fit the model to the input data and calculate the residuals.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data

        y : array-like of shape (n_samples,), default=None
            Target values

        Returns
        -------
        ndarray of shape (n_samples,)
            The residuals of the model
        """
        self.fit(X, y)
        return self.predict_residuals(X, y, validate)

    @abstractmethod
    def predict_residuals(
        self, X: np.ndarray, y: Optional[np.ndarray], validate: bool
    ) -> np.ndarray:
        """Calculate the residuals of the model.

        Returns
        -------
        ndarray of shape (n_samples,)
            The residuals of the model
        """

    @abstractmethod
    def _calculate_critical_value(self, X: np.ndarray) -> float:
        """Calculate the critical value for outlier detection.

        Returns
        -------
        float
            The calculated critical value for outlier detection
        """


def _validate_confidence(confidence: float) -> float:
    """Validate parameters using sklearn conventions.

    Parameters
    ----------
    confidence : float
        Confidence level for statistical calculations (between 0 and 1)

    Returns
    -------
    float
        The validated confidence level

    Raises
    ------
    ValueError
        If confidence is not between 0 and 1
    """
    if not 0 < confidence < 1:
        raise ValueError("Confidence must be between 0 and 1")
    return confidence
