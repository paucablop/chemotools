from abc import ABC, abstractmethod
from typing import Union, Optional

import numpy as np

from sklearn.base import BaseEstimator, OutlierMixin
from sklearn.decomposition._base import _BasePCA
from sklearn.cross_decomposition._pls import _PLS
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from ._utils import validate_confidence, validate_and_extract_model

ModelTypes = Union[_BasePCA, _PLS]


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
        model: Union[ModelTypes, Pipeline],
        confidence: float,
    ) -> None:
        (
            self.estimator_,
            self.transformer_,
            self.n_features_in_,
            self.n_components_,
            self.n_samples_,
        ) = validate_and_extract_model(model)
        self.confidence = validate_confidence(confidence)

    def fit_predict_residuals(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
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
        return self.predict_residuals(X, y, validate=True)

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
    def _calculate_critical_value(self, X: Optional[np.ndarray]) -> float:
        """Calculate the critical value for outlier detection.

        Returns
        -------
        float
            The calculated critical value for outlier detection
        """