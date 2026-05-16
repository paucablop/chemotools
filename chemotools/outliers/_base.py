from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

import numpy as np
from sklearn.base import BaseEstimator, OutlierMixin
from sklearn.cross_decomposition._pls import _PLS
from sklearn.decomposition._base import _BasePCA
from sklearn.pipeline import Pipeline
from sklearn.utils._param_validation import Interval, Real
from sklearn.utils.validation import check_is_fitted, validate_data

if TYPE_CHECKING:
    from typing_extensions import Self

from chemotools._doc_mixin import DocLinkMixin
from chemotools._types import EstimatorType, ModelInput
from chemotools._validation import get_model_parameters, validate_and_extract_model

# Backward-compatible alias – existing consumers import this name.
ModelTypes = EstimatorType


class _ModelResidualsBase(DocLinkMixin, ABC, BaseEstimator, OutlierMixin):
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

    _parameter_constraints: dict = {
        "model": [Pipeline, _BasePCA, _PLS],
        "confidence": [Interval(Real, 0, 1, closed="neither")],
    }

    critical_value_: float

    def __init__(
        self,
        model: ModelInput,
        confidence: float,
    ) -> None:
        self.model = model
        self.confidence = confidence

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Self:
        """Fit the model to the input data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        y : array-like of shape (n_samples,), default=None
            Target values.

        Returns
        -------
        self
            Fitted estimator.
        """
        self._validate_params()

        # Validate and extract the model and its parameters
        self.estimator_, self.transformer_ = validate_and_extract_model(self.model)
        self.n_features_in_, self.n_components_, self.n_samples_ = get_model_parameters(
            self.estimator_
        )
        self.confidence_ = self.confidence

        # Validate input data
        X = validate_data(
            self, X, y="no_validation", ensure_2d=True, reset=True, dtype=np.float64
        )

        # Transform through preprocessing pipeline if available
        X_transformed = self.transformer_.transform(X) if self.transformer_ else X

        # Subclass-specific fitting
        self._fit_residuals(X_transformed, y)

        return self

    def predict(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """Identify outliers in the input data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        y : array-like of shape (n_samples,), default=None
            Target values. Required by some subclasses (e.g. StudentizedResiduals).

        Returns
        -------
        ndarray of shape (n_samples,)
            Returns -1 for outliers and 1 for inliers.
        """
        residuals = self.predict_residuals(X, y)
        return np.where(residuals > self.critical_value_, -1, 1)

    def predict_residuals(
        self, X: np.ndarray, y: Optional[np.ndarray] = None, validate: bool = True
    ) -> np.ndarray:
        """Calculate the residuals of the model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        y : array-like of shape (n_samples,), default=None
            Target values. Required by some subclasses (e.g. StudentizedResiduals).

        validate : bool, default=True
            Whether to validate the input data.

        Returns
        -------
        ndarray of shape (n_samples,)
            The residuals of the model.
        """
        check_is_fitted(self, ["critical_value_"])
        if validate:
            X = validate_data(
                self,
                X,
                y="no_validation",
                ensure_2d=True,
                reset=False,
                dtype=np.float64,
            )
        if self.transformer_:
            X = self.transformer_.transform(X)
        return self._compute_residuals(X, y)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Return anomaly scores for each sample.

        Provides an sklearn-compatible interface. Returns the residuals
        computed by ``predict_residuals``. Higher values indicate
        more anomalous samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        ndarray of shape (n_samples,)
            Anomaly scores for each sample.
        """
        return self.predict_residuals(X)

    def fit_predict_residuals(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Fit the model to the input data and calculate the residuals.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        y : array-like of shape (n_samples,), default=None
            Target values.

        Returns
        -------
        ndarray of shape (n_samples,)
            The residuals of the model.
        """
        self.fit(X, y)
        return self.predict_residuals(X, y)

    @abstractmethod
    def _fit_residuals(self, X: np.ndarray, y: Optional[np.ndarray]) -> None:
        """Compute subclass-specific training state and set ``critical_value_``.

        Called at the end of ``fit`` with validated and transformed data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Validated and transformed training data.

        y : ndarray or None
            Target values.
        """

    @abstractmethod
    def _compute_residuals(self, X: np.ndarray, y: Optional[np.ndarray]) -> np.ndarray:
        """Compute residuals from validated, transformed data.

        Called by ``predict_residuals`` after validation and transformation.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Validated and transformed input data.

        y : ndarray or None
            Target values.

        Returns
        -------
        ndarray of shape (n_samples,)
            Computed residuals.
        """
