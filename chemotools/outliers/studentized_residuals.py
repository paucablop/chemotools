from typing import Optional, Union
import numpy as np

from sklearn.cross_decomposition._pls import _PLS
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import validate_data, check_is_fitted


from ._base import _ModelResidualsBase, ModelTypes
from .leverage import calculate_leverage


class StudentizedResiduals(_ModelResidualsBase):
    """
    Calculate the Studentized Residuals on a _PLS model preditions.

    Parameters
    ----------
    model : Union[ModelType, Pipeline]
        A fitted _PLS model or Pipeline ending with such a model

    Attributes
    ----------
    model_ : ModelType
        The fitted model of type _BasePCA or _PLS

    preprocessing_ : Optional[Pipeline]
        Preprocessing steps before the model

    References
    ----------

    """

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
        """
        # Validate the input data
        X = validate_data(
            self, X, y="no_validation", ensure_2d=True, reset=True, dtype=np.float64
        )

        # Preprocess the data
        if self.preprocessing_:
            X = self.preprocessing_.transform(X)

        # Calculate y residuals
        y_residuals = y - self.model_.predict(X)
        y_residuals = (
            y_residuals.reshape(-1, 1) if len(y_residuals.shape) == 1 else y_residuals
        )

        # Calculate the studentized residuals
        studentized_residuals = calculate_studentized_residuals(
            self.model_, X, y_residuals
        )

        # Calculate the critical threshold
        self.critical_value_ = self._calculate_critical_value(studentized_residuals)

        return self

    def predict(self, X: np.ndarray, y: Optional[np.ndarray]) -> np.ndarray:
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
        # Check the estimator has been fitted
        check_is_fitted(self, ["critical_value_"])

        # Validate the input data
        X = validate_data(
            self, X, y="no_validation", ensure_2d=True, reset=True, dtype=np.float64
        )

        # Preprocess the data
        if self.preprocessing_:
            X = self.preprocessing_.transform(X)

        # Calculate y residuals
        y_residuals = y - self.model_.predict(X)
        y_residuals = (
            y_residuals.reshape(-1, 1) if len(y_residuals.shape) == 1 else y_residuals
        )

        # Calculate the studentized residuals
        studentized_residuals = calculate_studentized_residuals(
            self.model_, X, y_residuals
        )
        return np.where(studentized_residuals > self.critical_value_, -1, 1)

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
        if self.preprocessing_:
            X = self.preprocessing_.transform(X)

        # Calculate y residuals
        y_residuals = y - self.model_.predict(X)
        y_residuals = (
            y_residuals.reshape(-1, 1) if len(y_residuals.shape) == 1 else y_residuals
        )

        return calculate_studentized_residuals(self.model_, X, y_residuals)

    def _calculate_critical_value(self, X: Optional[np.ndarray]) -> float:
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
    leverage = calculate_leverage(model, X)

    # Calculate the standard deviation of the residuals
    std = np.sqrt(np.sum(y_residuals**2, axis=0) / (X.shape[0] - model.n_components))

    return (y_residuals / (std * np.sqrt(1 - leverage.reshape(-1, 1)))).flatten()
