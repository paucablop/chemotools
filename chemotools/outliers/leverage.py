from typing import Optional, Union
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.utils.validation import validate_data, check_is_fitted


from ._base import _ModelResidualsBase, ModelTypes


class Leverage(_ModelResidualsBase):
    """
    Calculate the leverage of the training samples on  the latent space of a PCA or PLS models.
    This method allows to detect datapoints with high leverage in the model.

    Parameters
    ----------
    model : Union[ModelType, Pipeline]
        A fitted PCA/PLS model or Pipeline ending with such a model

    Attributes
    ----------
    model_ : ModelType
        The fitted model of type _BasePCA or _PLS

    preprocessing_ : Optional[Pipeline]
        Preprocessing steps before the model

    References
    ----------

    """

    def __init__(
        self, model: Union[ModelTypes, Pipeline], confidence: float = 0.95
    ) -> None:
        super().__init__(model, confidence)

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "Leverage":
        """
        Fit the model to the input data.

        Parameters

        """
        X = validate_data(
            self, X, y="no_validation", ensure_2d=True, reset=True, dtype=np.float64
        )

        if self.preprocessing_:
            X = self.preprocessing_.fit_transform(X)

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
        # Check the estimator has been fitted
        check_is_fitted(self, ["critical_value_"])

        # Validate the input data
        X = validate_data(
            self, X, y="no_validation", ensure_2d=True, reset=True, dtype=np.float64
        )

        # Preprocess the data
        if self.preprocessing_:
            X = self.preprocessing_.transform(X)

        # Calculate outliers based on samples with too high leverage
        leverage = calculate_leverage(self.model_, X)
        return np.where(leverage > self.critical_value_, -1, 1)

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
        if self.preprocessing_:
            X = self.preprocessing_.transform(X)

        # Calculate the leverage
        return calculate_leverage(self.model_, X)

    def _calculate_critical_value(self, X: Optional[np.ndarray]) -> float:
        """Calculate the critical value for outlier detection using the percentile outlier method."""

        # Calculate the leverage of the samples
        leverage = calculate_leverage(self.model_, X)

        # Calculate the critical value
        return np.percentile(leverage, self.confidence * 100)


def calculate_leverage(model: ModelTypes, X: Optional[np.ndarray]) -> np.ndarray:
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
