from typing import Optional, Union
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.utils.validation import validate_data


from ._base import _ModelDiagnosticsBase, ModelType


class Leverage(_ModelDiagnosticsBase):
    """
    Calculate the leverage on the latent space of a PCA or PLS models.

    Parameters
    ----------
    model : Union[ModelType, Pipeline]
        A fitted PCA/PLS model or Pipeline ending with such a model

    confidence : float, default=0.95
        Confidence level for statistical calculations (between 0 and 1)

    Attributes
    ----------
    model_ : ModelType
        The fitted model of type _BasePCA or _PLS

    preprocessing_ : Optional[Pipeline]
        Preprocessing steps before the model

    References
    ----------

    """

    def __init__(self, model: Union[ModelType, Pipeline]) -> None:
        super().__init__(model)

    def predict(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """Calculate Hotelling's T-squared statistics for input data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data

        Returns
        -------
        ndarray of shape (n_samples,)
            DModX statistics for each sample
        """
        X = validate_data(
            self, X, y="no_validation", ensure_2d=True, reset=True, dtype=np.float64
        )

        if self.preprocessing_:
            X = self.preprocessing_.transform(X)

        # Calculate the leverage of the samples
        return calculate_leverage(self.model_, X)


def calculate_leverage(model: ModelType, X: np.ndarray) -> np.ndarray:
    """
    Calculate the leverage of the training samples in a PLS/PCA-like model.

    Parameters
    ----------
    model : Union[_BasePCA, _PLS]
        A fitted PCA/PLS model

    X : np.ndarray
        Input data

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
