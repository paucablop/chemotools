from typing import Optional, Union
import numpy as np

from sklearn.cross_decomposition._pls import _PLS
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import validate_data


from ._base import _ModelDiagnosticsBase
from .leverage import calculate_leverage


class StudentizedResiduals(_ModelDiagnosticsBase):
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

    def __init__(self, model: Union[_PLS, Pipeline]) -> None:
        super().__init__(model)

    def predict(self, X: np.ndarray, y: Optional[np.ndarray]) -> np.ndarray:
        """Calculate studentized residuals in the model predictions.

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
        X = validate_data(
            self, X, y="no_validation", ensure_2d=True, reset=True, dtype=np.float64
        )

        if self.preprocessing_:
            X = self.preprocessing_.transform(X)

        # Calculate the leverage of the samples
        leverage = calculate_leverage(self.model_, X)

        # Calculate the residuals
        y_predict = self.model_.predict(X)

        residuals = y - y_predict
        residuals = residuals.reshape(-1, 1) if len(residuals.shape) == 1 else residuals

        # Calculate the standard deviation of the residuals
        std = np.sqrt(
            np.sum(residuals**2, axis=0) / (X.shape[0] - self.model_.n_components)
        )

        return residuals / (std * np.sqrt(1 - leverage.reshape(-1, 1)))
