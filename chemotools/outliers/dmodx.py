from typing import Optional, Union
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.utils.validation import validate_data
from scipy.stats import f as f_distribution


from ._base import _ModelDiagnostics, ModelType


class DModX(_ModelDiagnostics):
    """Calculate Distance to Model (DModX) statistics.

    DModX measures the distance between an observation and the model plane
    in the X-space, useful for detecting outliers.

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
        model: Union[ModelType, Pipeline],
        confidence: float = 0.95,
    ) -> None:
        super().__init__(model, confidence)

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "DModX":
        """
        Empty fit method to comply with sklearn API. Outlier detection does not need
        to be fitted to the data because it is based on an already fitted model and not
        on the data itself.
        """
        return self

    def predict_residuals(self, X: np.ndarray) -> np.ndarray:
        """Calculate DModX statistics for input data.

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

        X_transformed = self.model_.transform(X)
        X_reconstructed = self.model_.inverse_transform(X_transformed)
        squared_errors = np.sum((X - X_reconstructed) ** 2, axis=1)

        return np.sqrt(squared_errors / (self.n_features_in_ - self.n_components_))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Identify outliers in the input data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data

        Returns
        -------
        ndarray of shape (n_samples,)
            Boolean array indicating outliers
        """
        dmodx_values = self.predict_residuals(X)
        return np.where(dmodx_values > self.critical_value_, -1, 1)

    def _calculate_critical_value(self) -> float:
        """Calculate F-distribution based critical value.

        Returns
        -------
        float
            The critical value for outlier detection
        """

        dof_numerator = self.n_features_in_ - self.n_components_
        dof_denominator = self.n_features_in_ - self.n_components_ - 1

        upper_control_limit = f_distribution.ppf(
            self.confidence, dof_numerator, dof_denominator
        )
        return np.sqrt(upper_control_limit)
