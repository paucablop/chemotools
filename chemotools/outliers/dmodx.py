from typing import Optional, Union
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.utils.validation import validate_data, check_is_fitted
from scipy.stats import f as f_distribution


from ._base import _ModelResidualsBase, ModelTypes
from .utils import calculate_residual_spectrum


class DModX(_ModelResidualsBase):
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

    train_spe_: float
        The training sum of squared errors (SSE) for the model normalized by degrees of freedom
    """

    def __init__(
        self,
        model: Union[ModelTypes, Pipeline],
        confidence: float = 0.95,
    ) -> None:
        model, confidence = model, confidence
        super().__init__(model, confidence)

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "DModX":
        """
        Fit the model to the input data.

        This step calculates the critical value for the outlier detection. In the DmodX method,
        the critical value is not depend on the input data but on the model parameters.
        """
        X = validate_data(
            self, X, y="no_validation", ensure_2d=True, reset=True, dtype=np.float64
        )

        # Calculate the critical value
        self.critical_value_ = self._calculate_critical_value()

        # Calculate the degrees of freedom normalized SPE of the training set
        residuals = calculate_residual_spectrum(X, self.estimator_)
        squared_errors = np.sum((residuals) ** 2, axis=1)
        self.train_spe_ = np.sqrt(
            squared_errors
            / (self.n_samples_ - self.n_components_ - 1)
            * (self.n_features_in_ - self.n_components_)
        )

        return self

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
        # Check the estimator has been fitted
        check_is_fitted(self, ["critical_value_"])

        # Validate the input data
        X = validate_data(
            self, X, y="no_validation", ensure_2d=True, reset=True, dtype=np.float64
        )

        # Calculate outliers based on the DModX statistics
        dmodx_values = self.predict_residuals(X, validate=False)
        return np.where(dmodx_values > self.critical_value_, -1, 1)

    def predict_residuals(
        self, X: np.ndarray, y: Optional[np.ndarray] = None, validate: bool = True
    ) -> np.ndarray:
        """Calculate DModX statistics for input data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data

        validate : bool, default=True
            Whether to validate the input data

        Returns
        -------
        ndarray of shape (n_samples,)
            DModX statistics for each sample
        """
        # Check the estimator has been fitted
        check_is_fitted(self, ["critical_value_"])

        # Validate the input data
        if validate:
            X = validate_data(
                self, X, y="no_validation", ensure_2d=True, reset=True, dtype=np.float64
            )

        # Apply preprocessing if available
        if self.transformer_:
            X = self.transformer_.transform(X)

        # Calculate the DModX statistics
        residual = calculate_residual_spectrum(X, self.estimator_)
        squared_errors = np.sum((residual) ** 2, axis=1)

        return (
            np.sqrt(squared_errors / (self.n_features_in_ - self.n_components_))
            / self.train_spe_
        )

    def _calculate_critical_value(self, X: Optional[np.ndarray] = None) -> float:
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
