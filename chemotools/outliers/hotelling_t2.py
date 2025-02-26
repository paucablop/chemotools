from typing import Optional, Union
import numpy as np

from sklearn.cross_decomposition._pls import _PLS
from sklearn.decomposition._base import _BasePCA
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import validate_data, check_is_fitted
from scipy.stats import f as f_distribution


from ._base import _ModelResidualsBase
from ._utils import ModelTypes


class HotellingT2(_ModelResidualsBase):
    """
    Calculate Hotelling's T-squared statistics for PCA or PLS like models.

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

    References
    ----------
    Johan A. Westerhuis, Stephen P. Gurden, Age K. Smilde (2001) Generalized contribution plots in multivariate statistical process
    monitoring  Chemometrics and Intelligent Laboratory Systems 51 2000 95–114
    """

    def __init__(
        self, model: Union[ModelTypes, Pipeline], confidence: float = 0.95
    ) -> None:
        super().__init__(model, confidence)

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "HotellingT2":
        """
        Fit the model to the input data.

        This step calculates the critical value for the outlier detection. In the DmodX method,
        the critical value is not depend on the input data but on the model parameters.
        """
        X = validate_data(
            self, X, y="no_validation", ensure_2d=True, reset=True, dtype=np.float64
        )

        self.critical_value_ = self._calculate_critical_value()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray[bool]:
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

        # Calculate the Hotelling's T-squared statistics
        hotelling_t2_values = self.predict_residuals(X, validate=False)
        return np.where(hotelling_t2_values > self.critical_value_, -1, 1)


    def predict_residuals(self, X: np.ndarray, validate: bool = True) -> np.ndarray:
        """Calculate Hotelling's T-squared statistics for input data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data

        Returns
        -------
        ndarray of shape (n_samples,)
            Hotellin's T-squared statistics for each sample
        """
        # Check the estimator has been fitted
        check_is_fitted(self, ["critical_value_"])

        # Validate the input data
        if validate:
            X = validate_data(
                self, X, y="no_validation", ensure_2d=True, reset=True, dtype=np.float64
            )

        # Apply preprocessing steps
        if self.preprocessing_:
            X = self.preprocessing_.transform(X)

        # Calculate the Hotelling's T-squared statistics
        if isinstance(self.model_, _BasePCA):
            # For PCA-like models
            variances = self.model_.explained_variance_

        if isinstance(self.model_, _PLS):
            # For PLS-like models
            variances = np.var(self.model_.x_scores_, axis=0)

        # Equivalent to X @ model.components_.T for _BasePCA and X @ model.x_rotations_ for _PLS
        X_transformed = self.model_.transform(X)

        return np.sum((X_transformed**2) / variances, axis=1)


    def _calculate_critical_value(self):
        """
        Calculate the critical value for the Hotelling's T-squared statistics.
        
        Returns
        -------
        float
            The critical value for the Hotelling's T-squared statistics
        """

        critical_value = f_distribution.ppf(
            self.confidence, self.n_components_, self.n_samples_ - self.n_components_
        )
        return (
            critical_value
            * self.n_components_
            * (self.n_samples_ - 1)
            / (self.n_samples_ - self.n_components_)
        )
