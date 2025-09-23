"""
The :mod:`chemotools.outliers._hotelling_t2` module implements Hotelling's T-squared
outlier detection algorithm.
"""

# Authors: Pau Cabaneros
# License: MIT

from typing import Optional, Union
import numpy as np

from sklearn.cross_decomposition._pls import _PLS
from sklearn.decomposition._base import _BasePCA
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import validate_data, check_is_fitted
from sklearn.utils._param_validation import Interval, Real

from scipy.stats import f as f_distribution

from ._base import _ModelResidualsBase, ModelTypes


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

    References
    ----------
    [1] Johan A. Westerhuis, Stephen P. Gurden, Age K. Smilde
        Generalized contribution plots in multivariate statistical process
        monitoring  Chemometrics and Intelligent Laboratory Systems 51 2000 95–114 (2001).

    Examples
    --------
    >>> from chemotools.datasets import load_fermentation_train
    >>> from chemotools.outliers import HotellingT2
    >>> from sklearn.decomposition import PCA
    >>> # Load sample data
    >>> X, _ = load_fermentation_train()
    >>> # Instantiate the PCA model
    >>> pca = PCA(n_components=3).fit(X)
    >>> # Initialize HotellingT2 with the fitted PCA model
    >>> hotelling_t2 = HotellingT2(model=pca, confidence=0.95)
    HotellingT2(model=PCA(n_components=3), confidence=0.95)
    >>> hotelling_t2.fit(X)
    >>> # Predict outliers in the dataset
    >>> outliers = hotelling_t2.predict(X)
    >>> # Calculate Hotelling's T-squared statistics
    >>> t2_stats = hotelling_t2.predict_residuals(X)
    """

    _parameter_constraints = {
        "model": [Pipeline, ModelTypes],
        "confidence": [Interval(Real, 0, 1, closed="both")],
    }

    def __init__(
        self, model: Union[ModelTypes, Pipeline], confidence: float = 0.95
    ) -> None:
        self.model, self.confidence = model, confidence
        super().__init__(model, confidence)

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "HotellingT2":
        """
        Fit the model to the input data.

        This step calculates the critical value for the outlier detection.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data
        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : HotellingT2
            Fitted estimator with the critical threshold computed
        """
        X = validate_data(
            self, X, y="no_validation", ensure_2d=True, reset=True, dtype=np.float64
        )

        self.critical_value_ = self._calculate_critical_value()
        return self

    def predict(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """Identify outliers in the input data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data

        y : None
            Ignored to align with API.

        Returns
        -------
        ndarray of shape (n_samples,)
            Boolean array indicating outliers (-1) and inliers (1)
        """
        return super().predict(X, y)

    def predict_residuals(
        self, X: np.ndarray, y: Optional[np.ndarray] = None, validate: bool = True
    ) -> np.ndarray:
        """Calculate Hotelling's T-squared statistics for input data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data

        y : None
            Ignored.

        Returns
        -------
        ndarray of shape (n_samples,)
            Hotelling's T-squared statistics for each sample
        """
        # Check the estimator has been fitted
        check_is_fitted(self, ["critical_value_"])

        # Validate the input data
        if validate:
            X = validate_data(
                self, X, y="no_validation", ensure_2d=True, reset=True, dtype=np.float64
            )

        # Apply preprocessing steps
        if self.transformer_:
            X = self.transformer_.transform(X)

        # Calculate the Hotelling's T-squared statistics
        if isinstance(self.estimator_, _BasePCA):
            # For PCA-like models
            variances = self.estimator_.explained_variance_

        if isinstance(self.estimator_, _PLS):
            # For PLS-like models
            variances = np.var(self.estimator_.x_scores_, axis=0)

        # Equivalent to X @ model.components_.T for _BasePCA and X @ model.x_rotations_ for _PLS
        X_transformed = self.estimator_.transform(X)

        return np.sum((X_transformed**2) / variances, axis=1)

    def _calculate_critical_value(self, X: Optional[np.ndarray] = None) -> float:
        """
        Calculate the critical value for the Hotelling's T-squared statistics.

        Parameters
        ----------
        X : Ignored
            Not used, present for API consistency by convention.

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
