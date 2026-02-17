"""
The :mod:`chemotools.outliers._hotelling_t2` module implements Hotelling's T-squared
outlier detection algorithm.
"""

# Authors: Pau Cabaneros
# License: MIT

from typing import Optional, Union

import numpy as np
from scipy.stats import f as f_distribution
from sklearn.cross_decomposition._pls import _PLS
from sklearn.decomposition._base import _BasePCA
from sklearn.pipeline import Pipeline

from ._base import ModelTypes, _ModelResidualsBase


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
        monitoring  Chemometrics and Intelligent Laboratory
        Systems 51 2000 95–114 (2001).

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

    def __init__(
        self, model: Union[ModelTypes, Pipeline], confidence: float = 0.95
    ) -> None:
        super().__init__(model, confidence)

    def _fit_residuals(self, X: np.ndarray, y: Optional[np.ndarray]) -> None:
        """Calculate the critical value for Hotelling's T-squared statistics."""
        critical_value = f_distribution.ppf(
            self.confidence_, self.n_components_, self.n_samples_ - self.n_components_
        )
        self.critical_value_ = (
            critical_value
            * self.n_components_
            * (self.n_samples_ - 1)
            / (self.n_samples_ - self.n_components_)
        )

    def _compute_residuals(self, X: np.ndarray, y: Optional[np.ndarray]) -> np.ndarray:
        """Calculate Hotelling's T-squared statistics for input data."""
        if isinstance(self.estimator_, _BasePCA):
            variances = self.estimator_.explained_variance_  # type: ignore[unresolved-attribute]

        if isinstance(self.estimator_, _PLS):
            variances = np.var(self.estimator_.x_scores_, axis=0)  # type: ignore[unresolved-attribute]

        X_transformed = self.estimator_.transform(X)
        return np.sum((X_transformed**2) / variances, axis=1)
