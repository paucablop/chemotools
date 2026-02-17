"""
The :mod:`chemotools.outliers._q_residuals` module implements the Q Residuals
(Squared Prediction Error - SPE) outlier detection algorithm.
"""

# Authors: Pau Cabaneros
# License: MIT

from typing import Literal, Optional, Union

import numpy as np
from scipy.stats import chi2, norm
from sklearn.pipeline import Pipeline
from sklearn.utils._param_validation import StrOptions

from ._base import _ModelResidualsBase, ModelTypes
from ._utils import calculate_residual_spectrum


class QResiduals(_ModelResidualsBase):
    """
    Calculate Q residuals (Squared Prediction Error - SPE) for PCA or PLS models.

    Parameters
    ----------
    model : Union[ModelType, Pipeline]
        A fitted PCA/PLS model or Pipeline ending with such a model.

    confidence : float, default=0.95
        Confidence level for statistical calculations (between 0 and 1).

    method : str, default="jackson-mudholkar"
        The method used to compute the confidence threshold for Q residuals.
        Options:
        - "chi-square" : Uses mean and standard deviation to approximate Q residuals threshold.
        - "jackson-mudholkar" : Uses eigenvalue-based analytical approximation.
        - "percentile" : Uses empirical percentile threshold.

    Attributes
    ----------
    estimator_ : ModelType
        The fitted model of type _BasePCA or _PLS.

    transformer_ : Optional[Pipeline]
        Preprocessing steps before the model.

    n_features_in_ : int
        Number of features in the input data.

    n_components_ : int
        Number of components in the model.

    n_samples_ : int
        Number of samples used to train the model.

    critical_value_ : float
        The calculated critical value for outlier detection.

    Methods
    -------
    fit(X, y=None)
        Fit the Q Residuals model by computing residuals from the training set.
        Calculates the critical threshold based on the chosen method.

    predict(X)
        Identify outliers in the input data based on Q residuals threshold.

    predict_residuals(X, y=None, validate=True)
        Calculate Q residuals (Squared Prediction Error - SPE) for input data.

    _calculate_critical_value(X)
        Calculate the critical value for outlier detection using the specified method.

    References
    ----------
    [1] Johan A. Westerhuis, Stephen P. Gurden, Age K. Smilde (2001)
        Generalized contribution plots in multivariate statistical process
        monitoring  Chemometrics and Intelligent Laboratory Systems 51 95–114 (2000)

    Examples
    --------
    >>> from chemotools.datasets import load_fermentation_train
    >>> from chemotools.outliers import QResiduals
    >>> from sklearn.decomposition import PCA
    >>> X, _ = load_fermentation_train()
    >>> pca = PCA(n_components=3).fit(X)
    >>> # Initialize QResiduals with the fitted PCA model
    >>> q_residuals = QResiduals(model=pca, confidence=0.95)
    >>> q_residuals.fit(X)
    >>> # Predict outliers in the dataset
    >>> outliers = q_residuals.predict(X)
    >>> # Calculate Q-residuals
    >>> q_residuals_stats = q_residuals.predict_residuals(X)
    """

    _parameter_constraints: dict = {
        **_ModelResidualsBase._parameter_constraints,
        "method": [StrOptions({"chi-square", "jackson-mudholkar", "percentile"})],
    }

    def __init__(
        self,
        model: Union[ModelTypes, Pipeline],
        confidence: float = 0.95,
        method: Literal[
            "chi-square", "jackson-mudholkar", "percentile"
        ] = "jackson-mudholkar",
    ) -> None:
        super().__init__(model, confidence)
        self.method = method

    def _fit_residuals(self, X: np.ndarray, y: Optional[np.ndarray]) -> None:
        """Compute Q residuals from training set and calculate the critical threshold."""
        residuals = calculate_residual_spectrum(X, self.estimator_)

        if self.method == "chi-square":
            self.critical_value_ = self._chi_square_threshold(residuals)
        elif self.method == "jackson-mudholkar":
            self.critical_value_ = self._jackson_mudholkar_threshold(residuals)
        elif self.method == "percentile":
            Q_residuals = np.sum(residuals**2, axis=1)
            self.critical_value_ = self._percentile_threshold(Q_residuals)

    def _compute_residuals(self, X: np.ndarray, y: Optional[np.ndarray]) -> np.ndarray:
        """Calculate Q residuals (Squared Prediction Error - SPE) for input data."""
        residual = calculate_residual_spectrum(X, self.estimator_)
        return np.sum(residual**2, axis=1)

    def _chi_square_threshold(self, residuals: np.ndarray) -> float:
        """Compute Q residual threshold using Chi-Square Approximation."""
        eigenvalues = np.linalg.trace(np.cov(residuals.T))

        theta_1 = np.sum(eigenvalues)
        theta_2 = np.sum(eigenvalues**2)
        # Degrees of freedom approximation
        g = theta_2 / theta_1
        h = (2 * theta_1**2) / theta_2

        # Compute chi-square critical value at given confidence level
        chi_critical = chi2.ppf(self.confidence_, df=h)

        # Compute final Q residual threshold
        return g * chi_critical

    def _jackson_mudholkar_threshold(self, residuals: np.ndarray) -> float:
        """Compute Q residual threshold using Jackson & Mudholkar’s analytical method."""

        eigenvalues = np.linalg.trace(np.cov(residuals.T))
        theta_1 = np.sum(eigenvalues)
        theta_2 = np.sum(eigenvalues**2)
        theta_3 = np.sum(eigenvalues**3)
        z_alpha = norm.ppf(self.confidence_)

        h0 = 1 - (2 * theta_1 * theta_3) / (3 * theta_2**2)

        term1 = theta_2 * h0 * (1 - h0) / theta_1**2
        term2 = np.sqrt(z_alpha * 2 * theta_2 * h0**2) / theta_1

        return theta_1 * (1 - term1 + term2) ** (1 / h0)

    def _percentile_threshold(self, Q_residuals: np.ndarray) -> float:
        """Compute Q residual threshold using the empirical percentile method."""
        return np.percentile(Q_residuals, self.confidence_ * 100)
