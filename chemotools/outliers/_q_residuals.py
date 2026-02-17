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

from ._base import ModelTypes, _ModelResidualsBase
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
        - "chi-square" : Uses the first two moments of the residual
        eigenvalues (mean and variance) to compute a moment-matched
        chi-square threshold for Q residuals [1, 3].
        - "jackson-mudholkar" : Uses the first three moments of the
        residual eigenvalues to calculate an analytical threshold
        based on Jackson & Mudholkar's approximation [2, 3].
        - "percentile" : Uses the empirical percentile of the
        observed Q residuals to set a non-parametric threshold.

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
    [1] Box, G. E. P. (1954).
        Some theorems on quadratic forms applied in the study
        of analysis of variance problems, I.
        Effect of inequality of variance in the one-way
        classification.
        Annals of Mathematical Statistics, 25(2), 290–302.
    [2] Jackson, J. E., & Mudholkar, G. S. (1979).
        Control procedures for residuals associated with principal component analysis.
        Technometrics, 21(3), 341–349.
    [3] Johan A. Westerhuis, Stephen P. Gurden, Age K. Smilde (2001)
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
        """Compute SPE thresholds using proper eigenvalue decomposition."""
        residuals = calculate_residual_spectrum(X, self.estimator_)

        # Calculate Q residuals for the training set: sum of squared errors per row
        q_values = np.sum(residuals**2, axis=1)

        if self.method == "percentile":
            self.critical_value_ = np.percentile(q_values, self.confidence * 100)
            return

        # For statistical methods, we need the eigenvalues of the
        # residual covariance matrix.
        # This represents the variance remaining in each 'unused'
        # dimension
        theta1, theta2, theta3 = self._calculate_thetas(residuals)

        if self.method == "chi-square":
            self.critical_value_ = self._chi_square_threshold(theta1, theta2)

        elif self.method == "jackson-mudholkar":
            self.critical_value_ = self._jackson_mudholkar_threshold(
                theta1, theta2, theta3
            )

    def _compute_residuals(self, X: np.ndarray, y: Optional[np.ndarray]) -> np.ndarray:
        """Calculate Q residuals (SPE) for input data."""
        residual_matrix = calculate_residual_spectrum(X, self.estimator_)
        return np.sum(residual_matrix**2, axis=1)

    def _calculate_thetas(self, residuals: np.ndarray):
        """Compute the three moments of the residual eigenvalue distribution."""
        # Note: We use eigvalsh for symmetric matrices (covariance)
        cov_matrix = np.cov(residuals.T)
        lambdas = np.linalg.eigvalsh(cov_matrix)

        # Filter noise: only keep positive eigenvalues
        lambdas = lambdas[lambdas > 1e-12]

        theta1 = np.sum(lambdas)
        theta2 = np.sum(lambdas**2)
        theta3 = np.sum(lambdas**3)
        return theta1, theta2, theta3

    def _chi_square_threshold(self, t1: float, t2: float) -> float:
        """Box approximation: g * chi2(h)."""
        g = t2 / t1
        h = (t1**2) / t2
        return g * chi2.ppf(self.confidence, df=h)

    def _jackson_mudholkar_threshold(self, t1: float, t2: float, t3: float) -> float:
        """Standard Jackson-Mudholkar analytical threshold."""
        z_alpha = norm.ppf(self.confidence)
        h0 = 1 - (2 * t1 * t3) / (3 * t2**2)

        # Corrected Formula: z_alpha is outside the square root
        # If h0 is very close to 0, the SPE distribution is approximately log-normal
        if abs(h0) < 1e-6:
            return t1 * np.exp(z_alpha * np.sqrt(2 * t2) / t1)

        term1 = (z_alpha * np.sqrt(2 * t2 * h0**2)) / t1
        term2 = (t2 * h0 * (h0 - 1)) / (t1**2)

        return t1 * (1 + term1 + term2) ** (1 / h0)

    def _percentile_threshold(self, Q_residuals: np.ndarray) -> float:
        """Compute Q residual threshold using the empirical percentile method."""
        return np.percentile(Q_residuals, self.confidence_ * 100)
