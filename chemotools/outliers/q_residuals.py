from typing import Optional, Literal, Union

import numpy as np

from scipy.stats import norm, chi2
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import validate_data, check_is_fitted

from ._base import _ModelResidualsBase, ModelTypes


class QResiduals(_ModelResidualsBase):
    """
    Calculate Q residuals (Squared Prediction Error - SPE) for PCA or PLS models.

    Parameters
    ----------
    model : Union[ModelType, Pipeline]
        A fitted PCA/PLS model or Pipeline ending with such a model.

    confidence : float, default=0.95
        Confidence level for statistical calculations (between 0 and 1).

    method : str, default="chi-square"
        The method used to compute the confidence threshold for Q residuals.
        Options:
        - "chi-square" : Uses mean and standard deviation to approximate Q residuals threshold.
        - "jackson-mudholkar" : Uses eigenvalue-based analytical approximation.
        - "percentile" : Uses empirical percentile threshold.

    Attributes
    ----------
    model_ : ModelType
        The fitted model of type _BasePCA or _PLS.

    preprocessing_ : Optional[Pipeline]
        Preprocessing steps before the model.

    n_features_in_ : int
        Number of features in the input data.

    n_components_ : int
        Number of components in the model.

    n_samples_ : int
        Number of samples used to train the model.

    critical_value_ : float
        The calculated critical value for outlier detection.

    References
    ----------
    Johan A. Westerhuis, Stephen P. Gurden, Age K. Smilde (2001) Generalized contribution plots in multivariate statistical process
    monitoring  Chemometrics and Intelligent Laboratory Systems 51 2000 95–114
    """

    def __init__(
        self,
        model: Union[ModelTypes, Pipeline],
        confidence: float = 0.95,
        method: Literal["chi-square", "jackson-mudholkar", "percentile"] = "percentile",
    ) -> None:
        self.method = method
        super().__init__(model, confidence)

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "QResiduals":
        """
        Fit the Q Residuals model by computing residuals from the training set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        Returns
        -------
        self : object
            Fitted instance of QResiduals.
        """
        X = validate_data(self, X, ensure_2d=True, dtype=np.float64)

        if self.preprocessing_:
            X = self.preprocessing_.fit_transform(X)

        # Compute the critical threshold using the chosen method
        self.critical_value_ = self._calculate_critical_value(X)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Identify outliers in the input data based on Q residuals threshold.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        ndarray of shape (n_samples,)
            Boolean array indicating outliers (-1 for outliers, 1 for normal data).
        """
        # Check the estimator has been fitted
        check_is_fitted(self, ["critical_value_"])

        # Validate the input data
        X = validate_data(
            self, X, y="no_validation", ensure_2d=True, reset=True, dtype=np.float64
        )

        # Calculate outliers based on the Q residuals
        Q_residuals = self.predict_residuals(X, validate=False)
        return np.where(Q_residuals > self.critical_value_, -1, 1)

    def predict_residuals(
        self, X: np.ndarray, y: Optional[np.ndarray] = None, validate: bool = True
    ) -> np.ndarray:
        """Calculate Q residuals (Squared Prediction Error - SPE) for input data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        validate : bool, default=True
            Whether to validate the input data.

        Returns
        -------
        ndarray of shape (n_samples,)
            Q residuals for each sample.
        """
        # Check the estimator has been fitted
        check_is_fitted(self, ["critical_value_"])

        # Validate the input data
        if validate:
            X = validate_data(self, X, ensure_2d=True, dtype=np.float64)

        # Apply preprocessing if available
        if self.preprocessing_:
            X = self.preprocessing_.transform(X)

        # Compute reconstruction error (Q residuals)
        X_transformed = self.model_.transform(X)
        X_reconstructed = self.model_.inverse_transform(X_transformed)
        Q_residuals = np.sum((X - X_reconstructed) ** 2, axis=1)

        return Q_residuals

    def _calculate_critical_value(
        self,
        X: Optional[np.ndarray] = None,
    ) -> float:
        """Calculate the critical value for outlier detection.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        X_reconstructed : array-like of shape (n_samples, n_features)
            Reconstructed input data.

        method : str Literal["chi-square", "jackson-mudholkar", "percentile"]
            The method used to compute the confidence threshold for Q residuals.

        Returns
        -------
        float
            The calculated critical value for outlier detection.

        """
        # Compute Q residuals for training data
        X_transformed = self.model_.transform(X)
        X_reconstructed = self.model_.inverse_transform(X_transformed)
        residuals = X - X_reconstructed

        if self.method == "chi-square":
            return self._chi_square_threshold(residuals)
        elif self.method == "jackson-mudholkar":
            return self._jackson_mudholkar_threshold(residuals)
        elif self.method == "percentile":
            Q_residuals = np.sum((residuals) ** 2, axis=1)
            return self._percentile_threshold(Q_residuals)
        else:
            raise ValueError(
                "Invalid method. Choose from 'chi-square', 'jackson-mudholkar', or 'percentile'."
            )

    def _chi_square_threshold(self, residuals: np.ndarray) -> float:
        """Compute Q residual threshold using Chi-Square Approximation."""
        eigenvalues = np.linalg.trace(np.cov(residuals.T))

        theta_1 = np.sum(eigenvalues)
        theta_2 = np.sum(eigenvalues**2)
        # Degrees of freedom approximation
        g = theta_2 / theta_1
        h = (2 * theta_1**2) / theta_2

        # Compute chi-square critical value at given confidence level
        chi_critical = chi2.ppf(self.confidence, df=h)

        # Compute final Q residual threshold
        return g * chi_critical

    def _jackson_mudholkar_threshold(self, residuals: np.ndarray) -> float:
        """Compute Q residual threshold using Jackson & Mudholkar’s analytical method."""

        eigenvalues = np.linalg.trace(np.cov(residuals.T))
        theta_1 = np.sum(eigenvalues)
        theta_2 = np.sum(eigenvalues**2)
        theta_3 = np.sum(eigenvalues**3)
        z_alpha = norm.ppf(self.confidence)

        h0 = 1 - (2 * theta_1 * theta_3) / (3 * theta_2**2)

        term1 = theta_2 * h0 * (1 - h0) / theta_1**2
        term2 = np.sqrt(z_alpha * 2 * theta_2 * h0**2) / theta_1

        return theta_1 * (1 - term1 + term2) ** (1 / h0)

    def _percentile_threshold(self, Q_residuals: np.ndarray) -> float:
        """Compute Q residual threshold using the empirical percentile method."""
        return np.percentile(Q_residuals, self.confidence * 100)
