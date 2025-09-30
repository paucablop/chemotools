"""
The :mod:`chemotools.outliers._dmodx` module implements the Distance to Model (DModX) outlier detection algorithm.
"""

# Authors: Pau Cabaneros
# License: MIT

from typing import Optional, Union
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.utils.validation import validate_data, check_is_fitted
from sklearn.utils._param_validation import Interval, Real
from scipy.stats import f as f_distribution


from ._base import _ModelResidualsBase, ModelTypes
from ._utils import calculate_residual_spectrum


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

    mean_centered : bool, default=True
        Indicates if the input data was mean-centered before modeling

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

    train_sse_: float
        The training sum of squared errors (SSE) for the model normalized by degrees of freedom

    A0_ : int
        Adjustment factor for degrees of freedom based on mean centering

    References
    ----------
    [1] Max Bylesjö, Mattias Rantalainen, Oliver Cloarec, Johan K. Nicholson,
        Elaine Holmes, Johan Trygg.
        "OPLS discriminant analysis: combining the strengths of PLS-DA and SIMCA
        classification." Journal of Chemometrics 20 (8-10), 341-351 (2006).

    Examples
    --------
    >>> from chemotools.datasets import load_fermentation_train
    >>> from chemotools.outliers import DModX
    >>> from sklearn.decomposition import PCA
    >>> # Load sample data
    >>> X, _ = load_fermentation_train()
    >>> # Instantiate the PCA model
    >>> pca = PCA(n_components=3).fit(X)
    >>> # Initialize DModX with the fitted PCA model
    >>> dmodx = DModX(model=pca, confidence=0.95, mean_centered=True)
    DModX(model=PCA(n_components=3), confidence=0.95, mean_centered=True)
    >>> dmodx.fit(X)
    >>> # Predict outliers in the dataset
    >>> outliers = dmodx.predict(X)
    >>> # Calculate DModX residuals
    >>> residuals = dmodx.predict_residuals(X)
    """

    _parameter_constraints: dict = {
        "model": [Pipeline, ModelTypes],
        "confidence": [Interval(Real, 0, 1, closed="both")],
        "mean_centered": [bool],
    }

    def __init__(
        self,
        model: Union[ModelTypes, Pipeline],
        confidence: float = 0.95,
        mean_centered: bool = True,
    ) -> None:
        super().__init__(model, confidence)
        self.mean_centered = mean_centered

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "DModX":
        """
        Fit the model and compute training residual variance.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            The input data used to fit the model.

        y : None
            Ignored to align with API.

        Returns
        -------
        self : DModX
            Fitted estimator with computed training residuals and critical value.
        """
        X_validated = validate_data(
            self, X, y="no_validation", ensure_2d=True, reset=True, dtype=np.float64
        )

        # Process data through transformer if available
        X_processed = (
            self.transformer_.transform(X_validated)
            if self.transformer_
            else X_validated
        )

        # Calculate residuals for the training set
        residuals = calculate_residual_spectrum(X_processed, self.estimator_)

        # Sum of squared residuals for the training set
        self.train_sse_ = np.sum(residuals**2)

        # Set degrees of freedom depending on mean centering
        self.A0_ = 1 if self.mean_centered else 0

        # Compute the critical value
        self.critical_value_ = self._calculate_critical_value()

        return self

    def predict(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Identify outliers in the input data.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            The input data to predict outliers for.

        y : None
            Ignored to align with API.

        Returns
        -------
        outliers : np.ndarray of shape (n_samples,)
            Array indicating outliers (-1) and inliers (1).
        """
        return super().predict(X, y)

    def predict_residuals(
        self, X: np.ndarray, y: Optional[np.ndarray] = None, validate: bool = True
    ) -> np.ndarray:
        """
        Calculate normalized DModX statistics for input data.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            The input data to calculate DModX statistics for.

        y : None
            Ignored.

        validate : bool, default=True
            If True, validate the input data.

        Returns
        -------
        dmodx_values : np.ndarray of shape (n_samples,)
            The normalized DModX statistics for each sample.
        """
        # Ensure the model is fitted
        check_is_fitted(self, ["critical_value_"])

        # Validate input data if required
        if validate:
            X = validate_data(
                self, X, y="no_validation", ensure_2d=True, reset=True, dtype=np.float64
            )

        # Process data through transformer if available
        X_processed = self.transformer_.transform(X) if self.transformer_ else X

        # Calculate residuals for the input data
        residuals = calculate_residual_spectrum(X_processed, self.estimator_)
        sample_sse = np.sum(residuals**2, axis=1)

        # Normalize residuals per dimension
        residual_norm = np.sqrt(sample_sse / (self.n_features_in_ - self.n_components_))

        # Scale factor based on training set residuals
        training_residual_scale = np.sqrt(
            self.train_sse_
            / (
                (self.n_samples_ - self.n_components_ - self.A0_)
                * (self.n_features_in_ - self.n_components_)
            )
        )

        return residual_norm / training_residual_scale

    def _calculate_critical_value(self, X: Optional[np.ndarray] = None) -> float:
        """Calculate F-distribution based critical value."""
        dof_num = self.n_features_in_ - self.n_components_
        dof_den = self.n_samples_ - self.n_components_ - self.A0_

        f_quantile = f_distribution.ppf(self.confidence, dof_num, dof_den)
        return np.sqrt(f_quantile)
