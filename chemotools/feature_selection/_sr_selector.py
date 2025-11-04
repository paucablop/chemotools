"""
The :mod:`chemotools.feature_selection._sr_selector` module implements the Selectivity Ratio (SR)
feature selector for PLS regression models.
"""

# Author: Pau Cabaneros
# License: MIT

import numpy as np
from sklearn.utils.validation import validate_data
from sklearn.pipeline import Pipeline
from sklearn.utils._param_validation import Interval, Real

from ._base import _PLSFeatureSelectorBase, ModelTypes


class SRSelector(_PLSFeatureSelectorBase):
    """
    This selector is used to select features that contribute significantly
    to the latent variables in a PLS regression model using the Selectivity
    Ratio (SR) method.

    Parameters
    ----------
    model : Union[_PLS, Pipeline]
        The PLS regression model or a pipeline with a PLS regression model as last step.

    threshold : float, default=1.0
        The threshold for feature selection. Features with importance
        above this threshold will be selected.

    Attributes
    ----------
    estimator_ : ModelTypes
        The fitted model of type _BasePCA or _PLS

    feature_scores_ : np.ndarray
        The calculated feature scores based on the selected method.

    support_mask_ : np.ndarray
        The boolean mask indicating which features are selected.

    References
    ----------
    [1] Kim H. Esbensen,
        "Multivariate Data Analysis - In Practice", 5th Edition, 2002.

    Examples
    --------
    >>> from chemotools.datasets import load_fermentation_train
    >>> from chemotools.feature_selection import SRSelector
    >>> from sklearn.cross_decomposition import PLSRegression
    >>> # Load sample data
    >>> X, y = load_fermentation_train()
    >>> # Instantiate the PLS regression model
    >>> pls_model = PLSRegression(n_components=2).fit(X, y)
    >>> # Instantiate the SR selector with the PLS model
    >>> selector = SRSelector(model=pls_model, threshold=0.9)
    >>> selector.fit(X)
    SRSelector(model=PLSRegression(n_components=2), threshold=0.9)
    >>> # Get the selected features
    >>> X_selected = selector.transform(X)
    >>> X_selected.shape
    (21, 978)
    """

    _parameter_constraints: dict = {
        "model": [Pipeline, ModelTypes],
        "threshold": [Interval(Real, 0, None, closed="both")],
    }

    def __init__(
        self,
        model,
        threshold: float = 1.0,
    ):
        self.model = model
        self.threshold = threshold
        super().__init__(self.model)

    def fit(self, X: np.ndarray, y=None) -> "SRSelector":
        """
        Fit the transformer to calculate the feature scores and the support mask.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data to fit the transformer to.

        y : None
            Ignored to align with API.

        Returns
        -------
        self : SRSelector
            The fitted transformer.
        """
        # Check that X is a 2D array and has only finite values
        X = validate_data(
            self, X, y="no_validation", ensure_2d=True, reset=True, dtype=np.float64
        )

        # Calculate the SR scores
        self.feature_scores_ = self._calculate_features(X)

        # Calculate the support mask
        self.support_mask_ = self._get_support_mask()

        return self

    def _get_support_mask(self) -> np.ndarray:
        """
        Get the support mask based on the feature scores and threshold.
        Features with scores above the threshold are selected.
        Parameters
        ----------
        self : SRSelector
            The fitted transformer.

        Returns
        -------
        support_mask_ : np.ndarray
            The boolean mask indicating which features are selected.
        """
        return self.feature_scores_ > self.threshold

    def _calculate_features(self, X: np.ndarray) -> np.ndarray:
        """
        Vectorized Selectivity Ratio calculation from a fitted _PLS
        like model.

        Parameters:
        ----------
        - self: SRSelector
            The fitted transformer.

        - X: array-like of shape (n_samples, n_features)
            The input training data to calculate the feature scores from.

        Returns
        -------
        feature_scores_ : np.ndarray
            The calculated feature scores based on the selected method.
        """
        bpls = self.estimator_.coef_
        bpls_norm = bpls.T / np.linalg.norm(bpls)

        # Handle 1D case correctly
        if bpls.ndim == 1:
            bpls_norm = bpls_norm.reshape(-1, 1)

        # Project X onto the regression vector
        ttp = X @ bpls_norm
        ptp = X.T @ np.linalg.pinv(ttp).T

        # Predicted part of X
        X_hat = ttp @ ptp.T

        # Compute squared norms directly
        total_ss = np.linalg.norm(X, axis=0) ** 2
        explained_ss = np.linalg.norm(X_hat, axis=0) ** 2

        # Calculate residual sum of squares
        residual_ss = total_ss - explained_ss

        # Stability: avoid division by zero
        epsilon = 1e-12

        # Calculate Selectivity Ratio
        return explained_ss / (residual_ss + epsilon)
