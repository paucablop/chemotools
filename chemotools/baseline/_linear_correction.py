"""
The :mod:`chemotools.baseline._linear_correction` module implements
a linear baseline correction transformer.
"""

# Author: Pau Cabaneros
# License: MIT

import numpy as np
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin
from sklearn.utils.validation import check_is_fitted, validate_data

from chemotools._doc_mixin import DocLinkMixin


class LinearCorrection(
    DocLinkMixin, TransformerMixin, OneToOneFeatureMixin, BaseEstimator
):
    """
    A transformer that corrects a baseline by subtracting a linear baseline through the
    initial and final points of the spectrum.

    Attributes
    ----------
    n_features_in_ : int
        The number of features in the input data.

    Examples
    --------
    >>> from chemotools.baseline import LinearCorrection
    >>> from chemotools.datasets import load_fermentation_train
    >>> # Load sample data
    >>> X, _ = load_fermentation_train()
    >>> # Instantiate the transformer
    >>> transformer = LinearCorrection()
    LinearCorrection()
    >>> transformer.fit(X)
    >>> # Generate baseline-corrected data
    >>> X_corrected = transformer.transform(X)
    """

    _parameter_constraints: dict = {}

    def fit(self, X: np.ndarray, y=None) -> "LinearCorrection":
        """
        Fit the transformer to the input data.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            The input data to fit the transformer to.

        y : None
            Ignored to align with API.

        Returns
        -------
        self : LinearCorrection
            The fitted transformer.
        """
        # Validate the input parameters
        self._validate_params()

        # Check that X is a 2D array and has only finite values
        X = validate_data(
            self, X, y="no_validation", ensure_2d=True, reset=True, dtype=np.float64
        )

        return self

    def transform(self, X: np.ndarray, y=None) -> np.ndarray:
        """
        Transform the input data by subtracting the constant baseline value.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            The input data to transform.

        y : None
            Ignored to align with API.

        Returns
        -------
        X_transformed : np.ndarray of shape (n_samples, n_features)
            The transformed data.
        """
        # Check that the estimator is fitted
        check_is_fitted(self, "n_features_in_")

        # Check that X is a 2D array and has only finite values
        X_ = validate_data(
            self,
            X,
            y="no_validation",
            ensure_2d=True,
            copy=True,
            reset=False,
            dtype=np.float64,
        )

        # Calculate non-negative values
        x_range = np.arange(X_.shape[1])

        X0 = X_[:, 0]
        Xn = X_[:, -1]
        slope = (Xn - X0) / (x_range[-1] - x_range[0])
        intercept = X0 - slope * x_range[0]
        drift_correction = slope[:, np.newaxis] * x_range + intercept[:, np.newaxis]
        return X_ - drift_correction
