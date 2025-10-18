"""
The :mod:`chemotools.baseline._linear_correction` module implements
a linear baseline correction transformer.
"""

# Author: Pau Cabaneros
# License: MIT

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, OneToOneFeatureMixin
from sklearn.utils.validation import check_is_fitted, validate_data


class LinearCorrection(TransformerMixin, OneToOneFeatureMixin, BaseEstimator):
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
        for i, x in enumerate(X_):
            X_[i, :] = self._drift_correct_spectrum(x)
        return X_.reshape(-1, 1) if X_.ndim == 1 else X_

    def _drift_correct_spectrum(self, x: np.ndarray) -> np.ndarray:
        # Can take any array and returns with a linear baseline correction
        # Find the x values at the edges of the spectrum
        y1: float = x[0]
        y2: float = x[-1]

        # Find the max and min wavenumebrs
        x1 = 0
        x2 = len(x)
        x_range = np.linspace(x1, x2, x2)

        # Calculate the straight line initial and end point
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1
        drift_correction = slope * x_range + intercept

        # Return the drift corrected spectrum
        return x - drift_correction
