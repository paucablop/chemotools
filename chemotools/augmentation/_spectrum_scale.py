"""
The :mod:`chemotools.augmentation._spectrum_scale` module implements the SpectrumScale
transformer to scale spectral data by a random factor.
"""

# Authors: Pau Cabaneros
# License: MIT

from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, OneToOneFeatureMixin
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted, validate_data
from sklearn.utils._param_validation import Interval, Real


class SpectrumScale(TransformerMixin, OneToOneFeatureMixin, BaseEstimator):
    """
    Scales the data by a value drawn from the uniform distribution centered
    around 1.0.

    Parameters
    ----------
    scale : float, default=0.0
        Range of the uniform distribution to draw the scaling factor from.

    random_state : int, default=None
        The random state to use for the random number generator.

    Examples
    --------
    >>> from chemotools.augmentation import SpectrumScale
    >>> from chemotools.datasets import load_fermentation_train
    >>> # Load sample data
    >>> X, _ = load_fermentation_train()
    >>> # Instantiate the transformer
    >>> transformer = SpectrumScale(scale=0.1)
    SpectrumScale()
    >>> transformer.fit(X)
    >>> # Generate scaled data
    >>> X_scaled = transformer.transform(X)
    """

    _parameter_constraints: dict = {
        "scale": [Interval(Real, 0, None, closed="both")],
        "random_state": [None, int, np.random.RandomState],
    }

    def __init__(self, scale: float = 0.0, random_state: Optional[int] = None):
        self.scale = scale
        self.random_state = random_state

    def fit(self, X: np.ndarray, y=None) -> "SpectrumScale":
        """
        Fit the transformer to the input data.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            The input data to fit the transformer to.

        y : None
            Ignored.

        Returns
        -------
        self : SpectrumScale
            The fitted transformer.
        """
        # Check that X is a 2D array and has only finite values
        X = validate_data(
            self, X, y="no_validation", ensure_2d=True, reset=True, dtype=np.float64
        )

        # Instantiate the random number generator
        self._rng = check_random_state(self.random_state)

        return self

    def transform(self, X: np.ndarray, y=None) -> np.ndarray:
        """
        Transform the input data by scaling the spectrum.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            The input data to transform.

        y : None
            Ignored.

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

        # Calculate the scaled spectrum
        for i, x in enumerate(X_):
            X_[i] = self._scale_spectrum(x)

        return X_.reshape(-1, 1) if X_.ndim == 1 else X_

    def _scale_spectrum(self, x) -> np.ndarray:
        scaling_factor = self._rng.uniform(low=1 - self.scale, high=1 + self.scale)
        return np.multiply(x, scaling_factor)
