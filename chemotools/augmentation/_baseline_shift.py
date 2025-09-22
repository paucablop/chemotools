"""
The :mod:`chemotools.augmentation._baseline_shift` module implements the BaselineShift
transformer to add a constant baseline to the input data.
"""

# Authors: Pau Cabaneros
# License: MIT

from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, OneToOneFeatureMixin
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted, validate_data
from sklearn.utils._param_validation import Interval, Real


class BaselineShift(TransformerMixin, OneToOneFeatureMixin, BaseEstimator):
    """
    Adds a constant baseline to the data. The baseline is drawn from a one-sided
    uniform distribution between 0 and 0 + scale.

    Parameters
    ----------
    scale : float, default=0.0
        Range of the uniform distribution to draw the baseline factor from.

    random_state : int, default=None
        The random state to use for the random number generator.

    Attributes
    ----------
    n_features_in_ : int
        The number of features in the input data.

    Examples
    --------
    >>> from chemotools.augmentation import BaselineShift
    >>> from chemotools.datasets import load_fermentation_train
    >>> # Load sample data
    >>> X, _ = load_fermentation_train()
    >>> # Instantiate the transformer
    >>> transformer = BaselineShift(scale=0.1)
    BaselineShift()
    >>> transformer.fit(X)
    >>> # Generate baseline-shifted data
    >>> X_shifted = transformer.transform(X)
    """

    _parameter_constraints: dict = {
        "scale": [Interval(Real, 0, None, closed="both")],
        "random_state": [None, int, np.random.RandomState],
    }

    def __init__(self, scale: float = 0.0, random_state: Optional[int] = None):
        self.scale = scale
        self.random_state = random_state

    def fit(self, X: np.ndarray, y=None) -> "BaselineShift":
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
        self : BaselineShift
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
        Transform the input data by adding a baseline to the spectrum.

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
            X_[i] = self._add_baseline(x)

        return X_.reshape(-1, 1) if X_.ndim == 1 else X_

    def _add_baseline(self, x: np.ndarray) -> np.ndarray:
        adding_factor = self._rng.uniform(low=0, high=self.scale)
        return np.add(x, adding_factor)
