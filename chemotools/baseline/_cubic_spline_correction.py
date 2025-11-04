"""
The :mod:`chemotools.baseline._cubic_spline_correction` module implements
a cubic spline baseline correction transformer.
"""

# Author: Pau Cabaneros
# License: MIT

from typing import Optional

import numpy as np
from scipy.interpolate import CubicSpline
from sklearn.base import BaseEstimator, TransformerMixin, OneToOneFeatureMixin
from sklearn.utils.validation import check_is_fitted, validate_data


class CubicSplineCorrection(TransformerMixin, OneToOneFeatureMixin, BaseEstimator):
    """
    A transformer that corrects a baseline by subtracting a cubic spline through the
    points defined by the indices.

    Parameters
    ----------
    indices : list, optional, default=None
        The indices of the features to use for the baseline correction. If None,
        the first and last indices will be used.

    Attributes
    ----------
    n_features_in_ : int
        The number of features in the input data.

    indices_ : list
        The indices of the features used for the baseline correction.

    Examples
    --------
    >>> from chemotools.baseline import CubicSplineCorrection
    >>> from chemotools.datasets import load_fermentation_train
    >>> # Load sample data
    >>> X, _ = load_fermentation_train()
    >>> # Instantiate the transformer
    >>> transformer = CubicSplineCorrection(indices=[0, 100, 200, 300, 400, 500])
    CubicSplineCorrection(indices)
    >>> transformer.fit(X)
    >>> # Generate baseline-corrected data
    >>> X_corrected = transformer.transform(X)
    """

    _parameter_constraints: dict = {
        "indices": ["array-like", None],
    }

    def __init__(self, indices: Optional[list] = None) -> None:
        self.indices = indices

    def fit(self, X: np.ndarray, y=None) -> "CubicSplineCorrection":
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
        self : ConstantBaselineCorrection
            The fitted transformer.
        """
        # Check that X is a 2D array and has only finite values
        X = validate_data(
            self, X, y="no_validation", ensure_2d=True, reset=True, dtype=np.float64
        )

        if self.indices is None:
            self.indices_ = [0, len(X[0]) - 1]
        else:
            self.indices_ = self.indices

        return self

    def transform(self, X: np.ndarray, y=None) -> np.ndarray:
        """
        Transform the input data by subtracting the baseline.

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
        check_is_fitted(self, "indices_")

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

        # Calculate spline baseline correction
        for i, x in enumerate(X_):
            X_[i] = self._spline_baseline_correct(x)
        return X_.reshape(-1, 1) if X_.ndim == 1 else X_

    def _spline_baseline_correct(self, x: np.ndarray) -> np.ndarray:
        indices = self.indices_
        intensity = x[indices]
        spl = CubicSpline(indices, intensity)
        baseline = spl(range(len(x)))
        return x - baseline
