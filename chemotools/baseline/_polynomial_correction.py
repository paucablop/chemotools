"""
The :mod:`chemotools.baseline._polynomial_correction` module implements
a polynomial baseline correction transformer.
"""

# Author: Pau Cabaneros
# License: MIT

from typing import Optional
from numbers import Integral

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, OneToOneFeatureMixin
from sklearn.utils.validation import check_is_fitted, validate_data
from sklearn.utils._param_validation import Interval


class PolynomialCorrection(TransformerMixin, OneToOneFeatureMixin, BaseEstimator):
    """
    A transformer that subtracts a polynomial baseline from the input data. The polynomial is
    fitted to the points in the spectrum specified by the indices parameter.

    Parameters
    ----------
    order : int, optional, default=1
        The order of the polynomial to fit to the baseline. Defaults to 1.

    indices : list, optional, default=None
        The indices of the points in the spectrum to fit the polynomial to. Defaults to None,
        which fits the polynomial to all points in the spectrum (equivalent to detrend).

    Attributes
    ----------
    n_features_in_ : int
        The number of features in the input data.

    indices_ : list
        The indices of the points in the spectrum to fit the polynomial to.
        If indices is None, this will be a list of all indices in the spectrum.

    Examples
    --------
    >>> from chemotools.baseline import PolynomialCorrection
    >>> from chemotools.datasets import load_fermentation_train
    >>> # Load sample data
    >>> X, _ = load_fermentation_train()
    >>> # Instantiate the transformer
    >>> transformer = PolynomialCorrection(order=2, indices=[0, 100, 200, 300, 400, 500])
    PolynomialCorrection()
    >>> transformer.fit(X)
    >>> # Generate baseline-corrected data
    >>> X_corrected = transformer.transform(X)
    """

    _parameter_constraints: dict = {
        "order": [Interval(Integral, 0, None, closed="left")],
        "indices": ["array-like", None],
    }

    def __init__(self, order: int = 1, indices: Optional[list] = None) -> None:
        self.order = order
        self.indices = indices

    def fit(self, X: np.ndarray, y=None) -> "PolynomialCorrection":
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
        self : PolynomialCorrection
            The fitted transformer.
        """
        # Check that X is a 2D array and has only finite values
        X = validate_data(
            self, X, y="no_validation", ensure_2d=True, reset=True, dtype=np.float64
        )
        if self.indices is None:
            self.indices_ = list(range(0, len(X[0])))
        else:
            self.indices_ = self.indices

        return self

    def transform(self, X: np.ndarray, y=None) -> np.ndarray:
        """
        Transform the input data by subtracting the polynomial baseline.

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

        # Calculate polynomial baseline correction
        for i, x in enumerate(X_):
            X_[i] = self._baseline_correct_spectrum(x)
        return X_.reshape(-1, 1) if X_.ndim == 1 else X_

    def _baseline_correct_spectrum(self, x: np.ndarray) -> np.ndarray:
        """
        Subtract the polynomial baseline from a single spectrum.

        Parameters
        ----------
        x : np.ndarray
            The spectrum to correct.

        Returns
        -------
        x : np.ndarray
            The corrected spectrum.
        """
        intensity = x[self.indices_]
        poly = np.polyfit(self.indices_, intensity, self.order)
        baseline = [np.polyval(poly, i) for i in range(0, len(x))]
        return x - baseline
