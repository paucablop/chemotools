"""
The :mod:`chemotools.scale._point_scaler` module implements a Point Scaler transformer.
"""

# Authors: Pau Cabaneros
# License: MIT

from typing import Optional
from numbers import Integral

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, OneToOneFeatureMixin
from sklearn.utils.validation import check_is_fitted, validate_data
from sklearn.utils._param_validation import Interval


class PointScaler(TransformerMixin, OneToOneFeatureMixin, BaseEstimator):
    """
    A transformer that scales the input data by the intensity value at a given point.
    The point can be specified by an index or by a wavenumber.

    Parameters
    ----------
    point : int, optional, default=0
        The point to scale the data by. It can be an index or a wavenumber.

    wavenumber : array-like, optional, default=None
        The wavenumbers of the input data. If not provided, the indices will be used
        instead. Default is None. If provided, the wavenumbers must be provided in
        ascending order.

    Attributes
    ----------
    n_features_in_ : int
        The number of features in the input data.

    point_index_ : int
        The index of the point to scale the data by. It is 0 if the wavenumbers are not provided.

    Examples
    --------
    >>> from chemotools.datasets import load_fermentation_train
    >>> from chemotools.scale import PointScaler
    >>> # Load sample data
    >>> X, _ = load_fermentation_train()
    >>> # Initialize PointScaler with point index
    >>> scaler = PointScaler(point=10)
    PointScaler(point=10, wavenumbers=None)
    >>> # Fit and transform the data
    >>> X_scaled = scaler.fit_transform(X)
    """

    _parameter_constraints: dict = {
        "point": [Interval(Integral, 0, None, closed="left")],
        "wavenumbers": ["array-like", None],
    }

    def __init__(self, point: int = 0, wavenumbers: Optional[np.ndarray] = None):
        self.point = point
        self.wavenumbers = wavenumbers

    def fit(self, X: np.ndarray, y=None) -> "PointScaler":
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
        self : PointScaler
            The fitted transformer.
        """
        # Check that X is a 2D array and has only finite values
        X = validate_data(
            self, X, y="no_validation", ensure_2d=True, reset=True, dtype=np.float64
        )
        # Set the point index
        if self.wavenumbers is None:
            self.point_index_ = self.point
        else:
            self.point_index_ = self._find_index(self.point)

        return self

    def transform(self, X: np.ndarray, y=None) -> np.ndarray:
        """
        Transform the input data by scaling by the value at a given Point.

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
        check_is_fitted(self, "point_index_")

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

        # Scale the data by Point
        for i, x in enumerate(X_):
            X_[i] = x / x[self.point_index_]

        return X_.reshape(-1, 1) if X_.ndim == 1 else X_

    def _find_index(self, target: float) -> int:
        wavenumbers = np.array(self.wavenumbers)
        return int(np.argmin(np.abs(wavenumbers - target)))
