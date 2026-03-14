"""
The :mod:`chemotools.scale._band_scaler` module implements a Band Scaler transformer.
"""

# Authors: Pau Cabaneros
# License: MIT
import warnings
from numbers import Integral
from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin
from sklearn.utils._param_validation import Interval
from sklearn.utils.validation import check_is_fitted, validate_data

from chemotools._axis_mixin import XAxisMixin
from chemotools._deprecation import (
    DEPRECATED_PARAMETER,
    deprecated_parameter_constraint,
)


class BandScaler(XAxisMixin, TransformerMixin, OneToOneFeatureMixin, BaseEstimator):
    """
    A transformer that scales the input data by the average intensity of a specified
    band. The band can be specified by an index range or by a range of wavenumbers.

    Parameters
    ----------
    start : int, default=0
        Index or x-axis value of the start of the range.

    end : int, default=-1
        Index or x-axis value of the end of the range.

    x_axis : array-like, optional
        X-axis values corresponding to columns. Must be ascending if provided.

    wavenumbers : array-like, optional
        Deprecated alias for ``x_axis``. Use ``x_axis`` instead.

    Attributes
    ----------
    start_index_ : int
        The index of the start of the band.

    end_index_ : int
        The index of the end of the band.

    n_features_in_ : int
        The number of features in the input data.


    Examples
    --------
    >>> from chemotools.datasets import load_fermentation_train
    >>> from chemotools.scale import BandScaler
    >>> # Load sample data
    >>> X, _ = load_fermentation_train()
    >>> # Initialize BandScaler with band indices
    >>> scaler = BandScaler(start=10, end=20)
    BandScaler(start=10, end=20)
    >>> # Fit and transform the data
    >>> X_scaled = scaler.fit_transform(X)
    """

    _parameter_constraints: dict = {
        "start": Interval(Integral, 0, None, closed="left"),
        "end": [Integral],
        "x_axis": ["array-like", None],
        "wavenumbers": ["array-like", None, deprecated_parameter_constraint()],
    }

    def __init__(
        self,
        start: int = 0,
        end: int = -1,
        x_axis: Optional[np.ndarray] = None,
        wavenumbers=DEPRECATED_PARAMETER,
    ):
        self.start = start
        self.end = end
        self.x_axis = x_axis
        self.wavenumbers = wavenumbers

    def fit(self, X: np.ndarray, y=None) -> "BandScaler":
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
        self : BandScaler
            The fitted transformer.
        """
        # Validate the input parameters
        self._validate_params()

        # Check that X is a 2D array and has only finite values
        X = validate_data(
            self, X, y="no_validation", ensure_2d=True, reset=True, dtype=np.float64
        )

        axis_values = self._resolve_x_axis(self.x_axis, self.wavenumbers)

        # Resolve the point index
        if axis_values is None:
            self.start_index_ = self.start
            self.end_index_ = self.end
        else:
            self.start_index_ = self._find_index(self.start, axis_values)
            self.end_index_ = self._find_index(self.end, axis_values)

        # Validate that the end is greater than start
        if self.start_index_ >= self.end_index_ and self.end_index_ != -1:
            raise ValueError(
                f"start_index_ ({self.start_index_}) must be less than "
                f"end_index_ ({self.end_index_})."
            )

        return self

    def transform(self, X: np.ndarray, y=None) -> np.ndarray:
        """
        Transform the input data by scaling by the average intensity of the specified
        band.

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
        check_is_fitted(self, ["start_index_", "end_index_"])

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

        # Scale the data by the average intensity of the specified band
        band_mean = X_[:, self.start_index_ : self.end_index_].mean(
            axis=1, keepdims=True
        )

        # Avoid division by zero by setting zero means to one (no scaling) and raise
        # user warning
        if np.isclose(band_mean, 0).any():
            warnings.warn(
                "The mean for sample(s) is zero. These samples will not be scaled.",
                UserWarning,
            )

        return X_ / band_mean
