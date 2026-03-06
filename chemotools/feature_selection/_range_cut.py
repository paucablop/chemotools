"""
The :mod:`chemotools.feature_selection._range_cut` module implements the RangeCut
to select specific features from spectral data based on start and end indices or
x-axis values.
"""

from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.feature_selection._base import SelectorMixin
from sklearn.utils._param_validation import Integral, Interval
from sklearn.utils.validation import check_is_fitted, validate_data

from chemotools._deprecation import (
    DEPRECATED_PARAMETER,
    deprecated_parameter_constraint,
    resolve_renamed_parameter,
)


class RangeCut(SelectorMixin, BaseEstimator):
    """Select a contiguous spectral region by index or by x-axis value.

    The range can be specified in two ways:

    * By integer indices (``start`` and ``end``)
    * By x-axis values (``start`` and ``end`` interpreted against the provided
        ``x_axis`` array)

    If ``x_axis`` is supplied, the closest indices to the given start / end
    x-axis values are located. Otherwise numeric ``start`` / ``end`` are
    treated directly as indices. X-axis values must be in ascending order.

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
        Resolved start index.
    end_index_ : int
        Resolved end index.
    x_axis_ : array-like or None
        Selected x-axis values (if provided), else ``None``.
    wavenumbers_ : array-like or None
        Deprecated alias for ``x_axis_``.

    Examples
    --------
    >>> from chemotools.feature_selection import RangeCut
    >>> from chemotools.datasets import load_fermentation_train
    >>> X, _ = load_fermentation_train()
    >>> wavenumbers = X.columns.values
    >>> rc = RangeCut(start=1000, end=2000, x_axis=wavenumbers)
    >>> rc.fit(X)
    RangeCut(start=1000, end=2000, x_axis=wavenumbers)
    >>> X_cut = rc.transform(X)
    >>> X_cut.shape
    (21, 616)
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

    def fit(self, X: np.ndarray, y=None) -> "RangeCut":
        """
        Fit the transformer to the input data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data to fit the transformer to.

        y : None
            Ignored to align with API.

        Returns
        -------
        self : RangeCut
            The fitted transformer.
        """
        # Check that X is a 2D array and has only finite values
        X = validate_data(
            self, X, y="no_validation", ensure_2d=True, reset=True, dtype=np.float64
        )

        axis_values = self._resolve_x_axis()

        # Set the start and end indices
        if axis_values is None:
            self.start_index_ = self.start
            self.end_index_ = self.end
            self.x_axis_ = None
            self.wavenumbers_ = None
        else:
            axis = np.asarray(axis_values)
            self.start_index_ = self._find_index(self.start, axis)
            self.end_index_ = self._find_index(self.end, axis)
            self.x_axis_ = axis_values[self.start_index_ : self.end_index_]
            self.wavenumbers_ = self.x_axis_

        return self

    def _get_support_mask(self):
        """
        Get the boolean mask indicating which features are selected.

        Returns
        -------
        mask : np.ndarray of shape (n_features,)
            The boolean mask indicating which features are selected.
        """
        # Check that the estimator is fitted
        check_is_fitted(self, ["start_index_", "end_index_"])

        # Create the mask
        mask = np.zeros(self.n_features_in_, dtype=bool)  # type: ignore[unresolved-attribute]  # sklearn fitted attribute
        mask[self.start_index_ : self.end_index_] = True

        return mask

    def _resolve_x_axis(self):
        return resolve_renamed_parameter(
            new_name="x_axis",
            new_value=self.x_axis,
            new_default=None,
            old_name="wavenumbers",
            old_value=self.wavenumbers,
        )

    def _find_index(self, target: float, axis: np.ndarray) -> int:
        return int(np.argmin(np.abs(axis - target)))
