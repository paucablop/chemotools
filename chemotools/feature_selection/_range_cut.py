"""
The :mod:`chemotools.feature_selection._range_cut` module implements the RangeCut
to select specific features from spectral data based on start and end indices or
wavenumbers.
"""

from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.feature_selection._base import SelectorMixin
from sklearn.utils.validation import check_is_fitted, validate_data
from sklearn.utils._param_validation import Interval, Integral


class RangeCut(SelectorMixin, BaseEstimator):
    """Select a contiguous spectral region by index or by wavenumber.

    The range can be specified in two ways:

    * By integer indices (``start`` and ``end``)
    * By wavenumber values (``start`` and ``end`` interpreted against the
        provided ``wavenumbers`` array)

    If ``wavenumbers`` is supplied, the closest indices to the given start / end
    wavenumber values are located. Otherwise numeric ``start`` / ``end`` are
    treated directly as indices. Wavenumbers must be in ascending order.

    Parameters
    ----------
    start : int, default=0
        Index or wavenumber of the start of the range.
    end : int, default=-1
        Index or wavenumber of the end of the range.
    wavenumbers : array-like, optional
        Wavenumbers corresponding to columns. Must be ascending if provided.

    Attributes
    ----------
    start_index_ : int
        Resolved start index.
    end_index_ : int
        Resolved end index.
    wavenumbers_ : array-like or None
        Selected wavenumbers (if provided), else ``None``.

    Examples
    --------
    >>> from chemotools.feature_selection import RangeCut
    >>> from chemotools.datasets import load_fermentation_train
    >>> X, _ = load_fermentation_train()
    >>> wavenumbers = X.columns.values
    >>> rc = RangeCut(start=1000, end=2000, wavenumbers=wavenumbers)
    >>> rc.fit(X)
    RangeCut(start=1000, end=2000, wavenumbers=wavenumbers)
    >>> X_cut = rc.transform(X)
    >>> X_cut.shape
    (21, 616)
    """

    _parameter_constraints: dict = {
        "start": Interval(Integral, 0, None, closed="left"),
        "end": [Integral],
        "wavenumbers": ["array-like", None],
    }

    def __init__(
        self,
        start: int = 0,
        end: int = -1,
        wavenumbers: Optional[np.ndarray] = None,
    ):
        self.start = start
        self.end = end
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
        # Set the start and end indices
        if self.wavenumbers is None:
            self.start_index_ = self.start
            self.end_index_ = self.end
            self.wavenumbers_ = None
        else:
            self.start_index_ = self._find_index(self.start)
            self.end_index_ = self._find_index(self.end)
            self.wavenumbers_ = self.wavenumbers[self.start_index_ : self.end_index_]

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
        mask = np.zeros(self.n_features_in_, dtype=bool)
        mask[self.start_index_ : self.end_index_] = True

        return mask

    def _find_index(self, target: float) -> int:
        wavenumbers = np.array(self.wavenumbers)
        return int(np.argmin(np.abs(wavenumbers - target)))
