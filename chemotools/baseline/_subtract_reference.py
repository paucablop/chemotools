"""
The :mod:`chemotools.baseline._subtract_reference` module implements
a reference spectrum subtraction transformer.
"""

# Author: Pau Cabaneros
# License: MIT

from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin
from sklearn.utils._param_validation import Integral, Interval
from sklearn.utils.validation import check_is_fitted, validate_data

from chemotools._axis_mixin import XAxisMixin


class SubtractReference(
    XAxisMixin, TransformerMixin, OneToOneFeatureMixin, BaseEstimator
):
    """
    Subtract a reference spectrum from spectral data.

    By default, the transformer computes :math:`x - r` for each sample.
    When ``scale_reference=True``, the reference is first scaled by an
    optimal factor :math:`a` that solves:

    .. math::

        \min_a \|x - a \cdot r\|_2

    and returns :math:`x - a \cdot r`. The factor can be computed over a
    sub-range of the spectrum defined by ``start`` and ``end``.

    Parameters
    ----------
    reference : np.ndarray, optional, default=None
        The reference spectrum to subtract from the input
        data. If None, the original spectrum is returned.

    scale_reference : bool, default=False
        If True, the reference is scaled by a factor :math:`a` before
        subtraction, where :math:`a` minimises :math:`\|x - a \cdot r\|_2`
        (or over the sub-range defined by ``start`` / ``end``).
        If False, a simple subtraction :math:`x - r` is performed and
        ``start``, ``end``, and ``x_axis`` are ignored.

    start : int, default=0
        Index or x-axis value of the start of the range used to
        compute the scaling factor. Only used when
        ``scale_reference=True``.

    end : int, default=-1
        Index or x-axis value of the end of the range used to
        compute the scaling factor. Only used when
        ``scale_reference=True``.

    x_axis : array-like, optional
        X-axis values corresponding to columns. When provided,
        ``start`` and ``end`` are interpreted as x-axis values and
        the closest indices are used. Must be ascending if provided.

    Attributes
    ----------
    n_features_in_ : int
        The number of features in the input data.

    reference_ : np.ndarray or None
        The reference spectrum to subtract from the input
        data. None if the reference parameter was not provided.

    Examples
    --------
    >>> from chemotools.baseline import SubtractReference
    >>> from chemotools.datasets import load_fermentation_train
    >>> # Load sample data
    >>> X, _ = load_fermentation_train()
    >>> # Convert X to a numpy array
    >>> X = np.array(X)
    >>> # Instantiate the transformer with a reference spectrum
    >>> reference = X[0]
    >>> transformer = SubtractReference(reference=reference)
    SubtractReference()
    >>> transformer.fit(X)
    >>> # Generate baseline-corrected data
    >>> X_corrected = transformer.transform(X)
    """

    _parameter_constraints: dict = {
        "reference": ["array-like", None],
        "scale_reference": ["boolean"],
        "start": Interval(Integral, 0, None, closed="left"),
        "end": [Integral],
        "x_axis": ["array-like", None],
    }

    def __init__(
        self,
        reference: Optional[np.ndarray] = None,
        scale_reference: bool = False,
        start: int = 0,
        end: int = -1,
        x_axis: Optional[np.ndarray] = None,
    ):
        self.reference = reference
        self.scale_reference = scale_reference
        self.start = start
        self.end = end
        self.x_axis = x_axis

    def fit(self, X: np.ndarray, y=None) -> "SubtractReference":
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
        self : SubtractReference
            The fitted transformer.
        """
        # Check that X is a 2D array and has only finite values
        X = validate_data(
            self, X, y="no_validation", ensure_2d=True, reset=True, dtype=np.float64
        )

        # Set the reference
        if self.reference is not None:
            # Check that the reference is a 1D array and has only finite values
            reference = np.asarray(self.reference, dtype=np.float64)
            if reference.ndim != 1:
                raise ValueError(
                    f"Reference spectrum must be a 1D array. "
                    f"Got {reference.ndim}D array instead."
                )
            if not np.isfinite(reference).all():
                raise ValueError("Reference spectrum must contain only finite values.")

            self.reference_ = reference
            if self.reference_.shape[0] != X.shape[1]:
                raise ValueError(
                    f"Reference spectrum must have the same number of features as X. "
                    f"Got {self.reference_.shape[0]} features in reference "
                    f"and {X.shape[1]} features in X."
                )
        else:
            self.reference_ = None

        # Set the start and end indices if scale_reference is True
        if self.scale_reference:
            if self.x_axis is None:
                self.start_index_ = self.start
                self.end_index_ = self.end if self.end != -1 else X.shape[1]
                self.x_axis_ = None
            else:
                axis = np.asarray(self.x_axis)
                self.start_index_ = self._find_index(self.start, axis)
                self.end_index_ = (
                    X.shape[1] if self.end == -1 else self._find_index(self.end, axis)
                )
                self.x_axis_ = axis[self.start_index_ : self.end_index_]

            if self.start_index_ >= self.end_index_:
                raise ValueError(
                    f"start_index ({self.start_index_}) must be less than "
                    f"end_index ({self.end_index_})."
                )

        return self

    def transform(self, X: np.ndarray, y=None) -> np.ndarray:
        """
        Transform the input data by subtracting the reference spectrum.

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

        if self.reference_ is None:
            return X_

        # Calculate scaling factor if scale_reference is True
        if self.scale_reference:
            scaling_factor = self._calculate_scaling_factor(X_)
            return X_ - scaling_factor[:, np.newaxis] * self.reference_

        # Subtract the reference
        return X_ - self.reference_

    def _calculate_scaling_factor(self, X: np.ndarray) -> np.ndarray:
        """Calculate the optimal scaling factor per sample.

        Solves ``min_a ||x[n:m] - a * r[n:m]||_2`` for each row in *X*.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        scaling_factor : np.ndarray of shape (n_samples,)
            The optimal scaling factor for each sample.
        """
        assert self.reference_ is not None  # guaranteed by transform guard
        X_cut = X[:, self.start_index_ : self.end_index_]
        reference_cut = self.reference_[self.start_index_ : self.end_index_]

        denom = np.dot(reference_cut, reference_cut)
        if np.isclose(denom, 0.0):
            raise ValueError(
                "Reference spectrum has zero or near-zero norm in the "
                f"specified range [{self.start_index_}:{self.end_index_}] "
                f"(denom={denom}). Cannot compute scaling factor."
            )

        return np.dot(X_cut, reference_cut) / denom
