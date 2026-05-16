"""
The :mod:`chemotools.scale._band_scaler` module implements a Band Scaler transformer.
"""

# Authors: Pau Cabaneros
# License: MIT
import warnings
from numbers import Real
from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.validation import check_is_fitted, validate_data

from chemotools._axis_mixin import XAxisMixin
from chemotools._deprecation import (
    DEPRECATED_PARAMETER,
    deprecated_parameter_constraint,
)
from chemotools._doc_mixin import DocLinkMixin


class BandScaler(
    DocLinkMixin, XAxisMixin, TransformerMixin, OneToOneFeatureMixin, BaseEstimator
):
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

    aggregation : {'mean', 'area'}, default='mean'
        The aggregation method to use for calculating the band intensity.
        - 'mean': Calculate the mean intensity of the band.
        - 'area': Calculate the area under the band using the trapezoidal rule.

    baseline_correction : bool, default=False
        If True, a linear baseline connecting the band endpoints is
        subtracted from the band before computing the scaling factor.
        This removes the effect of a sloped baseline on the mean or
        area calculation.

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


    Notes
    -----
    The choice between 'mean' and 'area' aggregation depends on whether the
    normalization should be based on average signal intensity or total
    integrated signal:

    - **Mean Scaling ('mean')**: Normalizes by the average intensity across the
        band. This is standard for correcting global intensity fluctuations
        (e.g., source power drift or pathlength changes) while preserving the
        relative magnitude of the spectral profile.

    - **Area Scaling ('area')**: Normalizes by the numerical integral
        (Trapezoidal rule) of the band. In many spectroscopic applications,
        the area under a curve is more representative of the total concentration
        or molar abundance than a single peak height or average intensity.

    **Importance of Coordinate-Aware Scaling**:
    In some spectrometers, the sampling interval (distance between
    points on the x-axis) is not perfectly constant across the entire detector.
    - If the sampling is **non-linear**, a simple summation (equivalent to
        assuming :math:`\Delta x=1`) will mathematically over-weight regions where data
        points are more densely packed.
    - By providing an `x_axis`, the 'area' method uses the actual distances
        between points (:math:`\Delta x`) to calculate a physically accurate integral.

    When using ``aggregation='area'``, an ``x_axis`` must be provided. If it is
    omitted, the transformer raises a :class:`ValueError` rather than implicitly
    assuming uniform sampling density across the selected band.

    See also
    --------
    chemotools.scale.MinMaxScaler : Scales features to the Min-Max range.
    chemotools.scale.NormScaler : Scales features to unit norm.
    chemotools.scale.PointScaler : Scales features by the intensity at a specific point.
    """

    _parameter_constraints: dict = {
        "start": [Interval(Real, 0, None, closed="left")],
        "end": [Interval(Real, -1, None, closed="left")],
        "x_axis": ["array-like", None],
        "aggregation": [StrOptions({"mean", "area"})],
        "baseline_correction": ["boolean"],
        "wavenumbers": ["array-like", None, deprecated_parameter_constraint()],
    }

    def __init__(
        self,
        start: int = 0,
        end: int = -1,
        x_axis: Optional[np.ndarray] = None,
        aggregation: str = "mean",
        baseline_correction: bool = False,
        wavenumbers=DEPRECATED_PARAMETER,
    ):
        self.start = start
        self.end = end
        self.x_axis = x_axis
        self.aggregation = aggregation
        self.baseline_correction = baseline_correction
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
        self.axis_values_ = np.asarray(axis_values) if axis_values is not None else None

        # Resolve the point index
        if self.axis_values_ is None:
            self.start_index_ = self.start
            self.end_index_ = self.end
        else:
            self.start_index_ = self._find_index(self.start, self.axis_values_)
            self.end_index_ = self._find_index(self.end, self.axis_values_)

        # Validate that the end is greater than start
        if self.start_index_ >= self.end_index_ and self.end_index_ != -1:
            raise ValueError(
                f"start_index_ ({self.start_index_}) must be less than "
                f"end_index_ ({self.end_index_})."
            )

        # Validate that x_axis is provided when aggregation is 'area'
        if self.aggregation == "area" and self.axis_values_ is None:
            raise ValueError("x_axis must be provided when aggregation='area'.")

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

        # 1. Extract the band of interest
        band_y = X_[:, self.start_index_ : self.end_index_]

        # Resolve band x-axis values if available (used for baseline correction
        # and area). Use the persisted axis from fit() so the deprecated
        # ``wavenumbers`` path works correctly.
        band_x = (
            self.axis_values_[self.start_index_ : self.end_index_]
            if self.axis_values_ is not None
            else None
        )

        # 2. Apply baseline correction if enabled
        if self.baseline_correction:
            if band_x is not None:
                x_range = band_x[-1] - band_x[0]
                t = (
                    (band_x - band_x[0]) / x_range
                    if x_range != 0
                    else np.linspace(0, 1, num=band_y.shape[1])
                )
            else:
                t = np.linspace(0, 1, num=band_y.shape[1])
            baseline = band_y[:, 0:1] + t * (band_y[:, -1:] - band_y[:, 0:1])
            band_y = band_y - baseline

        # 3. Scale the data by the average intensity of the specified band
        if self.aggregation == "mean":
            scaling_factor = band_y.mean(axis=1, keepdims=True)

        # 4. Scale by the area under the band
        elif self.aggregation == "area":
            trapz_func = getattr(
                np, "trapezoid", getattr(np, "trapz", None)
            )  # support for numpy < 2.0.0
            assert trapz_func is not None  # available in all supported numpy versions
            # Handle non-constant sampling using the Trapezoidal rule
            assert band_x is not None  # narrow type (validated in fit())
            scaling_factor = trapz_func(band_y, x=band_x, axis=1)[:, np.newaxis]

        # Avoid division by zero by setting zero means to one (no scaling) and raise
        # user warning
        zero_mask = np.isclose(scaling_factor, 0).ravel()
        if zero_mask.any():
            zero_indices = np.flatnonzero(zero_mask).tolist()
            warnings.warn(
                f"The scaling factor for sample(s) {zero_indices} is zero. "
                "These samples will not be scaled.",
                UserWarning,
            )
            scaling_factor = np.where(np.isclose(scaling_factor, 0), 1, scaling_factor)

        return X_ / scaling_factor
