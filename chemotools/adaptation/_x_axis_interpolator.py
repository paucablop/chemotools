"""
The :mod:`chemotools.adaptation._x_axis_interpolator` allows resampling each X to a
common x_axis grid.
"""

# Author: Pau Cabaneros
# License: MIT

from numbers import Real

import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator
from sklearn.base import BaseEstimator, TransformerMixin, _fit_context
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.validation import check_is_fitted, validate_data

from chemotools._doc_mixin import DocLinkMixin


class XAxisInterpolator(DocLinkMixin, TransformerMixin, BaseEstimator):
    """Interpolate every row of ``X`` onto a shared ``common_x_axis``.

    The transformer resamples each row of ``X`` from a sample-specific (or shared)
    input grid ``x_axis`` onto a fixed ``common_x_axis`` provided at instantiation
    time.

    ``x_axis`` is consumed as **metadata** so it flows correctly through
    ``Pipeline``, ``ColumnTransformer``, ``GridSearchCV``, ``cross_validate`` etc.,
    once metadata routing is enabled via:

    ``sklearn.set_config(enable_metadata_routing=True)``.

    Parameters
    ----------
    common_x_axis : array-like of shape (n_output_features,)
        Strictly increasing target grid. ``transform`` returns an array with
        ``n_output_features`` columns.

    method : str, default="cubic"
        The interpolation mode. One of:
        ``"linear"``, ``"cubic"`` or ``"pchip"``.

    left : float, default=np.nan
        Value returned for query points below the input grid (passed to
        :func:`numpy.interp`).

    right : float, default=np.nan
        Value returned for query points above the input grid.

    Attributes
    ----------
    common_x_axis_ : ndarray of shape (n_output_features,)
        Validated copy of ``common_x_axis``.

    n_features_in_ : int
        Number of input features seen during ``fit``.

    feature_names_in_ : ndarray of shape (``n_features_in_``,)
        Names of features seen during ``fit`` (only if ``X`` had names).
    """

    _parameter_constraints: dict = {
        "common_x_axis": ["array-like"],
        "method": [StrOptions({"linear", "cubic", "pchip"})],
        "left": [Interval(Real, None, None, closed="neither"), None, np.nan.__class__],
        "right": [Interval(Real, None, None, closed="neither"), None, np.nan.__class__],
    }

    def __init__(
        self,
        common_x_axis: np.ndarray,
        method: str = "cubic",
        left=np.nan,
        right=np.nan,
    ):
        self.common_x_axis = common_x_axis
        self.method = method
        self.left = left
        self.right = right

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X: np.ndarray, y=None, x_axis=None) -> "XAxisInterpolator":
        """Validate input and the configured ``common_x_axis``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data to fit the transformer to.

        y : Ignored
            Ignored to align with API.

        x_axis : Ignored
            Accepted only so that metadata routing through ``fit_transform``
            (used by ``Pipeline``) generates a ``set_fit_request`` method.
            ``fit`` itself does not use it.

        Returns
        -------
        self : object
        """
        # Validate the input parameters
        self._validate_params()

        # Validate the X data
        validate_data(self, X, ensure_2d=True, dtype="numeric")

        # Validate the common_x_axis
        common_x_axis = np.asarray(self.common_x_axis, dtype=float)
        if common_x_axis.ndim != 1:
            raise ValueError(
                f"common_x_axis must be 1D, got shape {common_x_axis.shape}."
            )
        if common_x_axis.size < 2:
            raise ValueError("common_x_axis must contain at least 2 points.")
        if not np.all(np.isfinite(common_x_axis)):
            raise ValueError("common_x_axis must contain only finite values.")
        if not np.all(np.diff(common_x_axis) > 0):
            raise ValueError("common_x_axis must be strictly increasing.")

        self.common_x_axis_ = common_x_axis

        return self

    def transform(self, X: np.ndarray, x_axis=None):
        """Interpolate ``X`` from ``x_axis`` onto ``common_x_axis_``.

        ``x_axis`` is **metadata** and must be requested explicitly via
        ``set_transform_request(x_axis=True)`` for routing to work in a
        ``Pipeline`` / ``GridSearchCV``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Signal values sampled on ``x_axis``.

        x_axis : array-like, required metadata
            Either shape ``(n_features,)`` (shared grid for every row) or
            ``(n_samples, n_features)`` (per-row grid). Must be strictly
            increasing along the feature axis.

        Returns
        -------
        X_transformed : ndarray of shape (n_samples, n_output_features)
        """
        # Check the estimator is fitted
        check_is_fitted(self, "common_x_axis_")

        # Validate the X array
        X = validate_data(self, X, reset=False, ensure_2d=True, dtype="numeric")

        # Validate the x-axis
        if x_axis is None:
            raise ValueError(
                "`x_axis` metadata is required. Enable metadata routing with "
                "`sklearn.set_config(enable_metadata_routing=True)` and call "
                "`set_transform_request(x_axis=True)`, or pass it directly to "
                "`transform`."
            )

        n_samples = X.shape[0]
        x_per_row = self._prepare_x_axis(x_axis, X.shape)

        target = self.common_x_axis_
        method = self.method
        if method not in {"linear", "cubic", "pchip"}:
            raise ValueError(
                f"Unknown interpolation method={method!r}. "
                "Expected one of {'linear', 'cubic', 'pchip'}."
            )

        X_transformed = np.empty((n_samples, target.size), dtype=float)
        for i in range(n_samples):
            xi = x_per_row[i]
            left_value, right_value = self._resolve_fill_values(
                X[i], self.left, self.right
            )
            X_transformed[i] = self._interpolate_row(
                xi=xi,
                yi=X[i],
                target=target,
                left_value=left_value,
                right_value=right_value,
            )

        return X_transformed

    def fit_transform(self, X, y=None, **fit_params):
        """Fit to data, then transform it.

        ``fit`` does not consume ``x_axis``; only ``transform`` does. This
        custom implementation ensures ``x_axis`` is forwarded only to
        ``transform``, not to ``fit``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : Ignored
        x_axis : array-like, optional metadata
            Input grid for interpolation, forwarded to ``transform``.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_output_features)
        """
        # get the x_axis from **fit_params
        x_axis = fit_params.pop("x_axis", None)

        # fit does not consume x_axis, but may consume other forwarded metadata.
        return self.fit(X, y, **fit_params).transform(X, x_axis=x_axis)

    def get_feature_names_out(self, input_features=None):
        """Names are the positions of the common grid."""
        check_is_fitted(self)
        return np.asarray([f"x_{v:g}" for v in self.common_x_axis_], dtype=object)

    def _prepare_x_axis(
        self, x_axis: np.ndarray, X_shape: tuple[int, int]
    ) -> np.ndarray:
        """Validate and normalize x_axis to shape (n_samples, n_features)."""
        n_samples, n_features = X_shape

        x_axis = np.asarray(x_axis, dtype=float)
        if x_axis.ndim == 1:
            if x_axis.shape[0] != n_features:
                raise ValueError(
                    f"x_axis has shape {x_axis.shape}, expected ({n_features},) "
                    f"or ({n_samples}, {n_features})."
                )
            x_per_row = np.broadcast_to(x_axis, (n_samples, n_features))
        elif x_axis.ndim == 2:
            if x_axis.shape != X_shape:
                raise ValueError(
                    f"x_axis has shape {x_axis.shape}, expected {X_shape}."
                )
            x_per_row = x_axis
        else:
            raise ValueError(f"x_axis must be 1D or 2D, got {x_axis.ndim}D array.")

        if not np.all(np.isfinite(x_per_row)):
            raise ValueError("x_axis must contain only finite values.")

        if not np.all(np.diff(x_per_row, axis=1) > 0):
            raise ValueError("x_axis must be strictly increasing along axis=1.")

        return x_per_row

    @staticmethod
    def _resolve_fill_values(
        row: np.ndarray, left, right
    ) -> tuple[float | np.floating, float | np.floating]:
        """Resolve boundary fill values for one row."""
        left_value = row[0] if left is None else left
        right_value = row[-1] if right is None else right
        return left_value, right_value

    def _interpolate_row(
        self,
        *,
        xi: np.ndarray,
        yi: np.ndarray,
        target: np.ndarray,
        left_value,
        right_value,
    ) -> np.ndarray:
        """Interpolate one row according to the configured method."""
        method = self.method
        if method == "linear":
            return np.interp(target, xi, yi, left=left_value, right=right_value)

        if method == "cubic":
            interp_func = CubicSpline(xi, yi, bc_type="not-a-knot", extrapolate=False)
        else:  # method == "pchip"
            interp_func = PchipInterpolator(xi, yi, extrapolate=False)

        row = interp_func(target)
        row[target < xi[0]] = left_value
        row[target > xi[-1]] = right_value
        return row
