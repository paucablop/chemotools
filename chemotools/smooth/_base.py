# _base.py
# Authors: Niklas Zell <nik.zoe@web.de>, Nusret Emirhan Salli <nusret.emirhan.salli@gmail.com>, Pau Cabaneros
# License: MIT

from __future__ import annotations
from abc import ABC, abstractmethod
import logging
from typing import Callable, Literal, Optional

import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin, OneToOneFeatureMixin
from sklearn.utils.validation import check_is_fitted, validate_data

from chemotools.utils._linear_algebra import (
    compute_DtD_banded,
    compute_DtD_sparse,
    whittaker_smooth_banded,
    whittaker_smooth_sparse,
    whittaker_solver_dispatch,
)

logger = logging.getLogger(__name__)


class _BaseWhittaker(TransformerMixin, OneToOneFeatureMixin, BaseEstimator, ABC):
    """Base class for Whittaker-based algorithms (smoothing or baseline correction).

    This implements the sklearn boilerplate (validation, fitted checks)
    and delegates algorithm-specific behavior to subclasses via
    `_fit_core` and `_transform_core`.
    """

    def __init__(
        self,
        lam: float = 1e4,
        weights: Optional[np.ndarray] = None,
        solver_type: Literal["banded", "sparse"] = "banded",
    ):
        self.lam = lam
        self.weights = weights
        self.solver_type = solver_type

    def fit(self, X: np.ndarray, y=None) -> "_BaseWhittaker":
        X = validate_data(self, X, ensure_2d=True, reset=True, dtype=np.float64)
        self.DtD_ = self._precompute_DtD(X.shape[1])
        solver = whittaker_solver_dispatch(self.solver_type)
        return self._fit_core(X, y, solver=solver)

    def transform(self, X: np.ndarray, y=None) -> np.ndarray:
        check_is_fitted(self, ["DtD_"])
        X_ = validate_data(self, X, ensure_2d=True, copy=True, reset=False)
        solver = whittaker_solver_dispatch(self.solver_type)
        return self._transform_core(X_, y, solver=solver)

    @abstractmethod
    def _fit_core(
        self,
        X: np.ndarray,
        y=None,
        nr_iterations: int = 1,
        solver: Callable = whittaker_smooth_banded,
    ) -> "_BaseWhittaker":
        """Subclasses can extend fitting logic here."""
        ...

    @abstractmethod
    def _transform_core(
        self,
        X: np.ndarray,
        y=None,
        nr_iterations: int = 1,
        solver: Callable = whittaker_smooth_banded,
    ) -> np.ndarray:
        """Subclasses must override to implement algorithm-specific transform."""
        ...

    def _precompute_DtD(self, n_features: int):
        return (
            compute_DtD_banded(n_features)
            if self.solver_type == "banded"
            else compute_DtD_sparse(n_features)
        )

    def _solve_whittaker(
        self, x: np.ndarray, w: np.ndarray, solver: Optional[Callable]
    ) -> np.ndarray:
        """Solve (diag(w) + lam*D^T D) z = w*x."""
        if solver is None:
            solver = whittaker_solver_dispatch(self.solver_type)
        try:
            return solver(x, w, self.lam, self.DtD_)
        except Exception as e:
            logger.debug("Primary solver failed (%s); fallback to sparse LU.", e)
            DtD = compute_DtD_sparse(len(x))
            return whittaker_smooth_sparse(x, w, self.lam, DtD)


class _BaseFIRFilter(TransformerMixin, OneToOneFeatureMixin, BaseEstimator, ABC):
    """
    Base class for linear-phase FIR smoothers.

    Subclasses must implement `_compute_kernel(self) -> np.ndarray`
    returning a 1D symmetric kernel of odd length whose sum is 1.0.

    Parameters
    ----------
    window_size : int, odd >= 3
        Number of taps in the FIR kernel.
    mode : {"mirror","constant","nearest","wrap","interp"}, default="interp"
        Boundary handling. "interp" = linear extrapolation (recommended for MS).  # Schmid et al.
    axis : int, default=1
        Axis along which to smooth for 2D inputs (rows × features). Use 1 to
        smooth along feature axis for each row.
    """

    def __init__(
        self,
        window_size: int = 21,
        mode: Literal["mirror", "constant", "nearest", "wrap", "interp"] = "interp",
        axis: int = 1,
    ) -> None:
        self.window_size = window_size
        self.mode = mode
        self.axis = axis

    # sklearn API
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "_BaseFIRFilter":
        X = validate_data(
            self, X, y="no_validation", ensure_2d=True, reset=True, dtype=np.float64
        )

        if self.window_size < 3 or self.window_size % 2 == 0:
            raise ValueError("window_size must be an odd integer >= 3.")
        self.kernel_ = self._compute_kernel().astype(np.float64, copy=False)
        if self.kernel_.ndim != 1 or self.kernel_.size != self.window_size:
            raise ValueError("kernel must be 1D with length equal to window_size.")
        if not np.allclose(self.kernel_.sum(), 1.0, atol=1e-12):
            raise ValueError("kernel must be DC-preserving (sum == 1).")
        self._half_ = (self.window_size - 1) // 2
        return self

    def transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        check_is_fitted(self, "kernel_")
        X_ = validate_data(
            self,
            X,
            y="no_validation",
            ensure_2d=True,
            copy=True,
            reset=False,
            dtype=np.float64,
        )

        # move smoothing axis to last, convolve row-wise, then move back
        ax = self.axis if self.axis >= 0 else X_.ndim + self.axis
        X_sw = np.moveaxis(X_, ax, -1)
        lead = int(np.prod(X_sw.shape[:-1])) or 1
        L = X_sw.shape[-1]
        Z = X_sw.reshape(lead, L)
        for i in range(lead):
            Z[i] = self._apply_filter_1d(Z[i])
        out = X_sw.reshape(*X_sw.shape)
        return np.moveaxis(out, -1, ax)

    @abstractmethod
    def _compute_kernel(self) -> np.ndarray:
        """
        Subclasses must implement this method to compute the convolution kernel.
        """
        raise NotImplementedError

    # --- shared convolution/padding ---
    def _apply_filter_1d(self, x: np.ndarray) -> np.ndarray:
        m = self._half_
        xp = self._pad_1d(x, m)
        return np.convolve(xp, self.kernel_, mode="valid")  # same length as x

    def _pad_1d(self, x: np.ndarray, m: int) -> np.ndarray:
        if m == 0:
            return x.copy()
        mode = self.mode
        if mode == "interp":
            # Linear extrapolation using boundary slopes (paper’s recommendation).
            if x.size < 2:
                left = np.repeat(x[0], m)
                right = np.repeat(x[-1], m)
            else:
                ls = x[1] - x[0]
                rs = x[-1] - x[-2]
                left = x[0] - ls * np.arange(m, 0, -1, dtype=np.float64)
                right = x[-1] + rs * np.arange(1, m + 1, dtype=np.float64)
            return np.concatenate([left, x, right], axis=0)
        if mode == "nearest":
            return np.pad(x, (m, m), mode="edge")
        if mode == "mirror":
            return np.pad(x, (m, m), mode="reflect")  # mirror without repeating edge
        if mode == "wrap":
            return np.pad(x, (m, m), mode="wrap")
        if mode == "constant":
            # Match scipy's behavior: pad with zeros (cval=0.0 by default)
            return np.pad(x, (m, m), mode="constant", constant_values=0.0)
        raise ValueError(f"Unknown mode='{mode}'")
