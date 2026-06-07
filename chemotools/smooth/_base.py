# _base.py
# Authors: Niklas Zell <nik.zoe@web.de>,
#          Nusret Emirhan Salli <nusret.emirhan.salli@gmail.com>,
#          Pau Cabaneros
# License: MIT

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from numbers import Integral
from typing import TYPE_CHECKING, Literal, Optional

import numpy as np
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin
from sklearn.utils.validation import check_is_fitted, validate_data

from chemotools._deprecation import (
    DEPRECATED_PARAMETER,
    resolve_renamed_parameter,
)

if TYPE_CHECKING:
    from typing_extensions import Self

from sklearn.utils._param_validation import Interval

from chemotools._doc_mixin import DocLinkMixin
from chemotools._parallel import apply_rows
from chemotools.utils._linear_algebra import (
    compute_DtD_banded,
    compute_DtD_sparse,
)
from chemotools.utils._whittaker_solvers import (
    WhittakerSolver,
    whittaker_solver_factory,
)

logger = logging.getLogger(__name__)


class _BaseWhittaker(
    DocLinkMixin, TransformerMixin, OneToOneFeatureMixin, BaseEstimator, ABC
):
    """Base class for Whittaker-based algorithms (smoothing or baseline correction).

    This implements the sklearn boilerplate (validation, fitted checks)
    and delegates algorithm-specific behavior to subclasses via
    `_fit_core` and `_transform_block`.
    """

    _parameter_constraints: dict = {
        "n_jobs": [
            Interval(Integral, None, -1, closed="right"),
            Interval(Integral, 1, None, closed="left"),
        ],
    }

    def __init__(
        self,
        lam: float = 1e4,
        weights: Optional[np.ndarray] = None,
        solver_type: Literal["banded", "sparse"] = "banded",
        n_jobs: int = 1,
    ):
        self.lam = lam
        self.weights = weights
        self.solver_type = solver_type
        self.n_jobs = n_jobs

    def __setstate__(self, state: dict) -> None:
        """Restore state while keeping backward compatibility with old pickles."""
        super().__setstate__(state)
        if "n_jobs" not in self.__dict__:
            self.n_jobs = 1

    def fit(self, X: np.ndarray, y=None) -> Self:
        self._validate_params()
        X = validate_data(self, X, ensure_2d=True, reset=True, dtype=np.float64)
        self.DtD_ = self._precompute_DtD(X.shape[1])
        self.solver_ = whittaker_solver_factory(self.solver_type, self.lam, self.DtD_)
        return self._fit_core(X, y, solver=self.solver_)

    def transform(self, X: np.ndarray, y=None) -> np.ndarray:
        check_is_fitted(self, ["DtD_", "solver_"])
        X_ = validate_data(
            self, X, ensure_2d=True, copy=True, reset=False, dtype=np.float64
        )
        return apply_rows(X_, n_jobs=self.n_jobs, fn=self._transform_block)

    @abstractmethod
    def _fit_core(
        self,
        X: np.ndarray,
        y=None,
        solver: WhittakerSolver | None = None,
    ) -> Self:
        """Subclasses can extend fitting logic here."""
        ...

    @abstractmethod
    def _transform_block(self, X_block: np.ndarray) -> np.ndarray:
        """Subclasses must override to implement the per-block transform."""
        ...

    def _precompute_DtD(self, n_features: int):
        return (
            compute_DtD_banded(n_features)
            if self.solver_type == "banded"
            else compute_DtD_sparse(n_features)
        )


class _BaseFIRFilter(
    DocLinkMixin, TransformerMixin, OneToOneFeatureMixin, BaseEstimator, ABC
):
    """
    Base class for linear-phase FIR smoothers.

    Subclasses must implement `_compute_kernel(self) -> np.ndarray`
    returning a 1D symmetric kernel of odd length whose sum is 1.0.

    Parameters
    ----------
    window_length : int, odd >= 3
        Number of taps in the FIR kernel.
    mode : {"mirror","constant","nearest","wrap","interp"}, default="interp"
        Boundary handling. "interp" = linear extrapolation
        (recommended for MS).  # Schmid et al.
    axis : int, default=1
        Axis along which to smooth for 2D inputs (rows × features). Use 1 to
        smooth along feature axis for each row.
    window_size : int, optional
        Deprecated alias for ``window_length``.
    """

    def __init__(
        self,
        window_length: int = 21,
        mode: Literal["mirror", "constant", "nearest", "wrap", "interp"] = "interp",
        axis: int = 1,
        window_size=DEPRECATED_PARAMETER,
    ) -> None:
        self.window_length = window_length
        self.window_size = window_size
        self.mode = mode
        self.axis = axis

    def __setstate__(self, state: dict) -> None:
        """Restore old pickles that stored only the deprecated window alias."""
        super().__setstate__(state)
        if "window_length" not in self.__dict__ and "window_size" in self.__dict__:
            self.window_length = self.window_size
        if "window_size" not in self.__dict__ and "window_length" in self.__dict__:
            self.window_size = DEPRECATED_PARAMETER

    # sklearn API
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Self:
        self._validate_params()
        X = validate_data(
            self, X, y="no_validation", ensure_2d=True, reset=True, dtype=np.float64
        )

        self.window_length_ = resolve_renamed_parameter(
            new_name="window_length",
            new_value=self.window_length,
            new_default=21,
            old_name="window_size",
            old_value=self.window_size,
        )

        if self.window_length_ < 3 or self.window_length_ % 2 == 0:
            raise ValueError("window_length must be an odd integer >= 3.")
        self.kernel_ = self._compute_kernel().astype(np.float64, copy=False)
        if self.kernel_.ndim != 1 or self.kernel_.size != self.window_length_:
            raise ValueError("kernel must be 1D with length equal to window_length.")
        if not np.allclose(self.kernel_.sum(), 1.0, atol=1e-12):
            raise ValueError("kernel must be DC-preserving (sum == 1).")
        self._half_ = (self.window_length_ - 1) // 2
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
