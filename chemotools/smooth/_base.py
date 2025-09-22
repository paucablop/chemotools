# _base.py
# Authors: Niklas Zell <nik.zoe@web.de>, Pau Cabaneros
# License: MIT

from __future__ import annotations
from abc import ABC, abstractmethod
import logging
from typing import Callable, Literal, Optional
from typing_extensions import Self

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
    ) -> "Self":
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
        self, x: np.ndarray, w: np.ndarray, solver: Callable
    ) -> np.ndarray:
        """Solve (diag(w) + lam*D^T D) z = w*x."""
        try:
            return solver(x, w, self.lam, self.DtD_)
        except Exception as e:
            logger.debug("Primary solver failed (%s); fallback to sparse LU.", e)
            DtD = compute_DtD_sparse(len(x))
            return whittaker_smooth_sparse(x, w, self.lam, DtD)
