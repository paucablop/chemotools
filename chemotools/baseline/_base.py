# Authors: Niklas Zell <nik.zoe@web.de>, Pau Cabaneros
# License: MIT

from abc import abstractmethod
from typing import Callable
from typing_extensions import Literal, Self

import numpy as np

from chemotools.utils._linear_algebra import whittaker_solver_dispatch


class _BaselineWhittakerMixin:
    """Mixin class for Whittaker-based baseline correction.

    This mixin handles warm-start weights and iteration control for
    baseline correction algorithms.

    Requirements
    ------------
    Subclasses must provide:
    - a `_calculate_baseline(x, w, max_iter)` method returning (baseline, weights).
    - a `_solve_whittaker(x, w)` method (provided by `_BaseWhittaker`).

    Parameters expected
    -------------------
    nr_iterations : int, default=100
        Maximum iterations for baseline estimation.
    max_iter_after_warmstart : int, default=20
        Maximum iterations when warm-starting.
    """

    def __init__(
        self,
        solver_type: Literal["banded", "sparse"] = "banded",
        nr_iterations: int = 100,
        max_iter_after_warmstart: int = 20,
    ):
        self.solver_type = solver_type
        self.nr_iterations = nr_iterations
        self.max_iter_after_warmstart = max_iter_after_warmstart

    def _fit_core(
        self,
        X: np.ndarray,
        y=None,
        nr_iterations: int = 1,
    ) -> "Self":
        # Warm start weights from first spectrum
        x0 = X[0]
        solver = whittaker_solver_dispatch(self.solver_type)
        _, w = self._calculate_baseline(
            x0, np.ones_like(x0), max_iter=self.nr_iterations, solver=solver
        )
        self.w_init_ = w
        return self

    def _transform_core(
        self, X: np.ndarray, y=None, nr_iterations: int = 1
    ) -> np.ndarray:
        for i, x in enumerate(X):
            z, _ = self._calculate_baseline(
                x,
                self.w_init_.copy(),
                max_iter=min(self.nr_iterations, self.max_iter_after_warmstart),
                solver=whittaker_solver_dispatch(self.solver_type),
            )
            X[i] = x - z
        return X

    @abstractmethod
    def _calculate_baseline(
        self, x: np.ndarray, w: np.ndarray, max_iter: int, solver: Callable
    ) -> tuple[np.ndarray, np.ndarray]:
        """Subclasses must implement algorithm-specific baseline estimation."""
        ...
