# Authors: Niklas Zell <nik.zoe@web.de>, Pau Cabaneros
# License: MIT

from abc import abstractmethod
from typing import Callable, Optional

import numpy as np

from chemotools.utils._linear_algebra import whittaker_solver_dispatch


class _BaselineWhittakerMixin:
    """Mixin class for Whittaker-based baseline correction algorithms.

    This mixin provides helper methods for warm-start weights and baseline
    correction logic shared across iterative baseline correction algorithms
    (AirPLS, ArPLS, AsLS, etc.).

    Helper methods provided
    -----------------------
    - `_compute_warmstart_weights(X, solver)`: Computes initial weights from first spectrum.
    - `_apply_baseline_correction(X, solver)`: Applies baseline correction using warm-start.

    Requirements
    ------------
    Subclasses must provide:
    - a `_calculate_baseline(x, w, max_iter, solver)` method returning (baseline, weights).
    - a `_solve_whittaker(x, w, solver)` method (provided by `_BaseWhittaker`).
    - a `w_init_` attribute set during fit (typically from `_compute_warmstart_weights`).

    Parameters
    ----------
    nr_iterations : int, default=100
        Maximum iterations for baseline estimation.
    max_iter_after_warmstart : int, default=20
        Maximum iterations when warm-starting from previous weights.

    Attributes
    ----------
    w_init_ : np.ndarray
        Warm-start weights computed during fit. Set by subclasses in their
        `_fit_core` implementation using `_compute_warmstart_weights`.
    """

    w_init_: np.ndarray  # Type hint for attribute set by subclasses

    def __init__(
        self,
        nr_iterations: int = 100,
        max_iter_after_warmstart: int = 20,
    ):
        self.nr_iterations = nr_iterations
        self.max_iter_after_warmstart = max_iter_after_warmstart

    def _compute_warmstart_weights(
        self,
        X: np.ndarray,
        solver: Optional[Callable] = whittaker_solver_dispatch,
    ) -> np.ndarray:
        """Compute initial weights from the first spectrum.

        This method computes warm-start weights by running the baseline
        estimation algorithm on the first spectrum with uniform initial weights.
        The resulting weights can then be used to accelerate convergence when
        processing subsequent spectra.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input spectra.
        solver : Optional[Callable]
            Whittaker solver function to use.

        Returns
        -------
        weights : np.ndarray of shape (n_features,)
            Computed initial weights for warm-starting subsequent spectra.
        """
        x0 = X[0]
        _, w = self._calculate_baseline(
            x0, np.ones_like(x0), max_iter=self.nr_iterations, solver=solver
        )
        return w

    def _apply_baseline_correction(
        self,
        X: np.ndarray,
        solver: Optional[Callable] = whittaker_solver_dispatch,
    ) -> np.ndarray:
        """Apply baseline correction to all spectra using warm-start weights.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input spectra to correct (will be modified in-place).
        solver : Optional[Callable]
            Whittaker solver function to use.

        Returns
        -------
        X : np.ndarray of shape (n_samples, n_features)
            Baseline-corrected spectra.
        """
        for i, x in enumerate(X):
            z, _ = self._calculate_baseline(
                x,
                self.w_init_.copy(),
                max_iter=min(self.nr_iterations, self.max_iter_after_warmstart),
                solver=solver,
            )
            X[i] = x - z
        return X

    @abstractmethod
    def _calculate_baseline(
        self, x: np.ndarray, w: np.ndarray, max_iter: int, solver: Optional[Callable]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Subclasses must implement algorithm-specific baseline estimation."""
        ...
