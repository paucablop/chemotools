# Authors: Niklas Zell <nik.zoe@web.de>, Pau Cabaneros
# License: MIT

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import scipy.sparse as sp

from chemotools.utils._linear_algebra import (
    whittaker_smooth_banded,
    whittaker_smooth_banded_batch,
    whittaker_smooth_sparse,
)


class WhittakerSolver(ABC):
    """Abstract base class for Whittaker system solvers.

    Each solver owns the regularization parameter and precomputed penalty
    matrix, so call sites only need to supply the signal and weights.

    Parameters
    ----------
    lam : float
        Regularization parameter.
    DtD : array-like
        Precomputed penalty matrix in the format expected by the concrete solver.
    """

    def __init__(self, lam: float, DtD) -> None:
        self.lam = lam
        self.DtD = DtD

    @abstractmethod
    def solve(self, x: np.ndarray, w: np.ndarray) -> np.ndarray:
        """Solve the Whittaker system for a single row.

        Parameters
        ----------
        x : ndarray of shape (n_features,)
            Input signal.
        w : ndarray of shape (n_features,)
            Observation weights.

        Returns
        -------
        z : ndarray of shape (n_features,)
            Smoothed signal.
        """
        ...

    def solve_batch(self, X: np.ndarray, w: np.ndarray) -> np.ndarray:
        """Solve the Whittaker system for all rows of X.

        The default implementation applies ``solve`` row by row, which is
        always correct. Subclasses may override this with a vectorised
        implementation when the system matrix is shared across all rows.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data matrix.
        w : ndarray of shape (n_features,)
            Observation weights, identical for every row.

        Returns
        -------
        Z : ndarray of shape (n_samples, n_features)
            Smoothed output matrix.
        """
        return np.vstack([self.solve(x, w) for x in X])


class BandedSolver(WhittakerSolver):
    """Whittaker solver using a banded Cholesky factorization.

    Parameters
    ----------
    lam : float
        Regularization parameter.
    DtD : ndarray of shape (3, n_features)
        Banded representation of DᵀD in upper form.
    """

    def __init__(self, lam: float, DtD: np.ndarray) -> None:
        super().__init__(lam, DtD)

    def solve(self, x: np.ndarray, w: np.ndarray) -> np.ndarray:
        return whittaker_smooth_banded(x, w, self.lam, self.DtD)

    def solve_batch(self, X: np.ndarray, w: np.ndarray) -> np.ndarray:
        return whittaker_smooth_banded_batch(X, w, self.lam, self.DtD)


class SparseSolver(WhittakerSolver):
    """Whittaker solver using a sparse LU decomposition.

    Parameters
    ----------
    lam : float
        Regularization parameter.
    DtD : scipy.sparse.csc_matrix of shape (n_features, n_features)
        Sparse penalty matrix DᵀD.
    """

    def __init__(self, lam: float, DtD: sp.csc_matrix) -> None:
        super().__init__(lam, DtD)

    def solve(self, x: np.ndarray, w: np.ndarray) -> np.ndarray:
        return whittaker_smooth_sparse(x, w, self.lam, self.DtD)


def whittaker_solver_factory(solver_type: str, lam: float, DtD) -> WhittakerSolver:
    """Instantiate the appropriate ``WhittakerSolver`` for the given solver_type.

    Parameters
    ----------
    solver_type : {"banded", "sparse"}
        Which solver to create.
    lam : float
        Regularization parameter.
    DtD : array-like
        Precomputed penalty matrix in the format expected by the solver.

    Returns
    -------
    solver : WhittakerSolver
    """
    if solver_type == "banded":
        return BandedSolver(lam, DtD)
    return SparseSolver(lam, DtD)
