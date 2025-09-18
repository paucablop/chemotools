# Authors: Niklas Zell <nik.zoe@web.de>, Pau Cabaneros
# License: MIT

from typing import Literal

import numpy as np
from scipy.linalg import solveh_banded
import scipy.sparse as sp
from scipy.sparse.linalg import splu


def compute_DtD_banded(N: int) -> np.ndarray:
    """
    Construct the banded representation of the penalty matrix DᵀD.

    This precomputes the upper-banded form (with bandwidth u=2) of the
    second-order difference penalty matrix, required by the Whittaker
    banded solver.

    Parameters
    ----------
    N : int
        Length of the input signal.

    Returns
    -------
    ab : ndarray of shape (3, N)
        Banded representation of DᵀD in upper form.
    """
    if N < 3:
        return np.zeros((3, N))

    if N >= 5:
        DtD_main = np.concatenate(([1, 5], np.repeat(6, N - 4), [5, 1]))
    elif N == 4:
        DtD_main = np.array([1, 5, 5, 1])
    else:  # N == 3
        DtD_main = np.array([1, 5, 1])

    if N == 3:
        DtD_sup1 = np.array([-2, -2])
    elif N == 4:
        DtD_sup1 = np.array([-2, -4, -2])
    else:
        DtD_sup1 = np.concatenate(([-2], np.repeat(-4, N - 3), [-2]))

    DtD_sup2 = np.ones(N - 2)

    ab = np.zeros((3, N))
    ab[0, 2:] = DtD_sup2
    ab[1, 1:] = DtD_sup1
    ab[2, :] = DtD_main
    return ab


def compute_DtD_sparse(N: int) -> sp.csc_matrix:
    """
    Construct the sparse representation of the penalty matrix DᵀD.

    Builds D as a second-order difference operator, then computes
    DᵀD in compressed sparse column (CSC) format.

    Parameters
    ----------
    N : int
        Length of the input signal.

    Returns
    -------
    DtD : scipy.sparse.csc_matrix of shape (N, N)
        Sparse penalty matrix.
    """
    if N < 3:
        return sp.csc_matrix((N, N))
    D = sp.diags([1, -2, 1], [0, 1, 2], shape=(N - 2, N), format="csc")
    return D.T @ D


def whittaker_smooth_banded(x, w, lam, DtD_ab):
    """
    Solve the Whittaker smoothing system using a banded solver.

    Efficiently solves:
        (diag(w) + λ DᵀD) z = w ⊙ x
    where D is the second-order difference operator.

    Parameters
    ----------
    x : ndarray of shape (N,)
        Input signal.

    w : ndarray of shape (N,)
        Observation weights.

    lam : float
        Regularization parameter.

    DtD_ab : ndarray of shape (3, N)
        Banded representation of DᵀD.

    Returns
    -------
    z : ndarray of shape (N,)
        Smoothed signal (baseline estimate).
    """
    ab = np.empty_like(DtD_ab)
    ab[...] = DtD_ab
    ab[2, :] = lam * ab[2, :] + w  # main diag updated
    ab[1, 1:] = lam * ab[1, 1:]  # superdiag
    ab[0, 2:] = lam * ab[0, 2:]  # 2nd superdiag
    return solveh_banded(ab, w * x, lower=False, overwrite_ab=True, overwrite_b=True)


def whittaker_smooth_sparse(x, w, lam, DtD_sparse):
    """
    Solve the Whittaker smoothing system using a sparse LU decomposition.

    This is a more stable but slower fallback solver for ill-conditioned
    problems, solving:
        (diag(w) + λ DᵀD) z = w ⊙ x

    Parameters
    ----------
    x : ndarray of shape (N,)
        Input signal.

    w : ndarray of shape (N,)
        Observation weights.

    lam : float
        Regularization parameter.

    DtD_sparse : scipy.sparse.csc_matrix of shape (N, N)
        Sparse penalty matrix DᵀD.

    Returns
    -------
    z : ndarray of shape (N,)
        Smoothed signal (baseline estimate).
    """
    N = len(x)
    H = lam * DtD_sparse
    W = sp.diags(w, 0, shape=(N, N), format="csc")
    C = (H + W).tocsc()
    solver = splu(C)
    return solver.solve(w * x)


def whittaker_solver_dispatch(solver_type: Literal["banded", "sparse"]):
    """
    Dispatch Whittaker solver based on solver type.

    Parameters
    ----------
    solver_type : {"banded", "sparse"}
        Which solver to use:
        - "banded": fast, memory-efficient solver (recommended).
        - "sparse": sparse LU solver, more stable for ill-conditioned systems.

    Returns
    -------
    solver : callable
        Function that solves the Whittaker system given x, w, lam, and DᵀD.
    """
    if solver_type == "banded":
        return whittaker_smooth_banded
    elif solver_type == "sparse":
        return whittaker_smooth_sparse
    else:
        raise ValueError(f"Unknown solver_type: {solver_type}")
