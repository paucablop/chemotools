"""
This submodule contains the solver functions used by the ``WhittakerLikeSolver`` class
that would have cluttered the class implementation.

"""

### Imports ###


from typing import Union

import numpy as np

from chemotools._runtime import PENTAPY_AVAILABLE
from chemotools.utils import _banded_linalg as bla
from chemotools.utils import _models

if PENTAPY_AVAILABLE:
    import pentapy as pp

### Type Aliases ###

_Factorization = Union[
    _models.BandedLUFactorization, _models.BandedPentapyFactorization
]

### Functions ###


def solve_pentapy(
    lhs_a_banded: np.ndarray,
    rhs_b_weighted: np.ndarray,
) -> np.ndarray:
    """
    Solves the linear system of equations ``(W + lam * D.T @ D) @ x = W @ b`` with the
    ``pentapy`` package. This is the same as solving the linear system ``A @ x = b``
    where ``A = W + lam * D.T @ D`` and ``b = W @ b``.

    Notes
    -----
    Pentapy does not (maybe yet) allow for 2D right-hand side matrices, so the
    solution is computed for each column of ``bw`` separately.

    """

    # for 1-dimensional right-hand side vectors, the solution is computed directly
    if rhs_b_weighted.ndim == 1:
        return pp.solve(
            mat=lhs_a_banded,
            rhs=rhs_b_weighted,
            is_flat=True,
            index_row_wise=False,
            solver=1,
        )

    # for 2-dimensional right-hand side matrices, the solution is computed for each
    # column separately
    else:
        # NOTE: the solutions are first written into the rows of the solution matrix
        #       because row-access is more efficient for C-contiguous arrays;
        #       afterwards, the solution matrix is transposed
        solution = np.empty(shape=(rhs_b_weighted.shape[1], rhs_b_weighted.shape[0]))
        for iter_j in range(0, rhs_b_weighted.shape[1]):
            solution[iter_j, ::] = pp.solve(
                mat=lhs_a_banded,
                rhs=rhs_b_weighted[::, iter_j],
                is_flat=True,
                index_row_wise=False,
                solver=1,
            )

        return solution.transpose()


def solve_ppivoted_lu(
    l_and_u: bla.LAndUBandCounts,
    lhs_a_banded: np.ndarray,
    rhs_b_weighted: np.ndarray,
) -> tuple[np.ndarray, _models.BandedLUFactorization]:
    """
    Solves the linear system of equations ``(W + lam * D.T @ D) @ x = W @ b`` with a
    partially pivoted LU decomposition. This is the same as solving the linear system
    ``A @ x = b`` where ``A = W + lam * D.T @ D`` and ``b = W @ b``.

    If the LU decomposition fails, a ``LinAlgError`` is raised which is fatal since
    the next level of escalation would be using a QR-decomposition which is not
    implemented (yet).

    """

    lub_factorization = bla.lu_banded(
        l_and_u=l_and_u,
        ab=lhs_a_banded,
        check_finite=False,
    )
    return (
        bla.lu_solve_banded(
            lub_factorization=lub_factorization,
            b=rhs_b_weighted,
            check_finite=False,
            overwrite_b=True,
        ),
        lub_factorization,
    )


def solve_normal_equations(
    lam: float,
    differences: int,
    l_and_u: bla.LAndUBandCounts,
    penalty_mat_banded: np.ndarray,
    rhs_b_weighted: np.ndarray,
    weights: Union[float, np.ndarray],
    pentapy_enabled: bool,
) -> tuple[np.ndarray, _models.BandedSolvers, _Factorization]:
    """
    Solves the linear system of equations ``(W + lam * D.T @ D) @ x = W @ b`` where
    ``W`` is a diagonal matrix with the weights ``w`` on the main diagonal and ``D`` is
    the finite difference matrix of order ``differences``. ``lam`` represents the
    penalty weight for the smoothing.
    For details on why the system is not formulated in a more efficient way, please
    refer to the Notes section.

    Parameters
    ----------
    lam : float
        The penalty weight lambda to use for the smoothing.
    differences : int
        The order of the finite differences to use for the smoothing.
    l_and_u : LAndUBandCounts
        The number of sub- and super-diagonals of ``penalty_mat_banded``.
    penalty_mat_banded : ndarray of shape (2 * differences + 1, m)
        The penalty matrix ``D.T @ D`` in the banded storage format used for LAPACK's
        banded LU decomposition.
    b_weighted : ndarray of shape (m,) or (m, n)
        The weighted right-hand side vector or matrix of the linear system of equations
        given by ``W @ b``.
    w : float or ndarray of shape (m,)
        The weights to use for the linear system of equations given in terms of the main
        diagonal of the weight matrix ``W``.
        It can either be a vector of weights for each data point or a single scalar -
        namely ``1.0`` - if no weights are provided.
    pentapy_enabled : bool
        Determines whether the ``pentapy`` solver is enabled (``True``) or not
        (``False``).

    Returns
    -------
    x : np.ndarray of shape (m,)
        The solution vector of the linear system of equations.
    decomposition_type : BandedSolveDecompositions
        The type of decomposition used to solve the linear system of equations.
    decomposition : BandedLUFactorization or BandedPentapyFactorization
        The decomposition used to solve the linear system of equations which is stored
        as a class instance specifying everything required to solve the system with
        the ``decomposition_type`` used.

    Raises
    ------
    RuntimeError
        If all available solvers failed to solve the linear system of equations which
        indicates a highly ill-conditioned system.

    Notes
    -----
    It might seem more efficient to solve the linear system ``((1.0 / lam) * W + D.T @ D) @ x = (1.0 / lam) * W @ b``
    because this only requires a multiplication of ``m`` weights with the reciprocal of
    the penalty weight whereas the multiplication with ``D.T @ D`` requires roughly
    ``m * (1 + 2 * differences)`` multiplications with ``m`` as the number of data
    points and ``differences`` as the difference order. On top of that, ``m * differences``
    multiplications - so roughly 50% - would be redundant given that the penalty
    ``D.T @ D`` matrix is symmetric.
    However, NumPy's scalar multiplication is so highly optimized that the
    multiplication with ``D.T @ D`` without considering symmetry is almost as fast as
    the multiplication with the diagonal matrix ``W``, especially when compared to the
    computational load of the banded solvers.

    """  # noqa: E501

    # the banded storage format for the LAPACK LU decomposition is computed by
    # scaling the penalty matrix with the penalty weight lambda and then adding the
    # diagonal matrix with the weights
    lhs_a_banded = lam * penalty_mat_banded
    lhs_a_banded[differences, ::] += weights

    # the linear system of equations is solved with the most efficient method
    # Case 1: Pentapy can be used
    if pentapy_enabled:
        x = solve_pentapy(
            lhs_a_banded=lhs_a_banded,
            rhs_b_weighted=rhs_b_weighted,
        )
        if np.isfinite(x).all():
            return (
                x,
                _models.BandedSolvers.PENTAPY,
                _models.BandedPentapyFactorization(),
            )

    # Case 2: LU decomposition (final fallback for pentapy)
    try:
        x, lub_factorization = solve_ppivoted_lu(
            l_and_u=l_and_u,
            lhs_a_banded=lhs_a_banded,
            rhs_b_weighted=rhs_b_weighted,
        )
        return (
            x,
            _models.BandedSolvers.PIVOTED_LU,
            lub_factorization,
        )

    except np.linalg.LinAlgError:
        available_solvers = f"{_models.BandedSolvers.PIVOTED_LU}"
        if pentapy_enabled:
            available_solvers = f"{_models.BandedSolvers.PENTAPY}, {available_solvers}"

        raise RuntimeError(
            f"\nAll available solvers ({available_solvers}) failed to solve the "
            f"linear system of equations which indicates a highly ill-conditioned "
            f"system.\n"
            f"Please consider reducing the number of data points to smooth by, "
            f"e.g., binning or lowering the difference order."
        )
