"""
This utility submodule implements important models, i.e., constants, Enums, and
dataclasses used throughout the package.

"""

### Imports ###

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

# if possible, pentapy is imported since it provides a more efficient implementation
# of solving pentadiagonal systems of equations, but the package is not in the
# dependencies, so ``chemotools`` needs to be made aware of whether it is available
try:
    import pentapy as pp  # noqa: F401

    _PENTAPY_AVAILABLE = True
except ImportError:
    _PENTAPY_AVAILABLE = False


### Enums ###

# an Enum class for the solve types used for solving linear systems that involve banded
# matrices


class BandedSolvers(str, Enum):
    """
    Defines the types of solvers that can be used to solve linear systems involving
    banded matrices, i.e.,

    - ``PIVOTED_LU``: LU decomposition with partial pivoting
    - ``PENTAPY``: pentadiagonal "decomposition" (it's actually a direct solve)

    """

    PIVOTED_LU = "pivoted LU decomposition"
    PENTAPY = "direct pentadiagonal solver"


# an Enum class for the kinds of automated smoothing by the Whittaker-Henderson smoother
# that can be applied to the data


class AutoSmoothMethods(str, Enum):
    """
    Defines the types of automated smoothing methods that can be applied to the data
    using the Whittaker-Henderson smoother, i.e.,

    - ``LOG_MARGINAL_LIKELIHOOD``: smoothing based on the maximization of the log
        marginal likelihood

    """

    LOG_MARGINAL_LIKELIHOOD = "log marginal likelihood"


### (Data) Classes ###


class BandedPentapyFactorization:
    """
    A class that resembles the factorization of a pentadiagonal matrix with ``pentapy``.
    It has no attributes since the factorization is not stored, but the class is used to
    provide an easy way to check if the factorization is available.

    """

    pass


@dataclass()
class BandedLUFactorization:
    """
    A dataclass that holds the partially pivoted LU factorization of a banded matrix.

    Attributes
    ----------
    lub: ndarray of shape (n_rows, n_cols)
        The LU factorization of the matrix ``A`` in banded storage format.
    ipiv: ndarray of shape (n_rows,)
        The pivot indices.
    l_and_u: tuple[int, int]
        The number of lower and upper bands in the LU factorization.
    singular: bool
        If ``True``, the matrix ``A`` is singular.
    shape : (int, int)
        The shape of the matrix ``A`` in dense form.
    n_rows, n_cols : int
        The number of rows and columns of the matrix ``A`` in dense form.
    main_diag_row_idx : int
        The index of the main diagonal in the banded storage format.

    """

    lub: np.ndarray
    ipiv: np.ndarray
    l_and_u: tuple[int, int]
    singular: bool

    shape: tuple[int, int] = field(default=(-1, -1), init=False)
    n_rows: int = field(default=-1, init=False)
    n_cols: int = field(default=-1, init=False)
    main_diag_row_idx: int = field(default=-1, init=False)

    def __post_init__(self):
        self.shape = self.lub.shape  # type: ignore
        self.n_rows, self.n_cols = self.shape
        self.main_diag_row_idx = self.l_and_u[1]
