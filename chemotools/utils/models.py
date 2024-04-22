"""
This utility submodule implements important models, i.e., constants, Enums, and
dataclasses used throughout the package.

"""

### Imports ###

from dataclasses import dataclass, field
from enum import Enum
from typing import Union

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

    PIVOTED_LU = "partially pivoted LU decomposition"
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


# a dataclass for specification of the smoothing penalty weight lambda for the
# Whittaker-Henderson smoother


@dataclass()
class WhittakerSmoothLambda:
    """
    A dataclass that holds the specification of the smoothing penalty weight or
    smoothing parameter lambda for the Whittaker-Henderson smoother.

    Attributes
    ----------
    low_bound, upp_bound: int or float
        The lower and upper bound of the search space for the penalty weight.
        Flipped bounds are automatically corrected, but they have to differ by at least
        a factor of 10.
    method: AutoSmoothMethods
        The method to use for the automatic selection of the penalty weight.

    Raises
    ------
    ValueError
        If ``upp_bound`` is not greater than 10 times ``low_bound`` after eventually
        flipping the bounds.

    """

    low_bound: Union[int, float]
    upp_bound: Union[int, float]
    method: AutoSmoothMethods

    def __post_init__(self):
        # firs, the input types are checked
        if not isinstance(self.low_bound, (int, float)) or not isinstance(
            self.upp_bound, (int, float)
        ):
            raise TypeError(
                f"\nThe lower bound ({self.low_bound}) and upper bound "
                f"({self.upp_bound}) have to be integers or floats."
            )

        if not isinstance(self.method, AutoSmoothMethods):
            raise TypeError(
                f"\nThe method ({self.method}) has to be a member of the "
                f"AutoSmoothMethods."
            )

        # then, the lower and upper bound are sanitized by swapping them if necessary
        # and checking if the upper bound is at least 10 times the lower bound
        if self.low_bound >= self.upp_bound:
            self.low_bound, self.upp_bound = self.upp_bound, self.low_bound

        if self.upp_bound < 10 * self.low_bound:
            raise ValueError(
                f"\nThe upper bound ({self.upp_bound}) has to be at least 10 times the "
                f"lower bound ({self.low_bound})."
            )


# a fake class for representing the factorization of a pentadiagonal matrix with
# pentapy which is empty since pentapy does not factorize the matrix but directly solves
# the system of equations


class BandedPentapyFactorization:
    """
    A class that resembles the factorization of a pentadiagonal matrix with ``pentapy``.
    It has no attributes since the factorization is not stored, but the class is used to
    provide an easy way to check if the factorization is available.

    """

    pass


# a dataclass for the factorization of a banded matrix with LU decomposition with p
# partial pivoting


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
