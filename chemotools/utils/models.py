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

# an Enum class for the decomposition types used for solving linear systems that involve
# banded matrices


class BandedSolveDecompositions(str, Enum):
    CHOLESKY = "cholesky"
    PENTAPY = "pentapy"


@dataclass()
class BandedLUFactorization:
    lub: np.ndarray
    ipiv: np.ndarray
    l_and_u: tuple[int, int]
    singular: bool

    shape: tuple[int, int] = field(default=(-1, -1), init=False)

    def __post_init__(self):
        self.shape = self.lub.shape  # type: ignore
