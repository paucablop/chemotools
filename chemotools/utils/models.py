from enum import Enum

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
