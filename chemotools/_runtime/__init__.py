"""
This submodule checks for the presence of the required software packages at runtime.

The following optional packages are checked for:
- `pentapy` for solving pentadiagonal systems of equations for the Whittaker-Henderson
    smoothing algorithm.

"""

### Imports ###

# if possible, pentapy is imported since it provides a more efficient implementation
# of solving pentadiagonal systems of equations, but the package is not in the
# dependencies, so ``chemotools`` needs to be made aware of whether it is available
PENTAPY_AVAILABLE: bool = False
try:
    import pentapy as pp  # noqa: F401

    PENTAPY_AVAILABLE: bool = True
except ImportError:
    pass
