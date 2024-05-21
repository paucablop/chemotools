"""
The utility module of ``chemotools`` that offers access to various utility functions
that can come in handy when working with chemical data.

The module contains the following functions:

- :func:`estimate_noise_stddev` to estimate the local/global noise level of a spectrum
    which can then be used for weighting the data.

"""

### Imports ###

from chemotools.utils._finite_differences import estimate_noise_stddev  # noqa: F401
