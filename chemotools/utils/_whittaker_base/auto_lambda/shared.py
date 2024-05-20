"""
This submodule contains the shared logics when it comes to the automated fitting of the
penalty weight lambda within the ``WhittakerLikeSolver`` class that would have cluttered
the class implementation.

"""

### Imports ###

from typing import Union

import numpy as np

from chemotools.utils import _models

### Type Aliases ###

_Factorization = Union[
    _models.BandedLUFactorization, _models.BandedPentapyFactorization
]

### Functions ###


def get_smooth_wrss(
    rhs_b: np.ndarray,
    rhs_b_smooth: np.ndarray,
    weights: Union[float, np.ndarray],
) -> float:
    """
    Computes the (weighted) Sum of Squared Residuals (w)RSS between the original and
    the smoothed series.

    """

    # Case 1: no weights are provided
    if isinstance(weights, float):
        return np.square(rhs_b - rhs_b_smooth).sum()

    # Case 2: weights are provided
    return (weights * np.square(rhs_b - rhs_b_smooth)).sum()
