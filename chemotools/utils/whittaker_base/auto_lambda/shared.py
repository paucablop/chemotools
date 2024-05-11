"""
This submodule contains the shared logics when it comes to the automated fitting of the
penalty weight lambda within the ``WhittakerLikeSolver`` class that would have cluttered
the class implementation.

"""

### Imports ###

from typing import Union

import numpy as np

from chemotools.utils import models

### Type Aliases ###

_Factorization = Union[models.BandedLUFactorization, models.BandedPentapyFactorization]

### Functions ###


def get_smooth_wrss(
    b: np.ndarray,
    b_smooth: np.ndarray,
    w: Union[float, np.ndarray],
) -> float:
    """
    Computes the (weighted) Sum of Squared Residuals (w)RSS between the original and
    the smoothed series.

    """

    # Case 1: no weights are provided
    if isinstance(w, float):
        return np.square(b - b_smooth).sum()

    # Case 2: weights are provided
    return (w * np.square(b - b_smooth)).sum()
