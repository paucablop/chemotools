"""
This submodule contains miscellaneous functions used by ``WhittakerLikeSolver`` class
that would have cluttered the class implementation.

"""

### Imports ###

from typing import Generator, Optional, Union

import numpy as np

### Functions ###


def get_weight_generator(
    w: Optional[np.ndarray],
    n_series: int,
) -> Generator[Union[float, np.ndarray], None, None]:
    """
    Generates a generator that yields the weights for each series in a series matrix
    ``X``.

    """

    # Case 1: No weights
    if w is None:
        for _ in range(n_series):
            yield 1.0

    # Case 2: 1D weights
    elif w.ndim == 1:
        for _ in range(n_series):
            yield w

    # Case 3: 2D weights
    elif w.ndim == 2:
        for w_vect in w:
            yield w_vect
