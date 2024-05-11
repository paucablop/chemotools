"""
This submodule contains miscellaneous functions used by ``WhittakerLikeSolver`` class
that would have cluttered the class implementation.

"""

### Imports ###

from typing import Any, Generator, Union

import numpy as np

### Functions ###


def get_weight_generator(
    w: Any,
    n_series: int,
) -> Generator[Union[float, np.ndarray], None, None]:
    """
    Generates a generator that yields the weights for each series in a series matrix
    ``X``.

    """

    # if the weights are neither None, nor a 1D- or a 2D-Array, an error is raised
    if not (w is None or isinstance(w, np.ndarray)):
        raise TypeError(
            f"The weights must either be None, a NumPy-1D-, or a NumPy-2D-Array, but "
            f"they are of type '{type(w)}'."
        )

    # Case 1: No weights
    if w is None:
        for _ in range(n_series):
            yield 1.0

    # Case 2: 1D or 2D weights
    elif w.ndim == 1:
        for _ in range(n_series):
            yield w

    # Case 3: 2D weights
    elif w.ndim == 2:
        for idx in range(0, n_series):
            yield w[idx]

    # Case 4: Invalid weights
    elif w.ndim > 2:
        raise ValueError(
            f"The weights must be either a 1D- or a 2D-array, but they are "
            f"{w.ndim}-dimensional with shape {w.shape}."
        )
