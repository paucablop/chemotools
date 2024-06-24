"""
This submodule contains miscellaneous functions used by ``WhittakerLikeSolver`` class
that would have cluttered the class implementation.

"""

### Imports ###

from typing import Any, Generator, Union

import numpy as np

### Functions ###


def get_weight_generator(
    weights: Any,
    num_series: int,
) -> Generator[Union[float, np.ndarray], None, None]:
    """
    Generates a generator that yields the weights for each series in a series matrix
    ``X``.

    """

    # if the weights are neither None nor a 2D-Array, an error is raised
    if not (weights is None or isinstance(weights, np.ndarray)):
        raise TypeError(
            f"The weights must either be None or a NumPy-2D-Array, but they are of "
            f"type '{type(weights)}'."
        )

    # Case 1: No weights
    if weights is None:
        for _ in range(num_series):
            yield 1.0

    # Case 2: 2D weights
    elif weights.ndim == 2:
        for idx in range(0, num_series):
            yield weights[idx]

    # Case 3: Invalid weights
    elif weights.ndim != 2:
        raise ValueError(
            f"If provided as an Array, the weights must be a 2D-Array, but they are "
            f"{weights.ndim}-dimensional with shape {weights.shape}."
        )
