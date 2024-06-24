from typing import Literal, Optional, Tuple, Type, Union

import numpy as np
from sklearn.utils.validation import check_array


def check_input(
    X,
    y=None,
    dtype: Union[Type, Literal["numeric"], None] = "numeric",
):
    # Check that X is a 2D array and has only finite values
    X = check_array(X, ensure_2d=True, force_all_finite=True, dtype=dtype)

    # Check that y is None or a 1D array of the same length as X
    if y is not None:
        y = y.reshape(-1, 1) if y.ndim == 1 else y
        y = check_array(y, force_all_finite=True)
        if len(y) != X.shape[0]:
            raise ValueError("y must have the same number of samples as X")
    return X


def check_weights(
    weights: Optional[np.ndarray],
    n_samples: int,
    n_features: int,
) -> Tuple[Optional[np.ndarray], bool]:
    # if the weights are None, None is returned and a flag that the same weights should
    # be applied for all samples
    if weights is None:
        return None, True

    # if the weights are a 1D array, they are reshaped to a 2D array with one row
    if weights.ndim == 1:
        weights_checked = weights.reshape((1, -1))
    else:
        weights_checked = weights

    # now, the need to be checked for having the right shape
    weights_checked = check_array(
        weights_checked,
        ensure_2d=True,
        force_all_finite=True,
    )

    # afterwards, they are checked for having the right shape
    if weights_checked.shape[0] not in {1, n_samples}:
        raise ValueError(
            f"Weights must have either 1 or {n_samples} rows, but they have "
            f"{weights_checked.shape[0]} rows."
        )
    if weights_checked.shape[1] != n_features:
        raise ValueError(
            f"Weights must have {n_features} columns, but they have "
            f"{weights_checked.shape[1]} columns."
        )

    # finally, it needs to be checked that the weights are all non-negative ...
    if (weights_checked < 0.0).any():
        raise ValueError(
            f"Weights may not be negative, but {(weights_checked < 0.0).sum(axis=1)} "
            f"negative weights were found (one entry per vector)."
        )
    # ... and also at least one of them is positive
    if (weights_checked.sum(axis=1) <= 0.0).any():
        raise ValueError(
            f"At least one weights needs to be > 0, but all weights were 0.0 for "
            f"vector index {np.where(weights_checked.sum(axis=1) <= 0.0)[0]}."
        )

    # the weights are returned together with a flag whether to apply the same weights
    # for all samples or not
    return weights_checked, weights_checked.shape[0] == 1
