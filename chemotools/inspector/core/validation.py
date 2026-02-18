from typing import Optional, Tuple, Union

import numpy as np
from sklearn.cross_decomposition._pls import _PLS
from sklearn.decomposition._base import _BasePCA
from sklearn.pipeline import Pipeline

from chemotools._validation import validate_and_extract_model as _canonical_validate


def _validate_and_extract_model(
    model: Union[_BasePCA, _PLS, Pipeline],
) -> Tuple[Union[_BasePCA, _PLS], Optional[Pipeline]]:
    """Validate and extract model and transformer.

    Delegates to :func:`chemotools._validation.validate_and_extract_model`.

    Parameters
    ----------
    model : Union[_BasePCA, _PLS, Pipeline]
        The model to validate

    Returns
    -------
    estimator : Union[_BasePCA, _PLS]
        The extracted estimator

    transformer : Optional[Pipeline]
        The preprocessing pipeline (if model is a Pipeline)
    """
    return _canonical_validate(model)


def _validate_sample_length(name: str, y: Optional[np.ndarray], expected: int) -> None:
    """
    Validate that y has the expected number of samples."""
    if y is None:
        return
    y_arr = np.asarray(y)
    actual = y_arr.shape[0] if y_arr.ndim > 0 else 1
    if actual != expected:
        raise ValueError(
            f"{name} must have the same number of samples as its X data. "
            f"Got {actual} vs {expected}."
        )


def _validate_datasets_consistency(
    X_train: np.ndarray,
    y_train: Optional[np.ndarray],
    X_test: Optional[np.ndarray],
    y_test: Optional[np.ndarray],
    X_val: Optional[np.ndarray],
    y_val: Optional[np.ndarray],
    supervised: bool = False,
) -> None:
    """Validate that datasets have consistent shapes."""
    n_features = X_train.shape[1]
    n_train = X_train.shape[0]

    if X_test is not None and X_test.shape[1] != n_features:
        raise ValueError("X_test must have same number of features as X_train")

    if X_val is not None and X_val.shape[1] != n_features:
        raise ValueError("X_val must have same number of features as X_train")

    _validate_sample_length("y_train", y_train, n_train)

    if supervised:
        if y_train is None:
            raise ValueError("y_train required for supervised models")
        if X_test is not None and y_test is None:
            raise ValueError("y_test required when X_test is provided")
        if X_val is not None and y_val is None:
            raise ValueError("y_val required when X_val is provided")

    if X_test is not None:
        _validate_sample_length("y_test", y_test, X_test.shape[0])

    if X_val is not None:
        _validate_sample_length("y_val", y_val, X_val.shape[0])
