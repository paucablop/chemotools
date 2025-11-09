from typing import Union, Tuple, Optional

import numpy as np

from sklearn.decomposition._base import _BasePCA
from sklearn.cross_decomposition._pls import _PLS
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted


def _validate_model(
    model: Union[_BasePCA, _PLS, Pipeline],
) -> Union[_BasePCA, _PLS, Pipeline]:
    """Validate model without extracting components.

    Parameters
    ----------
    model : Union[_BasePCA, _PLS, Pipeline]
        The model to validate

    Returns
    -------
    model : Union[_BasePCA, _PLS, Pipeline]
        The validated model (same as input)

    Raises
    ------
    TypeError
        If model is not a valid type
    ValueError
        If model is not fitted
    """
    check_is_fitted(model)

    # Determine the estimator
    if isinstance(model, Pipeline):
        estimator = model[-1]
    else:
        estimator = model

    if not isinstance(estimator, (_BasePCA, _PLS)):
        raise TypeError(
            f"Model must be _BasePCA, _PLS, or Pipeline ending with one. "
            f"Got {type(estimator)}"
        )

    return model


def _validate_and_extract_model(
    model: Union[_BasePCA, _PLS, Pipeline],
) -> Tuple[Union[_BasePCA, _PLS], Optional[Pipeline]]:
    """Validate and extract model and transformer.

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

    Raises
    ------
    TypeError
        If model is not a valid type
    ValueError
        If model is not fitted
    """
    check_is_fitted(model)

    if isinstance(model, Pipeline):
        estimator = model[-1]
        # Create transformer from all but last step
        if len(model) > 1:
            transformer = Pipeline(model.steps[:-1])
        else:
            transformer = None
    else:
        estimator = model
        transformer = None

    if not isinstance(estimator, (_BasePCA, _PLS)):
        raise TypeError(
            f"Model must be _BasePCA, _PLS, or Pipeline ending with one. "
            f"Got {type(estimator)}"
        )

    return estimator, transformer


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

    if X_test is not None and X_test.shape[1] != n_features:
        raise ValueError("X_test must have same number of features as X_train")

    if X_val is not None and X_val.shape[1] != n_features:
        raise ValueError("X_val must have same number of features as X_train")

    if supervised:
        if y_train is None:
            raise ValueError("y_train required for supervised models")
        if X_test is not None and y_test is None:
            raise ValueError("y_test required when X_test is provided")
        if X_val is not None and y_val is None:
            raise ValueError("y_val required when X_val is provided")
