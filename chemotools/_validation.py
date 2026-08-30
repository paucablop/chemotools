"""Shared model validation and parameter extraction utilities."""

from typing import Optional, Tuple, Type, Union

from sklearn.cross_decomposition._pls import _PLS
from sklearn.decomposition._base import _BasePCA
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from ._types import EstimatorType


def validate_and_extract_model(
    model: Union[EstimatorType, Pipeline],
    *,
    require_fitted: bool = True,
    allowed_types: Tuple[Type, ...] = (_BasePCA, _PLS),
) -> Tuple[EstimatorType, Optional[Pipeline]]:
    """Validate a model and extract the estimator and optional preprocessing pipeline.

    Parameters
    ----------
    model : Union[EstimatorType, Pipeline]
        A fitted PCA/PLS model or a Pipeline ending with such a model.

    require_fitted : bool, default=True
        If ``True``, raise an error when the model is not fitted.

    allowed_types : tuple of types, default=(_BasePCA, _PLS)
        The estimator types that are permitted. Pass a narrower tuple
        (e.g. ``(_PLS,)``) to enforce stricter constraints.

    Returns
    -------
    estimator : EstimatorType
        The extracted estimator.

    transformer : Optional[Pipeline]
        The preprocessing steps before the estimator, or ``None`` when the
        input was not a Pipeline (or the Pipeline had only one step).

    Raises
    ------
    TypeError
        If the (final) estimator is not an instance of *allowed_types*.
    sklearn.exceptions.NotFittedError
        If *require_fitted* is ``True`` and the model is not fitted.
    """
    if require_fitted:
        check_is_fitted(model)

    if isinstance(model, Pipeline):
        estimator = model[-1]
        transformer = Pipeline(model.steps[:-1]) if len(model) > 1 else None
    else:
        estimator = model
        transformer = None

    if not isinstance(estimator, allowed_types):
        allowed_names = ", ".join(t.__name__ for t in allowed_types)
        raise TypeError(
            f"Model must be {allowed_names}, or a Pipeline ending with one of "
            f"these types. Got {type(estimator).__name__}."
        )

    return estimator, transformer


def get_model_parameters(estimator: EstimatorType) -> Tuple[int, int, int]:
    """Extract common dimensionality parameters from a fitted estimator.

    Parameters
    ----------
    estimator : EstimatorType
        A fitted ``_BasePCA`` or ``_PLS`` instance.

    Returns
    -------
    n_features_in : int
        Number of features seen during fit.

    n_components : int
        Number of latent components.

    n_samples : int
        Number of training samples.

    Raises
    ------
    TypeError
        If *estimator* is neither ``_BasePCA`` nor ``_PLS``.
    """
    if isinstance(estimator, _BasePCA):
        return estimator.n_features_in_, estimator.n_components_, estimator.n_samples_  # type: ignore[ty:unresolved-attribute]  # sklearn fitted attributes
    if isinstance(estimator, _PLS):
        return (
            estimator.n_features_in_,  # type: ignore[ty:unresolved-attribute]  # sklearn fitted attribute
            estimator.n_components,
            len(estimator.x_scores_),  # type: ignore[ty:unresolved-attribute]  # sklearn fitted attribute
        )
    raise TypeError(
        f"Cannot extract parameters from {type(estimator).__name__}. "
        "Expected _BasePCA or _PLS."
    )
