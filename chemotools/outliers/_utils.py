from typing import Optional, TypeVar, Tuple, Union

from sklearn.cross_decomposition._pls import _PLS
from sklearn.decomposition._base import _BasePCA
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

ModelTypes = TypeVar("ModelType", _BasePCA, _PLS)


def get_model_parameters(model: ModelTypes) -> Tuple[int, int, int]:
    """
    Get the number of features, components and samples from a model with PLS or PCA. types.

    Parameters
    ----------
    model : ModelType
        A fitted model of type _BasePCA or _PLS

    Returns
    -------
    Tuple[int, int, int]
        The number of features, components and samples in the model
    """
    if isinstance(model, _BasePCA):
        return model.n_features_in_, model.n_components_, model.n_samples_
    elif isinstance(model, _PLS):
        return model.n_features_in_, model.n_components, len(model.x_scores_)
    else:
        raise ValueError(
            "Model not a valid model. Must be of base type _BasePCA or _PLS or a Pipeline ending with one of these types."
        )


def validate_confidence(confidence: float) -> float:
    """Validate parameters using sklearn conventions.

    Parameters
    ----------
    confidence : float
        Confidence level for statistical calculations (between 0 and 1)

    Returns
    -------
    float
        The validated confidence level

    Raises
    ------
    ValueError
        If confidence is not between 0 and 1
    """
    if not 0 < confidence < 1:
        raise ValueError("Confidence must be between 0 and 1")
    return confidence


def validate_and_extract_model(
    model: Union[ModelTypes, Pipeline],
) -> Tuple[ModelTypes, Optional[Pipeline], int, int, int]:
    """Validate and extract the model and preprocessing steps.

    Parameters
    ----------
    model : Union[ModelTypes, Pipeline]
        A fitted PCA/PLS model or Pipeline ending with such a model

    Returns
    -------
    Tuple[ModelTypes, Optional[Pipeline]]
        The extracted model and preprocessing steps

    Raises
    ------
    ValueError
        If the model is not of type _BasePCA or _PLS or a Pipeline ending with one of these types or if the model is not fitted
    """
    if isinstance(model, Pipeline):
        preprocessing = model[:-1]
        model = model[-1]
    else:
        preprocessing = None

    if not isinstance(model, (_BasePCA, _PLS)):
        raise ValueError(
            "Model not a valid model. Must be of base type _BasePCA or _PLS or a Pipeline ending with one of these types."
        )

    check_is_fitted(model)
    n_features_in, n_components, n_samples = get_model_parameters(model)
    return model, preprocessing, n_features_in, n_components, n_samples
