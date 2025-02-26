from typing import TypeVar, Tuple

from sklearn.decomposition._base import _BasePCA
from sklearn.cross_decomposition._pls import _PLS

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