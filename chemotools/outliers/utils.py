from typing import TypeVar, Tuple

from sklearn.decomposition._base import _BasePCA
from sklearn.cross_decomposition._pls import _PLS

ModelType = TypeVar("ModelType", _BasePCA, _PLS)


def get_model_parameters(model: ModelType) -> Tuple[int, int, int]:
    """
    Get the number of features, components and samples from a model with PLS or PCA. types.
    """
    if isinstance(model, _BasePCA):
        return model.n_features_in_, model.n_components_, model.n_samples_
    elif isinstance(model, _PLS):
        return model.n_features_in_, model.n_components, len(model.x_scores_)
    else:
        raise ValueError(
            "Model not a valid model. Must be of base type _BasePCA or _PLS or a Pipeline ending with one of these types."
        )