from abc import ABC, abstractmethod
from typing import Union

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.cross_decomposition._pls import _PLS
from sklearn.feature_selection._base import SelectorMixin
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

ModelTypes = Union[_PLS, Pipeline]


class _PLSFeatureSelectorBase(ABC, BaseEstimator, SelectorMixin):
    """Feature selection base class for _PLS-like models.

    Parameters
    ----------
    model : Union[_PLS, Pipeline]
        A fitted  _PLS models or Pipeline ending with such a model

    threshold : float
        The threshold for feature selection. Features with importance
        above this threshold will be selected.

    Attributes
    ----------
    estimator_ : ModelTypes
        The fitted model of type _BasePCA or _PLS

    feature_scores_ : np.ndarray
        The calculated feature scores based on the selected method.

    support_mask : np.ndarray
        The boolean mask indicating which features are selected.
    """

    def __init__(
        self,
        model: Union[_PLS, Pipeline],
    ) -> None:
        self.estimator_ = _validate_and_extract_model(model)

    @abstractmethod
    def _calculate_features(self, X: np.ndarray) -> np.ndarray:
        """Calculate the residuals of the model.

        Returns
        -------
        ndarray of shape (n_samples,)
            The residuals of the model
        """


def _validate_and_extract_model(
    model: Union[_PLS, Pipeline],
) -> _PLS:
    """Validate and extract the model.

    Parameters
    ----------
    model : Union[_PLS, Pipeline]
        A fitted _PLS model or Pipeline ending with such a model

    Returns
    -------
    _PLS
        The extracted estimator

    Raises
    ------
    TypeError
        If the model is not of type _BasePCA or _PLS or a Pipeline ending with one of these types or if the model is not fitted
    """
    if isinstance(model, Pipeline):
        estimator = model[-1]
    else:
        estimator = model

    if not isinstance(estimator, _PLS):
        raise TypeError(
            "Model not a valid model. Must be of base type _BasePCA or _PLS or a Pipeline ending with one of these types."
        )

    check_is_fitted(model)
    return estimator
