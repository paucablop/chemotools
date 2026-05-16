from abc import ABC, abstractmethod
from typing import Union, cast

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.cross_decomposition._pls import _PLS
from sklearn.feature_selection._base import SelectorMixin
from sklearn.pipeline import Pipeline

from chemotools._doc_mixin import DocLinkMixin
from chemotools._types import ModelInput
from chemotools._validation import validate_and_extract_model

# Backward-compatible alias – existing consumers import this name.
ModelTypes = ModelInput


class _PLSFeatureSelectorBase(DocLinkMixin, ABC, BaseEstimator, SelectorMixin):
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
    estimator_ : _PLS
        The fitted model of type _PLS

    feature_scores_ : np.ndarray
        The calculated feature scores based on the selected method.

    support_mask : np.ndarray
        The boolean mask indicating which features are selected.
    """

    # Narrow from EstimatorType — this base enforces PLS-only via allowed_types.
    estimator_: _PLS

    def __init__(
        self,
        model: Union[_PLS, Pipeline],
    ) -> None:
        estimator, _ = validate_and_extract_model(model, allowed_types=(_PLS,))
        self.estimator_ = cast(_PLS, estimator)

    @abstractmethod
    def _calculate_features(self, X: np.ndarray) -> np.ndarray:
        """Calculate the residuals of the model.

        Returns
        -------
        ndarray of shape (n_samples,)
            The residuals of the model
        """
