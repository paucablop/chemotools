from abc import ABC, abstractmethod
from typing import TypeVar, Union, Optional, Tuple

import numpy as np

from sklearn.base import OutlierMixin
from sklearn.decomposition._base import _BasePCA
from sklearn.cross_decomposition._pls import _PLS
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted


ModelType = TypeVar("ModelType", _BasePCA, _PLS)


def get_model_parameters(model: ModelType) -> Tuple[int, int, int]:
    if isinstance(model, _BasePCA):
        return model.n_features_in_, model.n_components_, model.n_samples_
    elif isinstance(model, _PLS):
        return model.n_features_in_, model.n_components, len(model.x_scores_)
    else:
        raise ValueError(
            "Model must be of base type _BasePCA or _PLS or a Pipeline ending with one of these types."
        )


class _ModelResidualsBase(ABC, OutlierMixin):
    """Base class for model outlier calculations.

    Implements statistical calculations for outlier detection in dimensionality
    reduction models like PCA and PLS.

    Parameters
    ----------
    model : Union[ModelType, Pipeline]
        A fitted _BasePCA or _PLS models or Pipeline ending with such a model
    confidence : float
        Confidence level for statistical calculations (between 0 and 1)

    Attributes
    ----------
    model_ : ModelType
        The fitted model of type _BasePCA or _PLS

    preprocessing_ : Optional[Pipeline]
        Preprocessing steps before the model

    n_features_in_ : int
        Number of features in the input data

    n_components_ : int
        Number of components in the model

    n_samples_ : int
        Number of samples used to train the model

    critical_value_ : float
        The calculated critical value for outlier detection
    """

    def __init__(
        self,
        model: Union[ModelType, Pipeline],
        confidence: float,
    ) -> None:
        (
            self.model_,
            self.preprocessing_,
            self.n_features_in_,
            self.n_components_,
            self.n_samples_,
        ) = self._validate_and_extract_model(model)
        self.confidence = self._validate_confidence(confidence)
        self.critical_value_ = self._calculate_critical_value()

    def _validate_confidence(self, confidence: float) -> float:
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

    def _validate_and_extract_model(
        self, model: Union[ModelType, Pipeline]
    ) -> Tuple[ModelType, Optional[Pipeline], int, int, int]:
        """Validate and extract the model and preprocessing steps.

        Parameters
        ----------
        model : Union[ModelType, Pipeline]
            A fitted PCA/PLS model or Pipeline ending with such a model

        Returns
        -------
        Tuple[ModelType, Optional[Pipeline]]
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
                "Model must be of type _BasePCA or _PLS or a Pipeline "
                "ending with one of these types."
            )

        check_is_fitted(model)
        n_features_in, n_components, n_samples = get_model_parameters(model)
        return model, preprocessing, n_features_in, n_components, n_samples

    @abstractmethod
    def _calculate_critical_value(self) -> float:
        """Calculate the critical value for outlier detection.

        Returns
        -------
        float
            The calculated critical value for outlier detection
        """


class _ModelDiagnosticsBase(ABC):
    """Base class for model diagnostics methods. This does not implement outlier detection algorithms,
    but rather implements methods that are used to assess trained models.

    Parameters
    ----------
    model : Union[ModelType, Pipeline]
        A fitted PCA/PLS model or Pipeline ending with such a model

    Attributes
    ----------
    model_ : ModelType
        The fitted model of type _BasePCA or _PLS

    preprocessing_ : Optional[Pipeline]
        Preprocessing steps before the model

    """

    def __init__(self, model: Union[ModelType, Pipeline]):
        self.model_, self.preprocessing_ = self._validate_and_extract_model(model)

    def _validate_and_extract_model(self, model):
        """Validate and extract the model and preprocessing steps.

        Parameters
        ----------
        model : Union[ModelType, Pipeline]
            A fitted PCA/PLS model or Pipeline ending with such a model

        Returns
        -------
        Tuple[ModelType, Optional[Pipeline]]
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

        if isinstance(model, (_BasePCA, _PLS)):
            check_is_fitted(model)
        else:
            raise ValueError(
                "Model must be of base type _BasePCA or _PLS or a Pipeline ending with one of these types."
            )
        check_is_fitted(model)
        return model, preprocessing

    @abstractmethod
    def predict(self, X: np.ndarray, y: Optional[np.ndarray]) -> np.ndarray:
        """Predict the output of the model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data

        y : array-like of shape (n_samples,), default=None
            Target values

        Returns
        -------
        ndarray of shape (n_samples,)
            Predicted values
        """
