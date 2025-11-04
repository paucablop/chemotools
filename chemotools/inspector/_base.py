from abc import ABC, abstractmethod
from typing import Optional, Union, Dict
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition._base import _BasePCA
from sklearn.cross_decomposition._pls import _PLS
from sklearn.pipeline import Pipeline

from ._validate import _validate_and_extract_model

ModelTypes = Union[_BasePCA, _PLS, Pipeline]


class _BaseInspector(ABC):
    """Base class for model inspection and visualization.

    Note: This is NOT a sklearn estimator. It's a visualization and analysis
    tool that works with already-fitted models. It does not implement fit(),
    transform(), or predict() methods.

    Parameters
    ----------
    model : Union[_BasePCA, _PLS, Pipeline]
        A fitted model or pipeline ending with a PCA/PLS model

    X_train : np.ndarray
        Training data used to fit the model

    y_train : Optional[np.ndarray], default=None
        Training target values (required for supervised models)

    X_test : Optional[np.ndarray], default=None
        Test data for evaluation

    y_test : Optional[np.ndarray], default=None
        Test target values

    X_val : Optional[np.ndarray], default=None
        Validation data for evaluation

    y_val : Optional[np.ndarray], default=None
        Validation target values

    feature_names : Optional[list], default=None
        Names of features for labeling plots

    sample_labels : Optional[Dict[str, np.ndarray]], default=None
        Dictionary mapping dataset names ('train', 'test', 'val') to sample labels

    Attributes
    ----------
    estimator_ : Union[_BasePCA, _PLS]
        The extracted fitted model

    transformer_ : Optional[Pipeline]
        Preprocessing pipeline (if model is part of a Pipeline)

    n_components_ : int
        Number of components in the model

    n_features_in_ : int
        Number of input features

    datasets_ : Dict[str, Dict[str, np.ndarray]]
        Dictionary storing X and y for each dataset split
    """

    def __init__(
        self,
        model: ModelTypes,
        X_train: np.ndarray,
        y_train: Optional[np.ndarray] = None,
        X_test: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        feature_names: Optional[list] = None,
        sample_labels: Optional[Dict[str, np.ndarray]] = None,
    ):
        # Validate and extract model
        self.estimator_, self.transformer_ = _validate_and_extract_model(model)

        # Store data
        self.datasets_ = self._organize_datasets(
            X_train, y_train, X_test, y_test, X_val, y_val
        )

        # Extract model properties
        self.n_components_ = self._get_n_components()
        self.n_features_in_ = X_train.shape[1]

        # Store metadata
        self.feature_names = feature_names
        self.sample_labels = sample_labels or {}

    def _organize_datasets(self, X_train, y_train, X_test, y_test, X_val, y_val):
        """Organize datasets into a structured dictionary."""
        datasets = {"train": {"X": X_train, "y": y_train}}
        if X_test is not None:
            datasets["test"] = {"X": X_test, "y": y_test}
        if X_val is not None:
            datasets["val"] = {"X": X_val, "y": y_val}
        return datasets

    def _get_n_components(self) -> int:
        """Extract number of components from the model."""
        if hasattr(self.estimator_, "n_components_"):
            return self.estimator_.n_components_
        elif hasattr(self.estimator_, "n_components"):
            return self.estimator_.n_components
        else:
            raise AttributeError("Cannot determine number of components")

    def _transform_data(self, X: np.ndarray) -> np.ndarray:
        """Apply preprocessing pipeline if present, then return raw data."""
        if self.transformer_ is not None:
            return self.transformer_.transform(X)
        return X

    def _get_scores(self, X: np.ndarray, dataset_name: str = "train") -> np.ndarray:
        """Get scores (projections) for the data."""
        X_transformed = self._transform_data(X)
        return self.estimator_.transform(X_transformed)

    @abstractmethod
    def plot_scores(
        self,
        components: tuple = (0, 1),
        datasets: list = ["train"],
        color_by: Optional[str] = None,
        label_by: Optional[str] = None,
        **kwargs,
    ) -> plt.Figure:
        """Plot scores for specified components."""
        pass
