from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
from abc import ABC
from sklearn.cross_decomposition._pls import _PLS
from sklearn.decomposition._base import _BasePCA
from sklearn.pipeline import Pipeline
from sklearn.utils import check_array

from .validation import _validate_and_extract_model, _validate_datasets_consistency
from .summaries import BaseSummary

ModelTypes = Union[_BasePCA, _PLS, Pipeline]


@dataclass
class InspectorDataset:
    """Immutable container for a single dataset split used by inspectors."""

    X: np.ndarray
    y: Optional[np.ndarray] = None
    labels: Optional[np.ndarray] = None

    def __getitem__(self, key: str):
        if key == "X":
            return self.X
        if key == "y":
            return self.y
        if key == "labels":
            return self.labels
        raise KeyError(key)

    def __contains__(self, key: object) -> bool:
        return key in {"X", "y", "labels"}

    def keys(self) -> Tuple[str, str, str]:
        return ("X", "y", "labels")

    def items(self):
        for key in self.keys():
            yield key, self[key]

    def __iter__(self):
        return iter(self.keys())

    @property
    def n_samples(self) -> int:
        return self.X.shape[0]


class InspectorState:
    """Shared lifecycle state for inspector implementations."""

    def __init__(
        self,
        model: ModelTypes,
        X_train: np.ndarray,
        y_train: Optional[np.ndarray],
        X_test: Optional[np.ndarray],
        y_test: Optional[np.ndarray],
        X_val: Optional[np.ndarray],
        y_val: Optional[np.ndarray],
        *,
        supervised: bool,
        feature_names: Optional[Sequence] = None,
        sample_labels: Optional[Dict[str, Sequence]] = None,
    ) -> None:
        estimator, transformer = _validate_and_extract_model(model)

        X_train = check_array(
            X_train, dtype="numeric", ensure_2d=True, input_name="X_train"
        )
        y_train_arr = self._normalize_target_array(y_train)
        X_test_arr = (
            check_array(X_test, dtype="numeric", ensure_2d=True, input_name="X_test")
            if X_test is not None
            else None
        )
        y_test_arr = self._normalize_target_array(y_test)
        X_val_arr = (
            check_array(X_val, dtype="numeric", ensure_2d=True, input_name="X_val")
            if X_val is not None
            else None
        )
        y_val_arr = self._normalize_target_array(y_val)

        _validate_datasets_consistency(
            X_train,
            y_train_arr,
            X_test_arr,
            y_test_arr,
            X_val_arr,
            y_val_arr,
            supervised=supervised,
        )

        self.model: ModelTypes = model
        self.estimator: Union[_BasePCA, _PLS] = estimator
        self.transformer: Optional[Pipeline] = transformer

        self.datasets: Dict[str, InspectorDataset] = {
            "train": InspectorDataset(
                X=X_train,
                y=y_train_arr,
                labels=self._prepare_labels("train", X_train.shape[0], sample_labels),
            )
        }

        if X_test_arr is not None:
            self.datasets["test"] = InspectorDataset(
                X=X_test_arr,
                y=y_test_arr,
                labels=self._prepare_labels("test", X_test_arr.shape[0], sample_labels),
            )

        if X_val_arr is not None:
            self.datasets["val"] = InspectorDataset(
                X=X_val_arr,
                y=y_val_arr,
                labels=self._prepare_labels("val", X_val_arr.shape[0], sample_labels),
            )

        self.n_features_in_: int = X_train.shape[1]
        self.n_components_: int = self._resolve_n_components()

        self.feature_names_: Optional[np.ndarray] = None
        if feature_names is not None:
            feature_array = np.asarray(feature_names)
            if feature_array.shape[0] != self.n_features_in_:
                raise ValueError(
                    "x_axis length must match number of features. "
                    f"Got {feature_array.shape[0]} vs {self.n_features_in_}."
                )
            self.feature_names_ = feature_array

        self.sample_labels: Dict[str, np.ndarray] = {
            name: dataset.labels
            for name, dataset in self.datasets.items()
            if dataset.labels is not None
        }

        self._preprocessed_cache: Dict[str, np.ndarray] = {}

    @staticmethod
    def _normalize_target_array(target: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if target is None:
            return None
        arr = check_array(target, dtype=None, ensure_2d=False, input_name="target")
        if arr.ndim == 2 and arr.shape[1] == 1:
            return arr.ravel()
        return arr

    @staticmethod
    def _prepare_labels(
        dataset_name: str,
        expected_len: int,
        sample_labels: Optional[Dict[str, Sequence]],
    ) -> Optional[np.ndarray]:
        if not sample_labels or dataset_name not in sample_labels:
            return None
        labels = np.asarray(sample_labels[dataset_name])
        if labels.shape[0] != expected_len:
            raise ValueError(
                f"Sample labels for '{dataset_name}' must have length {expected_len}. "
                f"Got {labels.shape[0]}."
            )
        return labels

    def _resolve_n_components(self) -> int:
        if hasattr(self.estimator, "n_components_"):
            return int(self.estimator.n_components_)
        if hasattr(self.estimator, "n_components"):
            return int(self.estimator.n_components)
        raise AttributeError("Cannot determine number of components for estimator")

    def get_dataset(self, name: str) -> InspectorDataset:
        try:
            return self.datasets[name]
        except KeyError as exc:
            available = ", ".join(self.datasets.keys())
            if name == "test":
                raise ValueError(
                    "Test data not provided. Initialize with X_test/y_test."
                ) from exc
            if name == "val":
                raise ValueError(
                    "Validation data not provided. Initialize with X_val/y_val."
                ) from exc
            raise ValueError(
                f"Invalid dataset '{name}'. Available options: {available}."
            ) from exc

    def iter_datasets(
        self, names: Iterable[str]
    ) -> Iterable[Tuple[str, InspectorDataset]]:
        for name in names:
            yield name, self.get_dataset(name)

    def get_raw_xy(self, name: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        dataset = self.get_dataset(name)
        return dataset.X, dataset.y

    def get_preprocessed_X(self, name: str) -> np.ndarray:
        if name in self._preprocessed_cache:
            return self._preprocessed_cache[name]

        X = self.get_dataset(name).X
        if self.transformer is None:
            self._preprocessed_cache[name] = X
        else:
            self._preprocessed_cache[name] = self.transformer.transform(X)
        return self._preprocessed_cache[name]

    def get_feature_mask(self) -> Optional[np.ndarray]:
        from sklearn.feature_selection._base import SelectorMixin

        transformer = self.transformer
        if transformer is None:
            return None

        if isinstance(transformer, Pipeline):
            for _, step in transformer.steps:
                if isinstance(step, SelectorMixin):
                    return step.get_support()
        elif isinstance(transformer, SelectorMixin):
            return transformer.get_support()

        return None

    def get_preprocessed_feature_names(self, base_dataset: str = "train") -> np.ndarray:
        mask = self.get_feature_mask()
        if mask is not None and self.feature_names_ is not None:
            return self.feature_names_[mask]
        if self.feature_names_ is not None:
            return self.feature_names_
        X = self.get_preprocessed_X(base_dataset)
        return np.arange(X.shape[1])


class _BaseInspector(ABC):
    """Base class encapsulating shared inspector responsibilities."""

    def __init__(
        self,
        *,
        model: ModelTypes,
        X_train: np.ndarray,
        y_train: Optional[np.ndarray] = None,
        X_test: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        supervised: bool = False,
        feature_names: Optional[Sequence] = None,
        sample_labels: Optional[Dict[str, Sequence]] = None,
        confidence: float = 0.95,
    ) -> None:
        if not 0 < confidence < 1:
            raise ValueError(f"confidence must be between 0 and 1, got {confidence}")
        self._confidence = confidence

        self._state = InspectorState(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            X_val=X_val,
            y_val=y_val,
            supervised=supervised,
            feature_names=feature_names,
            sample_labels=sample_labels,
        )

        self._model = self._state.model
        self.estimator_ = self._state.estimator
        self.transformer_ = self._state.transformer
        self.datasets_: Dict[str, InspectorDataset] = self._state.datasets
        self.n_components_: int = self._state.n_components_
        self.n_features_in_: int = self._state.n_features_in_
        self.feature_names = self._state.feature_names_
        self.sample_labels = self._state.sample_labels

        if self.feature_names is not None:
            self._x_axis = np.array(self.feature_names, copy=True)
        else:
            self._x_axis = np.arange(self.n_features_in_)

        # Backwards-compatible attributes used in existing tests/extensions
        train_dataset = self.datasets_["train"]
        self._X_train = train_dataset.X
        self._y_train = train_dataset.y

        test_dataset = self.datasets_.get("test")
        self._X_test = test_dataset.X if test_dataset is not None else None
        self._y_test = test_dataset.y if test_dataset is not None else None

        val_dataset = self.datasets_.get("val")
        self._X_val = val_dataset.X if val_dataset is not None else None
        self._y_val = val_dataset.y if val_dataset is not None else None

    # ---------------------------------------------------------------------
    # Convenience helpers shared by concrete inspectors
    # ---------------------------------------------------------------------
    @property
    def model(self) -> ModelTypes:
        return self._model

    @property
    def estimator(self) -> Union[_BasePCA, _PLS]:
        """Return the underlying estimator (PCA or PLS)."""
        return self.estimator_

    @property
    def transformer(self) -> Optional[Pipeline]:
        return self.transformer_

    @property
    def nr_features(self) -> int:
        """Return the number of features in original data."""
        return self.n_features_in_

    @property
    def nr_samples(self) -> Dict[str, int]:
        """Return the number of samples in each dataset."""
        return {name: dataset.X.shape[0] for name, dataset in self.datasets_.items()}

    @property
    def x_axis(self) -> np.ndarray:
        """Return the feature names/indices."""
        return self._x_axis

    @property
    def confidence(self) -> float:
        """Return the confidence level for outlier detection."""
        return self._confidence

    def _base_summary(self) -> BaseSummary:
        """Generate common summary fields shared by all inspectors.

        Returns
        -------
        summary : BaseSummary
            Object containing common model information:
            - 'model_type': Name of the estimator class
            - 'has_preprocessing': Whether preprocessing pipeline exists
            - 'nr_features': Number of features in original data
            - 'nr_samples': Dictionary with sample counts per dataset
            - 'preprocessing_steps': List of preprocessing step info (if available)
        """
        return BaseSummary(
            model_type=type(self.estimator).__name__,
            has_preprocessing=self.transformer is not None,
            nr_features=self.nr_features,
            nr_samples=self.nr_samples.copy(),
            preprocessing_steps=self._get_preprocessing_steps(),
        )

    def _get_preprocessing_steps(self) -> List[Dict[str, Union[int, str]]]:
        """Get list of preprocessing steps with their details.

        Returns
        -------
        steps : list of dict
            List of dictionaries with 'step', 'name', and 'type' keys.
            Empty list if no preprocessing pipeline exists.
        """
        if self.transformer is None:
            return []
        return [
            {"step": i, "name": name, "type": type(transform).__name__}
            for i, (name, transform) in enumerate(self.transformer.steps, 1)
        ]

    def _get_dataset(self, name: str) -> InspectorDataset:
        return self._state.get_dataset(name)

    def _get_raw_data(self, name: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        return self._state.get_raw_xy(name)

    def _get_preprocessed_data(self, name: str) -> np.ndarray:
        return self._state.get_preprocessed_X(name)

    def _get_preprocessed_feature_names(self) -> np.ndarray:
        return self._state.get_preprocessed_feature_names()

    def _get_preprocessed_x_axis(self) -> np.ndarray:
        """Get x_axis after feature selection.

        Returns
        -------
        x_axis : np.ndarray
            X-axis/feature indices after feature selection. If no feature
            selector is present, returns the original x_axis.
        """
        return self._get_preprocessed_feature_names()

    def _get_feature_mask(self) -> Optional[np.ndarray]:
        return self._state.get_feature_mask()

    def _iter_datasets(
        self, names: Iterable[str]
    ) -> Iterable[Tuple[str, InspectorDataset]]:
        return self._state.iter_datasets(names)

    def _transform_data(self, X: np.ndarray) -> np.ndarray:
        X_array = np.asarray(X)
        if self.transformer_ is None:
            return X_array
        return self.transformer_.transform(X_array)

    def _get_scores(
        self,
        X: Union[str, np.ndarray],
        dataset: Optional[str] = None,
    ) -> np.ndarray:
        dataset_name: Optional[str] = dataset

        if isinstance(X, str):
            dataset_name = X if dataset_name is None else dataset_name
            X_preprocessed = self._get_preprocessed_data(dataset_name)
            return self.estimator_.transform(X_preprocessed)

        if dataset_name and isinstance(dataset_name, str):
            # Dataset name provided but explicit data supplied - use provided data
            X_transformed = self._transform_data(np.asarray(X))
            return self.estimator_.transform(X_transformed)

        X_transformed = self._transform_data(np.asarray(X))
        return self.estimator_.transform(X_transformed)


@dataclass
class InspectorPlotConfig:
    """Configuration for inspector plots."""

    scores_figsize: Tuple[float, float] = (6, 6)
    loadings_figsize: Tuple[float, float] = (10, 5)
    variance_figsize: Tuple[float, float] = (10, 5)
    spectra_figsize: Tuple[float, float] = (12, 5)
    distances_figsize: Tuple[float, float] = (8, 6)
    regression_figsize: Tuple[float, float] = (8, 6)
