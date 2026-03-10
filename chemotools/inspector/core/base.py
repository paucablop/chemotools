from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
from sklearn.cross_decomposition._pls import _PLS
from sklearn.decomposition._base import _BasePCA
from sklearn.pipeline import Pipeline

if TYPE_CHECKING:
    from matplotlib.figure import Figure
from sklearn.utils import check_array

from chemotools._types import ModelInput

from .summaries import InspectorSummary
from .utils import normalize_datasets
from .validation import _validate_and_extract_model, _validate_datasets_consistency


@dataclass(frozen=True)
class InspectorDataset:
    """Immutable container for a single dataset split used by inspectors.

    This is a frozen dataclass, meaning instances cannot be modified after creation.
    Use direct attribute access (e.g., ``dataset.X``, ``dataset.y``) to retrieve data.

    Attributes
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    y : np.ndarray, optional
        Target values of shape (n_samples,) or (n_samples, n_targets).
    labels : np.ndarray, optional
        Sample labels for annotation purposes.
    """

    X: np.ndarray
    y: Optional[np.ndarray] = None
    labels: Optional[np.ndarray] = None

    @property
    def n_samples(self) -> int:
        """Return the number of samples in the dataset."""
        return self.X.shape[0]


@dataclass
class InspectorPlotConfig:
    """Configuration for inspector plots."""

    scores_figsize: Tuple[float, float] = (6, 6)
    loadings_figsize: Tuple[float, float] = (10, 5)
    variance_figsize: Tuple[float, float] = (10, 5)
    spectra_figsize: Tuple[float, float] = (12, 5)
    distances_figsize: Tuple[float, float] = (8, 6)
    regression_figsize: Tuple[float, float] = (8, 6)


class _DataHoldingBase:
    """Lightweight base providing dataset storage, feature-axis management,
    preprocessed-data caching, and figure lifecycle.

    This class holds the shared "data management" responsibilities that are
    common to **all** inspectors (including ``PreprocessingInspector``).
    Estimator/model-specific logic lives in :class:`_BaseInspector`, which
    extends this class.
    """

    # -------------------------------------------------------------------------
    # Initialisation
    # -------------------------------------------------------------------------
    def __init__(
        self,
        *,
        datasets: Dict[str, InspectorDataset],
        n_features_in: int,
        feature_names: Optional[np.ndarray] = None,
    ) -> None:
        self.datasets_: Dict[str, InspectorDataset] = datasets
        self.n_features_in_: int = n_features_in

        # Process feature names
        self.feature_names: Optional[np.ndarray] = None
        if feature_names is not None:
            feature_array = np.asarray(feature_names)
            if feature_array.shape[0] != self.n_features_in_:
                raise ValueError(
                    "x_axis length must match number of features. "
                    f"Got {feature_array.shape[0]} vs {self.n_features_in_}."
                )
            self.feature_names = feature_array

        # Set up x_axis for plotting
        if self.feature_names is not None:
            self._x_axis = np.array(self.feature_names, copy=True)
        else:
            self._x_axis = np.arange(self.n_features_in_)

        # Caches
        self._preprocessed_cache: Dict[str, np.ndarray] = {}

        # Figure tracking for automatic cleanup
        self._tracked_figures: List["Figure"] = []

    # -------------------------------------------------------------------------
    # Input normalization helpers
    # -------------------------------------------------------------------------
    @staticmethod
    def _normalize_target_array(target: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Normalize target array to 1D if needed."""
        if target is None:
            return None
        arr = check_array(
            target,
            dtype=None,
            ensure_2d=False,
            ensure_all_finite=True,
            input_name="target",
        )
        if arr.ndim == 2 and arr.shape[1] == 1:
            return arr.ravel()
        return arr

    @staticmethod
    def _prepare_labels(
        dataset_name: str,
        expected_len: int,
        sample_labels: Optional[Dict[str, Sequence]],
    ) -> Optional[np.ndarray]:
        """Prepare and validate sample labels for a dataset."""
        if not sample_labels or dataset_name not in sample_labels:
            return None
        labels = np.asarray(sample_labels[dataset_name])
        if labels.shape[0] != expected_len:
            raise ValueError(
                f"Sample labels for '{dataset_name}' must have length {expected_len}. "
                f"Got {labels.shape[0]}."
            )
        return labels

    # -------------------------------------------------------------------------
    # Dataset access methods
    # -------------------------------------------------------------------------
    def _get_dataset(self, name: str) -> InspectorDataset:
        """Get a dataset by name with helpful error messages."""
        try:
            return self.datasets_[name]
        except KeyError as exc:
            available = ", ".join(self.datasets_.keys())
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

    def _get_raw_data(self, name: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Get raw X and y for a dataset."""
        dataset = self._get_dataset(name)
        return dataset.X, dataset.y

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------
    @property
    def n_features(self) -> int:
        """Return the number of features in original data."""
        return self.n_features_in_

    @property
    def n_samples(self) -> Dict[str, int]:
        """Return the number of samples in each dataset."""
        return {name: dataset.n_samples for name, dataset in self.datasets_.items()}

    @property
    def x_axis(self) -> np.ndarray:
        """Return the feature names/indices."""
        return self._x_axis

    # -------------------------------------------------------------------------
    # Figure Management
    # -------------------------------------------------------------------------
    def close_figures(self) -> None:
        """Close all figures created by this inspector.

        This method closes all matplotlib figures that were created by previous
        calls to `inspect()` or `inspect_spectra()`. Use this to free memory
        when you're done with the plots.

        Examples
        --------
        >>> inspector = PCAInspector(model, X_train)
        >>> figures = inspector.inspect()
        >>> # ... work with figures ...
        >>> inspector.close_figures()  # Free memory
        """
        import matplotlib.pyplot as plt

        for fig in self._tracked_figures:
            plt.close(fig)
        self._tracked_figures.clear()

    def _track_figures(self, figures: Dict[str, "Figure"]) -> Dict[str, "Figure"]:
        """Track figures for later cleanup and return them.

        Parameters
        ----------
        figures : dict
            Dictionary of figure name to Figure object

        Returns
        -------
        dict
            The same dictionary (for chaining)
        """
        self._tracked_figures.extend(figures.values())
        return figures


class _BaseInspector(_DataHoldingBase, ABC):
    """Base class for estimator-backed inspectors (PCA, PLS).

    Extends :class:`_DataHoldingBase` with estimator/model extraction,
    component resolution, confidence level, and transformer management.
    """

    def __init__(
        self,
        *,
        model: ModelInput,
        X_train: np.ndarray,
        y_train: Optional[np.ndarray] = None,
        X_test: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        supervised: bool = False,
        feature_names: Optional[np.ndarray] = None,
        sample_labels: Optional[Dict[str, Sequence]] = None,
        confidence: float = 0.95,
    ) -> None:
        if not 0 < confidence < 1:
            raise ValueError(f"confidence must be between 0 and 1, got {confidence}")
        self._confidence = confidence

        # Validate and extract model components
        estimator, transformer = _validate_and_extract_model(model)

        # Store model components
        self._model: ModelInput = model
        self.estimator_: Union[_BasePCA, _PLS] = estimator
        self.transformer_: Optional[Pipeline] = transformer

        # Build and validate datasets
        X_train_arr, datasets = self._build_datasets(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            X_val=X_val,
            y_val=y_val,
            supervised=supervised,
            sample_labels=sample_labels,
        )

        # Initialise data-holding base
        super().__init__(
            datasets=datasets,
            n_features_in=X_train_arr.shape[1],
            feature_names=feature_names,
        )

        # Store dimensions (estimator-specific)
        self.n_components_: int = self._resolve_n_components()

    @staticmethod
    def _build_datasets(
        *,
        X_train: np.ndarray,
        y_train: Optional[np.ndarray] = None,
        X_test: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        supervised: bool = False,
        sample_labels: Optional[Dict[str, Sequence]] = None,
    ) -> Tuple[np.ndarray, Dict[str, InspectorDataset]]:
        """Validate inputs and build the datasets dictionary.

        Returns
        -------
        X_train : np.ndarray
            The validated training feature matrix.
        datasets : dict of str to InspectorDataset
            Validated datasets keyed by ``'train'``, ``'test'``, ``'val'``.
        """
        normalize = _DataHoldingBase._normalize_target_array
        prepare = _DataHoldingBase._prepare_labels

        X_train = check_array(
            X_train,
            dtype="numeric",
            ensure_2d=True,
            ensure_all_finite=True,
            input_name="X_train",
        )
        y_train_arr = normalize(y_train)
        X_test_arr = (
            check_array(
                X_test,
                dtype="numeric",
                ensure_2d=True,
                ensure_all_finite=True,
                input_name="X_test",
            )
            if X_test is not None
            else None
        )
        y_test_arr = normalize(y_test)
        X_val_arr = (
            check_array(
                X_val,
                dtype="numeric",
                ensure_2d=True,
                ensure_all_finite=True,
                input_name="X_val",
            )
            if X_val is not None
            else None
        )
        y_val_arr = normalize(y_val)

        _validate_datasets_consistency(
            X_train,
            y_train_arr,
            X_test_arr,
            y_test_arr,
            X_val_arr,
            y_val_arr,
            supervised=supervised,
        )

        datasets: Dict[str, InspectorDataset] = {
            "train": InspectorDataset(
                X=X_train,
                y=y_train_arr,
                labels=prepare("train", X_train.shape[0], sample_labels),
            )
        }

        if X_test_arr is not None:
            datasets["test"] = InspectorDataset(
                X=X_test_arr,
                y=y_test_arr,
                labels=prepare("test", X_test_arr.shape[0], sample_labels),
            )

        if X_val_arr is not None:
            datasets["val"] = InspectorDataset(
                X=X_val_arr,
                y=y_val_arr,
                labels=prepare("val", X_val_arr.shape[0], sample_labels),
            )

        return X_train, datasets

    def _resolve_n_components(self) -> int:
        """Resolve the number of components from the estimator."""
        # Try n_components_ (standard for fitted sklearn models)
        n_comp = getattr(
            self.estimator_,
            "n_components_",
            getattr(self.estimator_, "n_components", None),
        )

        if n_comp is not None:
            return int(n_comp)

        raise AttributeError("Cannot determine number of components for estimator")

    # -------------------------------------------------------------------------
    # Preprocessed data access (estimator-backed inspectors)
    # -------------------------------------------------------------------------
    def _get_preprocessed_data(self, name: str) -> np.ndarray:
        """Get preprocessed X for a dataset (cached)."""
        if name in self._preprocessed_cache:
            return self._preprocessed_cache[name]

        X = self._get_dataset(name).X
        if self.transformer_ is None:
            self._preprocessed_cache[name] = X
        else:
            self._preprocessed_cache[name] = self.transformer_.transform(X)
        return self._preprocessed_cache[name]

    def _get_feature_mask(self) -> Optional[np.ndarray]:
        """Get feature selection mask if a selector is present in the pipeline."""
        from sklearn.feature_selection._base import SelectorMixin

        transformer = self.transformer_
        if transformer is None:
            return None

        if isinstance(transformer, Pipeline):
            for _, step in transformer.steps:
                if isinstance(step, SelectorMixin):
                    return step.get_support()
        elif isinstance(transformer, SelectorMixin):
            return transformer.get_support()

        return None

    def _get_preprocessed_feature_names(
        self, base_dataset: str = "train"
    ) -> np.ndarray:
        """Get feature names after preprocessing (accounting for feature selection)."""
        mask = self._get_feature_mask()
        if mask is not None and self.feature_names is not None:
            return self.feature_names[mask]
        if self.feature_names is not None:
            return self.feature_names
        X = self._get_preprocessed_data(base_dataset)
        return np.arange(X.shape[1])

    def _get_preprocessed_x_axis(self) -> np.ndarray:
        """Get x_axis after feature selection.

        Returns
        -------
        x_axis : np.ndarray
            X-axis/feature indices after feature selection. If no feature
            selector is present, returns the original x_axis.
        """
        return self._get_preprocessed_feature_names()

    # -------------------------------------------------------------------------
    # Configuration helpers
    # -------------------------------------------------------------------------
    def _prepare_inspection_config(
        self,
        dataset: Union[str, Sequence[str]],
        color_by: Optional[
            Union[str, Dict[str, Any], Sequence[Any], np.ndarray]
        ] = None,
        annotate_by: Optional[
            Union[str, Dict[str, Any], Sequence[Any], np.ndarray]
        ] = None,
    ) -> Tuple[
        List[str],
        Optional[Union[str, Dict[str, Any]]],
        Optional[Union[str, Dict[str, Any]]],
    ]:
        """
        Prepare the configuration for inspection by normalizing datasets and arguments.

        This method handles the logic for:
        1. Normalizing the dataset argument to a list of strings.
        2. Wrapping raw array inputs for color_by/annotate_by into dictionaries
           when inspecting a single dataset.
        3. Setting default values for color_by if not provided.

        Parameters
        ----------
        dataset : str or sequence of str
            The dataset(s) to inspect.
        color_by : str, dict, or sequence, optional
            The coloring configuration.
        annotate_by : str, dict, or sequence, optional
            The annotation configuration.

        Returns
        -------
        tuple
            A tuple containing:
            - datasets (List[str]): The list of dataset names to inspect.
            - color_by (Optional[Union[str, Dict[str, Any]]]):
              The normalized color_by configuration.
            - annotate_by
              (Optional[Union[str, Dict[str, Any]]]):
              The normalized annotate_by configuration.
        """
        datasets = normalize_datasets(dataset)
        use_suffix = len(datasets) > 1

        # Handle color_by
        if color_by is not None:
            if not isinstance(color_by, (str, dict)):
                if use_suffix:
                    raise ValueError(
                        "When inspecting multiple datasets, "
                        "color_by must be a string "
                        "or a dictionary."
                    )
                color_by = {datasets[0]: color_by}

        # Handle annotate_by
        if annotate_by is not None:
            if not isinstance(annotate_by, (str, dict)):
                if use_suffix:
                    raise ValueError(
                        "When inspecting multiple datasets, "
                        "annotate_by must be a string "
                        "or a dictionary."
                    )
                annotate_by = {datasets[0]: annotate_by}

        # Default color_by logic
        if color_by is None and not use_suffix:
            color_by = "y"

        return datasets, color_by, annotate_by  # type: ignore

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------
    @property
    def model(self) -> ModelInput:
        """Return the original model."""
        return self._model

    @property
    def estimator(self) -> Union[_BasePCA, _PLS]:
        """Return the underlying estimator (PCA or PLS)."""
        return self.estimator_

    @property
    def transformer(self) -> Optional[Pipeline]:
        """Return the preprocessing transformer (if any)."""
        return self.transformer_

    @property
    def confidence(self) -> float:
        """Return the confidence level for outlier detection."""
        return self._confidence

    # -------------------------------------------------------------------------
    # Summary helpers
    # -------------------------------------------------------------------------
    def _base_summary(self) -> InspectorSummary:
        """Create base summary with common model information."""
        return InspectorSummary(
            model_type=self.model.__class__.__name__,
            has_preprocessing=self.transformer is not None,
            n_features=self.n_features_in_,
            n_samples={name: ds.n_samples for name, ds in self.datasets_.items()},
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
