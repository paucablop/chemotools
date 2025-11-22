"""Shared latent-variable plotting utilities for inspector implementations."""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple, TYPE_CHECKING, Union

import numpy as np

from ..helpers import _latent_space as _latent_plots
from chemotools.inspector._utils import (
    ComponentSpec,
    normalize_components,
    normalize_datasets,
)
from chemotools.plotting._styles import DATASET_COLORS
from chemotools.outliers import HotellingT2, QResiduals

if TYPE_CHECKING:  # pragma: no cover
    from typing import Protocol

    from matplotlib.figure import Figure
    from chemotools.inspector._base import ModelTypes

    class _LatentInspectorProto(Protocol):
        @property
        def model(self) -> ModelTypes:  # pragma: no cover
            ...

        @property
        def confidence(self) -> float:  # pragma: no cover
            ...

        def _get_raw_data(
            self, dataset: str
        ) -> Tuple[np.ndarray, Optional[np.ndarray]]:  # pragma: no cover
            ...

        def _get_preprocessed_feature_names(self) -> np.ndarray:  # pragma: no cover
            ...


class LatentVariableMixin:
    """Mixin providing reusable helpers for latent-space visualisations."""

    component_label: str = "LV"

    def _latent_inspector(self) -> "_LatentInspectorProto":
        return self  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def nr_components(self) -> int:
        """Return the number of latent variables/components."""
        inspector = self._latent_inspector()
        # Access n_components_ from the inspector instance (provided by _BaseInspector)
        return getattr(inspector, "n_components_", 0)

    @property
    def hotelling_t2_limit(self) -> float:
        """Return the Hotelling's T² critical value at the specified confidence level.

        Calculated using the training data. The limit is cached after first calculation.
        """
        limit = getattr(self, "_hotelling_t2_limit", None)
        if limit is None:
            inspector = self._latent_inspector()
            hotelling = HotellingT2(inspector.model, confidence=inspector.confidence)
            X_train, _ = inspector._get_raw_data("train")
            hotelling.fit(X_train)
            limit = hotelling.critical_value_
            setattr(self, "_hotelling_t2_limit", limit)
        return limit

    @property
    def q_residuals_limit(self) -> float:
        """Return the Q residuals critical value at the specified confidence level.

        Calculated using the training data. The limit is cached after first calculation.
        """
        limit = getattr(self, "_q_residuals_limit", None)
        if limit is None:
            inspector = self._latent_inspector()
            q_detector = QResiduals(inspector.model, confidence=inspector.confidence)
            X_train, _ = inspector._get_raw_data("train")
            q_detector.fit(X_train)
            limit = q_detector.critical_value_
            setattr(self, "_q_residuals_limit", limit)
        return limit

    # ------------------------------------------------------------------
    # Abstract hooks expected from concrete inspectors
    # ------------------------------------------------------------------
    def get_latent_scores(self, dataset: str) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def get_latent_explained_variance(self) -> Optional[np.ndarray]:  # pragma: no cover
        return None

    def get_latent_loadings(self) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Public helpers used by inspectors
    # ------------------------------------------------------------------
    def create_latent_variance_figure(
        self,
        variance_threshold: float,
        figsize: Tuple[float, float],
    ) -> Optional["Figure"]:
        """Create explained-variance plot for latent components, if available."""

        variance = self.get_latent_explained_variance()
        if variance is None:
            return None
        return _latent_plots.create_variance_plot(
            explained_variance_ratio=variance,
            variance_threshold=variance_threshold,
            figsize=figsize,
        )

    def create_latent_loadings_figure(
        self,
        loadings_components: Union[int, Sequence[int]],
        xlabel: str,
        figsize: Tuple[float, float],
    ) -> "Figure":
        """Create loadings-style plot for the latent variables."""

        loadings = self.get_latent_loadings()
        feature_names = self._get_latent_feature_names()
        return _latent_plots.create_loadings_plot(
            loadings=loadings,
            feature_names=feature_names,
            loadings_components=loadings_components,
            xlabel=xlabel,
            figsize=figsize,
            component_label=self._get_latent_component_label(),
        )

    def create_latent_scores_figures(
        self,
        dataset: Union[str, Sequence[str]],
        components: Union[ComponentSpec, Sequence[ComponentSpec]],
        *,
        color_by_y: bool,
        annotate_by: Optional[Union[str, Dict[str, np.ndarray]]],
        figsize: Tuple[float, float],
    ) -> Dict[str, "Figure"]:
        """Generate per-component latent scores plots for requested datasets."""

        dataset_names = list(normalize_datasets(dataset))
        if not dataset_names:
            raise ValueError("At least one dataset is required for scores plotting")

        components_list = normalize_components(components)
        figures: Dict[str, "Figure"] = {}
        multi_dataset = len(dataset_names) > 1
        requested_color_by_y = color_by_y
        color_by_for_multi = False if multi_dataset else color_by_y
        explained_var = self._get_explained_variance_for_scores(dataset_names[0])
        component_label = self._get_latent_component_label()

        if multi_dataset:
            datasets_data = self._prepare_scores_datasets(dataset_names)

            # Get training scores for confidence ellipse reference (even if train not requested)
            train_scores_for_ellipse = self.get_latent_scores("train")

            # Get confidence level from inspector
            inspector = self._latent_inspector()
            confidence_level = inspector.confidence

            for idx, component_spec in enumerate(components_list, start=1):
                fig = _latent_plots.create_scores_plot_multi_dataset(
                    component_spec=component_spec,
                    datasets_data=datasets_data,
                    explained_var=explained_var,
                    color_by_y=color_by_for_multi,
                    annotate_by=annotate_by,
                    figsize=figsize,
                    component_label=component_label,
                    train_scores_for_ellipse=train_scores_for_ellipse,
                    confidence=confidence_level,
                )
                figures[f"scores_{idx}"] = fig
        else:
            dataset_name = dataset_names[0]
            scores = self.get_latent_scores(dataset_name)
            inspector = self._latent_inspector()
            _, y = inspector._get_raw_data(dataset_name)

            # Get confidence level from inspector
            confidence_level = inspector.confidence

            # Get training scores for ellipse reference (if not already train dataset)
            train_scores_for_ellipse = None
            if dataset_name.lower() != "train":
                try:
                    train_scores_for_ellipse = self.get_latent_scores("train")
                except (ValueError, KeyError):
                    # Train dataset not available, skip ellipse
                    pass

            for idx, component_spec in enumerate(components_list, start=1):
                fig = _latent_plots.create_scores_plot_single_dataset(
                    component_spec=component_spec,
                    scores=scores,
                    y=y,
                    explained_var=explained_var,
                    dataset_name=dataset_name,
                    color_by_y=requested_color_by_y,
                    annotate_by=annotate_by,
                    figsize=figsize,
                    component_label=component_label,
                    dataset_color=DATASET_COLORS.get(dataset_name, "gray"),
                    confidence=confidence_level,
                    train_scores_for_ellipse=train_scores_for_ellipse,
                )
                figures[f"scores_{idx}"] = fig

        return figures

    def create_latent_distance_figure(
        self,
        dataset: Union[str, Sequence[str]],
        *,
        color_by_y: bool,
        figsize: Tuple[float, float],
        annotate_by: Optional[Union[str, Dict[str, np.ndarray]]] = None,
    ) -> "Figure":
        """Create Hotelling T² vs Q residuals plot for the provided datasets."""

        dataset_names = list(normalize_datasets(dataset))
        datasets_data: Dict[str, Dict[str, Optional[np.ndarray]]] = {}
        inspector = self._latent_inspector()
        for ds in dataset_names:
            X, y = inspector._get_raw_data(ds)
            datasets_data[ds] = {"X": X, "y": y}

        # Fit detectors once on the training data to ensure consistent limits
        train_X, _ = inspector._get_raw_data("train")
        hotelling = HotellingT2(inspector.model, confidence=inspector.confidence)
        hotelling.fit(train_X)

        q_detector = QResiduals(inspector.model, confidence=inspector.confidence)
        q_detector.fit(train_X)

        return _latent_plots.create_model_distances_plot(
            datasets_data=datasets_data,
            model=inspector.model,
            confidence=inspector.confidence,
            color_by_y=color_by_y,
            figsize=figsize,
            hotelling_detector=hotelling,
            q_residuals_detector=q_detector,
            annotate_by=annotate_by,
        )

    # ------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------
    def _get_latent_component_label(self) -> str:
        return getattr(self, "component_label", "LV")

    def _get_latent_feature_names(self) -> np.ndarray:
        inspector = self._latent_inspector()
        return inspector._get_preprocessed_feature_names()

    def _get_explained_variance_for_scores(self, reference_dataset: str) -> np.ndarray:
        variance = self.get_latent_explained_variance()
        if variance is not None:
            return variance
        scores = self.get_latent_scores(reference_dataset)
        return np.zeros(scores.shape[1], dtype=float)

    def _prepare_scores_datasets(
        self, dataset_names: Sequence[str]
    ) -> Dict[str, Dict[str, Optional[np.ndarray]]]:
        datasets_data: Dict[str, Dict[str, Optional[np.ndarray]]] = {}
        inspector = self._latent_inspector()
        for ds in dataset_names:
            scores = self.get_latent_scores(ds)
            _, y = inspector._get_raw_data(ds)
            datasets_data[ds] = {"scores": scores, "y": y}
        return datasets_data
