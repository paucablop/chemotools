"""Shared latent-variable plotting utilities for inspector implementations."""

from __future__ import annotations

from typing import Dict, Literal, Optional, Sequence, Tuple, TYPE_CHECKING, Union

import numpy as np

from chemotools.inspector.helpers import _latent as _latent_plots
from chemotools.plotting._styles import DATASET_COLORS
from chemotools.outliers import HotellingT2, QResiduals
from .utils import (
    ComponentSpec,
    normalize_components,
    normalize_datasets,
    get_xlabel_for_features,
)
from .summaries import LatentSummary

if TYPE_CHECKING:  # pragma: no cover
    from typing import Protocol

    from matplotlib.figure import Figure
    from chemotools.inspector.core.base import ModelTypes

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

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_components(self) -> int:
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
    # Private methods
    # ------------------------------------------------------------------
    def _latent_inspector(self) -> "_LatentInspectorProto":
        return self  # type: ignore[return-value]

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
        return np.full(scores.shape[1], np.nan)

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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def create_latent_variance_figure(
        self,
        variance_threshold: float = 0.95,
        figsize: Tuple[float, float] = (8, 4),
    ) -> Optional["Figure"]:
        """Create explained-variance plot for latent components.

        Parameters
        ----------
        variance_threshold : float, default=0.95
            Cumulative variance threshold to highlight on the plot.
        figsize : tuple, default=(8, 4)
            Figure dimensions (width, height) in inches.

        Returns
        -------
        Figure or None
            Matplotlib figure, or None if variance info unavailable.

        Examples
        --------
        >>> fig = inspector.create_latent_variance_figure()
        >>> fig = inspector.create_latent_variance_figure(variance_threshold=0.99)
        """

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
        loadings_components: Union[int, Sequence[int]] = 0,
        xlabel: Optional[str] = None,
        figsize: Tuple[float, float] = (8, 4),
    ) -> "Figure":
        """Create loadings plot for the latent variables.

        Parameters
        ----------
        loadings_components : int or sequence of int, default=0
            Component index(es) to plot.
        xlabel : str, optional
            Label for x-axis. Auto-detected if not provided.
        figsize : tuple, default=(8, 4)
            Figure dimensions (width, height) in inches.

        Returns
        -------
        Figure
            Matplotlib figure with loadings plot.

        Examples
        --------
        >>> fig = inspector.create_latent_loadings_figure()  # first component
        >>> fig = inspector.create_latent_loadings_figure([0, 1, 2])  # multiple
        """

        loadings = self.get_latent_loadings()
        feature_names = self._get_latent_feature_names()

        if xlabel is None:
            xlabel = get_xlabel_for_features(feature_names is not None)

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
        dataset: Union[str, Sequence[str]] = "train",
        components: Union[ComponentSpec, Sequence[ComponentSpec]] = (0, 1),
        *,
        color_by: Optional[Union[str, Dict[str, np.ndarray]]] = None,
        annotate_by: Optional[Union[str, Dict[str, np.ndarray]]] = None,
        figsize: Tuple[float, float] = (8, 6),
        color_mode: Optional[Literal["continuous", "categorical"]] = None,
    ) -> Dict[str, "Figure"]:
        """Generate scores plots for the specified component pairs.

        Parameters
        ----------
        dataset : str or list of str, default='train'
            Dataset(s) to plot.
        components : tuple or list of tuples, default=(0, 1)
            Component pair(s) to plot, e.g., (0, 1) or [(0, 1), (1, 2)].
        color_by : str or dict, optional
            How to color points ('y', 'sample_index', or custom dict).
        annotate_by : str or dict, optional
            Labels for annotating individual points.
        figsize : tuple, default=(8, 6)
            Figure dimensions (width, height) in inches.
        color_mode : {'continuous', 'categorical'}, optional
            Coloring mode for points.

        Returns
        -------
        dict of Figure
            Dictionary with keys like 'scores_1', 'scores_2', etc.

        Examples
        --------
        >>> figs = inspector.create_latent_scores_figures()
        >>> figs = inspector.create_latent_scores_figures(components=[(0, 1), (1, 2)])
        >>> figs = inspector.create_latent_scores_figures(["train", "test"])
        """

        dataset_names = list(normalize_datasets(dataset))
        if not dataset_names:
            raise ValueError("At least one dataset is required for scores plotting")

        components_list = normalize_components(components)
        figures: Dict[str, "Figure"] = {}
        multi_dataset = len(dataset_names) > 1
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
                    color_by=color_by,
                    annotate_by=annotate_by,
                    figsize=figsize,
                    component_label=component_label,
                    train_scores_for_ellipse=train_scores_for_ellipse,
                    confidence=confidence_level,
                    color_mode=color_mode,
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
                    color_by=color_by,
                    annotate_by=annotate_by,
                    figsize=figsize,
                    component_label=component_label,
                    dataset_color=DATASET_COLORS.get(dataset_name, "gray"),
                    confidence=confidence_level,
                    train_scores_for_ellipse=train_scores_for_ellipse,
                    color_mode=color_mode,
                )
                figures[f"scores_{idx}"] = fig

        return figures

    def create_latent_distance_figure(
        self,
        dataset: Union[str, Sequence[str]],
        *,
        color_by: Optional[Union[str, Dict[str, np.ndarray]]],
        figsize: Tuple[float, float],
        annotate_by: Optional[Union[str, Dict[str, np.ndarray]]] = None,
        color_mode: Optional[Literal["continuous", "categorical"]] = None,
        hotelling_detector: Optional[HotellingT2] = None,
        q_residuals_detector: Optional[QResiduals] = None,
    ) -> "Figure":
        """Create Hotelling T² vs Q residuals plot.

        Parameters
        ----------
        dataset : str or list of str
            Dataset(s) to plot.
        color_by : str or dict
            How to color points ('y', 'sample_index', or custom dict).
        figsize : tuple
            Figure dimensions (width, height) in inches.
        annotate_by : str or dict, optional
            Labels for annotating individual points.
        color_mode : {'continuous', 'categorical'}, optional
            Coloring mode for points.
        hotelling_detector : HotellingT2, optional
            Pre-fitted detector; auto-fitted on train data if not provided.
        q_residuals_detector : QResiduals, optional
            Pre-fitted detector; auto-fitted on train data if not provided.

        Returns
        -------
        Figure
            Matplotlib figure with distance plot.

        Examples
        --------
        >>> fig = inspector.create_latent_distance_figure("train", color_by="y", figsize=(8, 6))
        """

        dataset_names = list(normalize_datasets(dataset))
        datasets_data: Dict[str, Dict[str, Optional[np.ndarray]]] = {}
        inspector = self._latent_inspector()
        for ds in dataset_names:
            X, y = inspector._get_raw_data(ds)
            datasets_data[ds] = {"X": X, "y": y}

        # Fit detectors once on the training data to ensure consistent limits
        # Only if not provided
        if hotelling_detector is None or q_residuals_detector is None:
            train_X, _ = inspector._get_raw_data("train")

            if hotelling_detector is None:
                hotelling_detector = HotellingT2(
                    inspector.model, confidence=inspector.confidence
                )
                hotelling_detector.fit(train_X)

            if q_residuals_detector is None:
                q_residuals_detector = QResiduals(
                    inspector.model, confidence=inspector.confidence
                )
                q_residuals_detector.fit(train_X)

        return _latent_plots.create_model_distances_plot(
            datasets_data=datasets_data,
            model=inspector.model,
            confidence=inspector.confidence,
            color_by=color_by,
            figsize=figsize,
            hotelling_detector=hotelling_detector,
            q_residuals_detector=q_residuals_detector,
            annotate_by=annotate_by,
            color_mode=color_mode,
        )

    def latent_summary(self) -> LatentSummary:
        """Return summary of latent variable model properties.

        Returns
        -------
        summary : LatentSummary
            Object containing:
            - 'n_components': Number of latent variables
            - 'hotelling_t2_limit': Critical value for Hotelling's T²
            - 'q_residuals_limit': Critical value for Q residuals
        """
        return LatentSummary(
            n_components=int(self.n_components),
            hotelling_t2_limit=float(self.hotelling_t2_limit),
            q_residuals_limit=float(self.q_residuals_limit),
        )
