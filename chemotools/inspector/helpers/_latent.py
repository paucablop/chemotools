"""Core plot creation functions for inspectors.

This module contains reusable plotting functions for scores, loadings,
and variance plots that are common across PCA, PLS, ICA, and other
decomposition methods.
"""

from __future__ import annotations
from typing import Dict, Union, Optional, Tuple, Sequence, TYPE_CHECKING, Literal
import numpy as np
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from matplotlib.figure import Figure

from chemotools.plotting import (
    DistancesPlot,
    ExplainedVariancePlot,
    LoadingsPlot,
    ScoresPlot,
)
from chemotools.plotting._utils import annotate_points
from chemotools.plotting._styles import DATASET_COLORS, DATASET_MARKERS
from chemotools.outliers import HotellingT2, QResiduals

from ..core.utils import (
    ComponentSpec,
    prepare_annotations,
    prepare_color_values,
    normalize_components,
)


def create_variance_plot(
    explained_variance_ratio: np.ndarray,
    variance_threshold: float,
    figsize: Tuple[float, float],
) -> Figure:
    """Create explained variance plot.

    This plot works for any decomposition method (PCA, PLS, ICA) that has
    explained variance ratios.

    Parameters
    ----------
    explained_variance_ratio : np.ndarray
        Explained variance ratio for each component
    variance_threshold : float
        Threshold line to show on plot (e.g., 0.95 for 95%)
    figsize : Tuple[float, float]
        Figure size (width, height) in inches

    Returns
    -------
    Figure
        Matplotlib figure with variance plot

    Examples
    --------
    >>> var_ratios = np.array([0.45, 0.25, 0.15, 0.10, 0.05])
    >>> fig = create_variance_plot(var_ratios, 0.95, (10, 5))
    >>> fig.savefig('variance.png')
    """
    fig, ax = plt.subplots(figsize=figsize)
    variance_plot = ExplainedVariancePlot(
        explained_variance_ratio=explained_variance_ratio,
        threshold=variance_threshold,
    )
    variance_plot.render(ax=ax)

    # Apply decorations
    ax.set_title("Explained Variance by Component", fontsize=12, fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    plt.tight_layout()

    return fig


def create_loadings_plot(
    loadings: np.ndarray,
    feature_names: np.ndarray,
    loadings_components: Union[int, Sequence[int]],
    xlabel: str,
    figsize: Tuple[float, float],
    *,
    component_label: str = "PC",
) -> Figure:
    """Create loadings plot.

    This plot works for any decomposition method (PCA, PLS, ICA) that has
    loadings/components.

    Parameters
    ----------
    loadings : np.ndarray
        Loadings matrix of shape (n_features, n_components)
    feature_names : np.ndarray
        Feature names/wavenumbers/indices
    loadings_components : Union[int, Sequence[int]]
        Which component(s) to plot
    xlabel : str
        Label for x-axis (e.g., "Wavenumber (cm⁻¹)" or "Feature Index")
    figsize : Tuple[float, float]
        Figure size (width, height) in inches

    component_label : str, optional
        Prefix for component naming in titles (default "PC").

    Returns
    -------
    Figure
        Matplotlib figure with loadings plot

    Examples
    --------
    >>> loadings = np.random.rand(100, 5)
    >>> features = np.arange(100)
    >>> fig = create_loadings_plot(loadings, features, [0, 1, 2], "Feature", (10, 5))
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Convert to list if needed
    loadings_comps = (
        loadings_components
        if isinstance(loadings_components, int)
        else list(loadings_components)
    )

    loadings_plot = LoadingsPlot(
        loadings=loadings,
        feature_names=feature_names,
        components=loadings_comps,
        component_label=component_label,
    )
    loadings_plot.render(ax=ax, linewidth=1, alpha=0.7)

    # Apply decorations
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel("Loading", fontsize=10)

    if isinstance(loadings_components, int):
        title = f"{component_label}{loadings_components + 1} Loadings"
    else:
        comp_str = ", ".join([f"{component_label}{c + 1}" for c in loadings_components])
        title = f"Loadings: {comp_str}"
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3)
    plt.tight_layout()

    return fig


def create_scores_plot_single_dataset(
    component_spec: ComponentSpec,
    scores: np.ndarray,
    y: Optional[np.ndarray],
    explained_var: np.ndarray,
    dataset_name: str,
    color_by: Optional[Union[str, Dict[str, np.ndarray]]],
    annotate_by: Optional[Union[str, Dict[str, np.ndarray]]],
    figsize: Tuple[float, float],
    *,
    component_label: str = "PC",
    dataset_color: Optional[str] = None,
    confidence: float = 0.95,
    train_scores_for_ellipse: Optional[np.ndarray] = None,
    color_mode: Optional[Literal["continuous", "categorical"]] = None,
) -> Figure:
    """Create scores plot for a single dataset.

    Works for any decomposition method (PCA, PLS, ICA) that produces scores.

    Parameters
    ----------
    component_spec : ComponentSpec
        Either an int (1D plot) or tuple of two ints (2D plot)
    scores : np.ndarray
        Scores array of shape (n_samples, n_components)
    y : Optional[np.ndarray]
        Target values for coloring
    explained_var : np.ndarray
        Explained variance ratios for axis labels
    dataset_name : str
        Name of the dataset (e.g., 'train', 'test', 'val')
    color_by : str or dict, optional
        Coloring specification
    annotate_by : Optional[Union[str, Dict]]
        Annotation specification ('sample_index', 'y', or dict)
    figsize : Tuple[float, float]
        Figure size (width, height) in inches

    component_label : str, optional
        Prefix used in axis labels and titles (default "PC").
    dataset_color : Optional[str], optional
        Fixed colour for the dataset when ``color_by`` is None. When
        provided, this colour is applied to the rendered points.
    confidence : float, optional
        Confidence level for the ellipse (default 0.95).
    train_scores_for_ellipse : Optional[np.ndarray], optional
        Training scores to use for drawing confidence ellipse reference.
        If provided, a confidence ellipse will be drawn even if dataset_name != 'train'.
        If None and dataset_name == 'train', will use the scores parameter.
    color_mode : Literal["continuous", "categorical"], default="continuous"
        Mode for coloring points.

    Returns
    -------
    Figure
        Matplotlib figure with scores plot

    Examples
    --------
    >>> scores = np.random.rand(50, 5)
    >>> var_ratios = np.array([0.45, 0.25, 0.15, 0.10, 0.05])
    >>> fig = create_scores_plot_single_dataset(
    ...     (0, 1), scores, None, var_ratios, 'train', None, None, (6, 6)
    ... )
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Get number of available components
    n_components = scores.shape[1]

    # Validate component indices
    if isinstance(component_spec, int):
        if component_spec < 0 or component_spec >= n_components:
            raise ValueError(
                f"Component index {component_spec} is invalid. "
                f"Valid range: 0-{n_components - 1} (have {n_components} components)"
            )
    else:
        comp_x, comp_y = component_spec
        if comp_x < 0 or comp_x >= n_components:
            raise ValueError(
                f"Component index {comp_x} is invalid. "
                f"Valid range: 0-{n_components - 1} (have {n_components} components)"
            )
        if comp_y < 0 or comp_y >= n_components:
            raise ValueError(
                f"Component index {comp_y} is invalid. "
                f"Valid range: 0-{n_components - 1} (have {n_components} components)"
            )
        if comp_x == comp_y:
            raise ValueError(
                f"Component indices must be different, got both as {comp_x}"
            )

    # Prepare color values
    color_values = prepare_color_values(color_by, dataset_name, y, scores.shape[0])

    if isinstance(component_spec, int):
        # 1D plot: Single component vs sample index or y-value
        pc_scores = scores[:, component_spec]
        var_pct = explained_var[component_spec] * 100
        var_label = f" ({var_pct:.1f}%)" if not np.isnan(var_pct) else ""

        if color_values is not None:
            x_values = color_values
            xlabel_text = "Color Value"
            if color_by == "y":
                xlabel_text = "y-value"
            elif color_by == "sample_index":
                xlabel_text = "Sample Index"
        else:
            x_values = np.arange(len(pc_scores))
            xlabel_text = "Sample Index"

        # Create synthetic 2D data for ScoresPlot (x_values, pc_scores)
        scores_for_plot = np.column_stack([x_values, pc_scores])

        # Create and render ScoresPlot
        scores_plot = ScoresPlot(
            scores=scores_for_plot,
            components=(0, 1),
            color_by=color_values,
            label=dataset_name.capitalize(),
            color=dataset_color if color_values is None else None,
            colormap=None,
            confidence_ellipse=None,
            color_mode=color_mode,
        )
        scores_plot.render(ax)

        # Apply decorations
        ax.set_xlabel(xlabel_text, fontsize=10)
        ax.set_ylabel(f"{component_label}{component_spec + 1}{var_label}", fontsize=10)
        ax.set_title(
            f"Scores: {component_label}{component_spec + 1} ({dataset_name.capitalize()})",
            fontsize=12,
            fontweight="bold",
        )
        ax.grid(alpha=0.3)
    else:
        # 2D plot: Component pair scatter plot
        components_pair = component_spec
        var_x = explained_var[components_pair[0]] * 100
        var_y = explained_var[components_pair[1]] * 100
        var_x_label = f" ({var_x:.1f}%)" if not np.isnan(var_x) else ""
        var_y_label = f" ({var_y:.1f}%)" if not np.isnan(var_y) else ""

        # Determine which scores to use for the confidence ellipse
        ellipse_scores = train_scores_for_ellipse
        if ellipse_scores is None and dataset_name.lower() == "train":
            ellipse_scores = scores

        # First: Draw training confidence ellipse as reference (if available)
        if ellipse_scores is not None:
            ellipse_plot = ScoresPlot(
                scores=ellipse_scores,
                components=components_pair,
                color_by=None,
                label="",  # Empty label - won't show in legend
                color="red",  # Use red color for training ellipse visibility
                colormap=None,
                confidence_ellipse=confidence,
            )
            # Render only the ellipse
            ellipse_plot.render(ax)
            # Remove the scatter points from this plot (keep only ellipse)
            from matplotlib.collections import PathCollection

            for collection in ax.collections:
                if isinstance(collection, PathCollection):
                    collection.remove()
                    break

        # Create and render ScoresPlot for the actual dataset (without ellipse)
        scores_plot = ScoresPlot(
            scores=scores,
            components=components_pair,
            color_by=color_values,
            label=dataset_name.capitalize(),
            color=dataset_color if color_values is None else None,
            colormap=None,
            confidence_ellipse=None,  # Ellipse already drawn above
            color_mode=color_mode,
        )
        scores_plot.render(ax=ax)

        # Add annotations if requested
        labels = prepare_annotations(annotate_by, dataset_name, scores, y)
        if labels is not None:
            annotate_points(
                ax,
                scores[:, components_pair[0]],
                scores[:, components_pair[1]],
                labels,
                fontsize=8,
                alpha=0.7,
                xytext=(3, 3),
                textcoords="offset points",
            )

        # Apply decorations with variance percentages
        ax.set_xlabel(
            f"{component_label}{components_pair[0] + 1}{var_x_label}", fontsize=10
        )
        ax.set_ylabel(
            f"{component_label}{components_pair[1] + 1}{var_y_label}", fontsize=10
        )
        ax.set_title(
            f"Scores: {component_label}{components_pair[0] + 1} vs {component_label}{components_pair[1] + 1} ({dataset_name.capitalize()})",
            fontsize=12,
            fontweight="bold",
        )
        ax.grid(alpha=0.3)

    plt.tight_layout()
    return fig


def create_scores_plot_multi_dataset(
    component_spec: ComponentSpec,
    datasets_data: Dict[str, Dict[str, Optional[np.ndarray]]],
    explained_var: np.ndarray,
    color_by: Optional[Union[str, Dict[str, np.ndarray]]],
    annotate_by: Optional[Union[str, Dict[str, np.ndarray]]],
    figsize: Tuple[float, float],
    *,
    component_label: str = "PC",
    train_scores_for_ellipse: Optional[np.ndarray] = None,
    confidence: float = 0.95,
    color_mode: Optional[Literal["continuous", "categorical"]] = None,
) -> Figure:
    """Create scores plot with multiple datasets on same axes.

    Works for any decomposition method (PCA, PLS, ICA) that produces scores.

    Parameters
    ----------
    component_spec : ComponentSpec
        Either an int (1D plot) or tuple of two ints (2D plot)
    datasets_data : Dict[str, Dict[str, Optional[np.ndarray]]]
        Dictionary mapping dataset names to {'scores': ..., 'y': ...}
        'y' can be None
    explained_var : np.ndarray
        Explained variance ratios for axis labels
    color_by : str or dict, optional
        Coloring specification
    annotate_by : Optional[Union[str, Dict]]
        Annotation specification
    figsize : Tuple[float, float]
        Figure size (width, height) in inches
    component_label : str, optional
        Prefix used in axis labels and titles (default "PC").
    train_scores_for_ellipse : Optional[np.ndarray], optional
        Training scores to use for drawing confidence ellipse reference.
    color_mode : Literal["continuous", "categorical"], default="continuous"
        Mode for coloring points.
    confidence : float, optional
        Confidence level for the ellipse (default 0.95).

    Returns
    -------
    Figure
        Matplotlib figure with scores plot showing all datasets

    Examples
    --------
    >>> train_scores = np.random.rand(50, 5)
    >>> test_scores = np.random.rand(30, 5)
    >>> data = {
    ...     'train': {'scores': train_scores, 'y': None},
    ...     'test': {'scores': test_scores, 'y': None}
    ... }
    >>> var_ratios = np.array([0.45, 0.25, 0.15, 0.10, 0.05])
    >>> fig = create_scores_plot_multi_dataset(
    ...     (0, 1), data, var_ratios, None, None, (6, 6)
    ... )
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Get number of available components from first dataset
    first_dataset_scores = next(iter(datasets_data.values()))["scores"]
    if first_dataset_scores is None:
        raise ValueError("At least one dataset must have scores data")
    n_components = first_dataset_scores.shape[1]

    # Validate component indices
    if isinstance(component_spec, int):
        if component_spec < 0 or component_spec >= n_components:
            raise ValueError(
                f"Component index {component_spec} is invalid. "
                f"Valid range: 0-{n_components - 1} (have {n_components} components)"
            )
    else:
        comp_x, comp_y = component_spec
        if comp_x < 0 or comp_x >= n_components:
            raise ValueError(
                f"Component index {comp_x} is invalid. "
                f"Valid range: 0-{n_components - 1} (have {n_components} components)"
            )
        if comp_y < 0 or comp_y >= n_components:
            raise ValueError(
                f"Component index {comp_y} is invalid. "
                f"Valid range: 0-{n_components - 1} (have {n_components} components)"
            )
        if comp_x == comp_y:
            raise ValueError(
                f"Component indices must be different, got both as {comp_x}"
            )

    if isinstance(component_spec, int):
        # 1D plot: Single component vs sample index or y-value
        var_pct = explained_var[component_spec] * 100
        var_label = f" ({var_pct:.1f}%)" if not np.isnan(var_pct) else ""
        ylabel_text = f"{component_label}{component_spec + 1}{var_label}"
        xlabel_text = "Sample Index"

        for ds_name, data in datasets_data.items():
            scores = data["scores"]
            y = data["y"]

            # Scores should always be present
            assert scores is not None, f"Scores data is required for dataset {ds_name}"

            pc_scores = scores[:, component_spec]
            marker = DATASET_MARKERS.get(ds_name, "o")

            # Prepare color values
            color_values = prepare_color_values(color_by, ds_name, y, scores.shape[0])

            if color_values is not None:
                x_values = color_values
                xlabel_for_dataset = "Color Value"
                if color_by == "y":
                    xlabel_for_dataset = "y-value"
                    xlabel_text = "y-value"
                elif color_by == "sample_index":
                    xlabel_for_dataset = "Sample Index"
            else:
                x_values = np.arange(pc_scores.shape[0])
                xlabel_for_dataset = "Sample Index"

            scores_for_plot = np.column_stack([x_values, pc_scores])

            plot = ScoresPlot(
                scores=scores_for_plot,
                components=(0, 1),
                color_by=color_values,
                label=ds_name.capitalize(),
                color=DATASET_COLORS.get(ds_name) if color_values is None else None,
                confidence_ellipse=None,
                color_mode=color_mode,
            )
            plot.render(
                ax=ax,
                xlabel=xlabel_for_dataset,
                ylabel=ylabel_text,
                marker=marker,
                s=50,
            )

        # Apply decorations
        ax.set_xlabel(xlabel_text, fontsize=10)
        ax.set_ylabel(ylabel_text, fontsize=10)
        ax.set_title(
            f"Scores: {component_label}{component_spec + 1}",
            fontsize=12,
            fontweight="bold",
        )
        ax.grid(alpha=0.3)
        ax.legend(loc="best")
    else:
        # 2D plot: Component pair scatter plot
        components_pair = component_spec
        var_x = explained_var[components_pair[0]] * 100
        var_y = explained_var[components_pair[1]] * 100
        var_x_label = f" ({var_x:.1f}%)" if not np.isnan(var_x) else ""
        var_y_label = f" ({var_y:.1f}%)" if not np.isnan(var_y) else ""

        # First pass: Draw training confidence ellipse as reference
        # Use train_scores_for_ellipse if provided, otherwise check datasets_data
        ellipse_scores = train_scores_for_ellipse
        if ellipse_scores is None and "train" in datasets_data:
            ellipse_scores = datasets_data["train"]["scores"]

        if ellipse_scores is not None:
            # Draw only the confidence ellipse for training (invisible points)
            ellipse_plot = ScoresPlot(
                scores=ellipse_scores,
                components=components_pair,
                color_by=None,
                label="",  # Empty label - won't show in legend
                color="red",  # Use red color for training ellipse visibility
                colormap=None,
                confidence_ellipse=confidence,
            )
            # Render only the ellipse (we'll plot points separately below if train is in datasets)
            ellipse_plot.render(ax)
            # Remove the scatter points from this plot (keep only ellipse)
            from matplotlib.collections import PathCollection

            for collection in ax.collections:
                if isinstance(collection, PathCollection):
                    collection.remove()
                    break

        # Second pass: Compose multiple datasets on same axes
        for ds_name, data in datasets_data.items():
            scores = data["scores"]
            y = data["y"]

            # Scores should always be present
            assert scores is not None, f"Scores data is required for dataset {ds_name}"

            color = DATASET_COLORS.get(ds_name, "grey")
            marker = DATASET_MARKERS.get(ds_name, "grey")

            # Prepare color values
            color_values = prepare_color_values(color_by, ds_name, y, scores.shape[0])

            # Don't draw ellipse again (already drawn above)
            ellipse = None

            # Create and render ScoresPlot for this dataset
            plot = ScoresPlot(
                scores=scores,
                components=components_pair,
                color_by=color_values,
                label=ds_name.capitalize(),
                color=color if color_values is None else None,
                colormap=None,
                confidence_ellipse=ellipse,
                color_mode=color_mode,
            )
            plot.render(ax, marker=marker)

            # Add annotations if requested
            labels = prepare_annotations(annotate_by, ds_name, scores, y)
            if labels is not None:
                annotate_points(
                    ax,
                    scores[:, components_pair[0]],
                    scores[:, components_pair[1]],
                    labels,
                    fontsize=8,
                    alpha=0.7,
                    xytext=(3, 3),
                    textcoords="offset points",
                )

        # Apply decorations with variance percentages
        ax.set_xlabel(
            f"{component_label}{components_pair[0] + 1}{var_x_label}", fontsize=10
        )
        ax.set_ylabel(
            f"{component_label}{components_pair[1] + 1}{var_y_label}", fontsize=10
        )
        ax.set_title(
            f"Scores: {component_label}{components_pair[0] + 1} vs {component_label}{components_pair[1] + 1}",
            fontsize=12,
            fontweight="bold",
        )
        ax.grid(alpha=0.3)
        ax.legend(loc="best")

    plt.tight_layout()
    return fig


def create_model_distances_plot(
    datasets_data: Dict[str, Dict[str, Optional[np.ndarray]]],
    model,
    confidence: float,
    color_by: Optional[Union[str, Dict[str, np.ndarray]]],
    figsize: Tuple[float, float],
    *,
    hotelling_detector: Optional[HotellingT2] = None,
    q_residuals_detector: Optional[QResiduals] = None,
    training_dataset: str = "train",
    annotate_by: Optional[Union[str, Dict[str, np.ndarray]]] = None,
    color_mode: Optional[Literal["continuous", "categorical"]] = None,
) -> Figure:
    """Create model diagnostic distances plot across one or more datasets.

    This function renders Hotelling's T² vs Q residuals for the provided
    datasets, drawing confidence limits and optional colouring by target
    values. It replaces the previous single- and multi-dataset helpers with
    a unified implementation.

    Parameters
    ----------
    datasets_data : Dict[str, Dict[str, Optional[np.ndarray]]]
        Mapping from dataset name to a dictionary containing ``"X"``
        (required) and optional ``"y"`` arrays. The function renders each
        dataset on the same axes, applying dataset-specific colours when
        ``color_by`` is None or target values are unavailable.
    model : fitted model
        Fitted decomposition model (PCA, PLS, etc.) that provides latent
        scores used by the distance detectors.
    confidence : float
        Confidence level for the Hotelling's T² and Q residual detectors.
    color_by : str or dict, optional
        Coloring specification
    figsize : Tuple[float, float]
        Figure size (width, height) in inches.
    annotate_by : str or dict, optional
        Annotations for plot points.
    color_mode : Literal["continuous", "categorical"], default="continuous"
        Mode for coloring points.

    Other Parameters
    ----------------
    hotelling_detector : Optional[HotellingT2], default=None
        Pre-fitted Hotelling's T² detector. When provided, ``datasets_data`` is
        evaluated using this detector without refitting. When omitted, the
        function fits a fresh detector on the training dataset (see below).
    q_residuals_detector : Optional[QResiduals], default=None
        Pre-fitted Q residuals detector. Behaviour mirrors
        ``hotelling_detector``.
    training_dataset : str, default="train"
        Name of the dataset used to train the detectors when they are not
        supplied. If the named dataset is absent, the first dataset in
        ``datasets_data`` is used as a fallback.

    Returns
    -------
    Figure
        Matplotlib figure containing the composed distances plot.
    """

    if not datasets_data:
        raise ValueError("datasets_data must contain at least one dataset")

    fig, ax = plt.subplots(figsize=figsize)
    dataset_items = list(datasets_data.items())
    multi_dataset = len(dataset_items) > 1

    training_dataset_lower = training_dataset.lower()

    if hotelling_detector is None or q_residuals_detector is None:
        if not dataset_items:
            raise ValueError("datasets_data must contain at least one dataset")

        if training_dataset in datasets_data:
            train_entry = datasets_data[training_dataset]
        else:
            # Fallback to first dataset while preserving its name for limit drawing
            first_name, first_entry = dataset_items[0]
            training_dataset_lower = first_name.lower()
            train_entry = first_entry

        train_X = train_entry.get("X")

        if train_X is None:
            raise ValueError(
                "X data is required for detector fitting when detectors are not supplied"
            )

        hotelling_detector = HotellingT2(model, confidence=confidence)
        hotelling_detector.fit(train_X)

        q_residuals_detector = QResiduals(model, confidence=confidence)
        q_residuals_detector.fit(train_X)

    for ds_name, data in dataset_items:
        X = data.get("X")
        y = data.get("y")

        if X is None:
            raise ValueError(f"X data is required for dataset '{ds_name}'")

        t2 = hotelling_detector.predict_residuals(X)
        q = q_residuals_detector.predict_residuals(X)

        # Prepare color values
        color_values = prepare_color_values(color_by, ds_name, y, X.shape[0])

        if multi_dataset:
            dataset_color = DATASET_COLORS.get(
                ds_name,
            )
            marker = DATASET_MARKERS.get(ds_name, "o")
        else:
            dataset_color = None
            marker = "o"

        # Only draw confidence limits when plotting the training dataset
        should_draw_limits = (not multi_dataset) or (
            ds_name.lower() == training_dataset_lower
        )
        confidence_lines = (
            (
                hotelling_detector.critical_value_,
                q_residuals_detector.critical_value_,
            )
            if should_draw_limits
            else None
        )

        dist_plot = DistancesPlot(
            y=q,
            x=t2,
            color_by=color_values,
            label=ds_name.capitalize(),
            color=dataset_color if color_values is None else None,
            colormap=None,
            marker=marker,
            confidence_lines=confidence_lines,
            color_mode=color_mode,
        )
        dist_plot.render(ax)

        # Add annotations if requested
        labels = prepare_annotations(annotate_by, ds_name, X, y)
        if labels is not None:
            annotate_points(
                ax,
                t2,
                q,
                labels,
                fontsize=8,
                alpha=0.7,
                xytext=(3, 3),
                textcoords="offset points",
            )

    ax.set_xlabel("Hotelling's T²", fontsize=10)
    ax.set_ylabel("Q Residuals", fontsize=10)

    title_prefix = "Model Distances: Hotelling's T² vs Q Residuals"
    if multi_dataset:
        ax.set_title(title_prefix, fontsize=12, fontweight="bold")
        ax.legend(loc="best")
    else:
        dataset_name = dataset_items[0][0].capitalize()
        ax.set_title(
            f"{title_prefix} ({dataset_name})",
            fontsize=12,
            fontweight="bold",
        )

    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig


def create_q_vs_y_residuals_plot(
    datasets_data: Dict[str, Dict[str, Optional[np.ndarray]]],
    model,
    confidence: float,
    color_by: Optional[Union[str, Dict[str, np.ndarray]]],
    figsize: Tuple[float, float],
    *,
    q_residuals_detector: Optional[QResiduals] = None,
    training_dataset: str = "train",
    annotate_by: Optional[Union[str, Dict[str, np.ndarray]]] = None,
    color_mode: Literal["continuous", "categorical"] = "continuous",
) -> Figure:
    """Create Q residuals vs Y residuals diagnostic plot for regression models.

    This function renders Q residuals (SPE) vs Y residuals (prediction errors) for
    regression models with latent variables (PLS, PCR). This plot helps identify
    different types of problematic samples:
    - High Q, Low Y residuals → Poor X-space fit but good predictions
    - Low Q, High Y residuals → Good X-space fit but poor predictions
    - High Q, High Y residuals → Problems in both spaces (true outliers)
    - Low Q, Low Y residuals → Well-behaved samples

    Parameters
    ----------
    datasets_data : Dict[str, Dict[str, Optional[np.ndarray]]]
        Mapping from dataset name to a dictionary containing ``"X"``
        (required), ``"y_pred"`` (required), ``"y_true"`` (required),
        and optional ``"y"`` arrays.
        The function renders each dataset on the same axes, applying
        dataset-specific colours when ``color_by`` is None or target
        values are unavailable.
    model : fitted model
        Fitted regression model with latent variables (PLS, PCR, etc.) that
        provides both X-space reconstruction and Y predictions.
    confidence : float
        Confidence level for the Q residual detector.
    color_by : str or dict, optional
        Coloring specification
    figsize : Tuple[float, float]
        Figure size (width, height) in inches.
    annotate_by : str or dict, optional
        Annotations for plot points.
    color_mode : Literal["continuous", "categorical"], default="continuous"
        Mode for coloring points.

    Other Parameters
    ----------------
    q_residuals_detector : Optional[QResiduals], default=None
        Pre-fitted Q residuals detector. When provided, ``datasets_data`` is
        evaluated using this detector without refitting. When omitted, the
        function fits a fresh detector on the training dataset (see below).
    training_dataset : str, default="train"
        Name of the dataset used to train the detectors when they are not
        supplied. If the named dataset is absent, the first dataset in
        ``datasets_data`` is used as a fallback.

    Returns
    -------
    Figure
        Matplotlib figure containing the Q vs Y residuals plot.

    Notes
    -----
    This plot is specifically designed for regression models with latent
    variables (like PLS or PCR). It combines X-space diagnostics (Q residuals)
    with Y-space diagnostics (prediction residuals) to provide a comprehensive
    view of model fit quality.

    For multi-target regression models, Y residuals are computed as the L2 norm
    (Euclidean distance) of residuals across all targets, providing a single
    overall measure of prediction error per sample.
    """

    if not datasets_data:
        raise ValueError("datasets_data must contain at least one dataset")

    fig, ax = plt.subplots(figsize=figsize)
    dataset_items = list(datasets_data.items())
    multi_dataset = len(dataset_items) > 1

    training_dataset_lower = training_dataset.lower()

    # Fit Q residuals detector if not provided
    if q_residuals_detector is None:
        if not dataset_items:
            raise ValueError("datasets_data must contain at least one dataset")

        if training_dataset in datasets_data:
            train_entry = datasets_data[training_dataset]
        else:
            # Fallback to first dataset while preserving its name for limit drawing
            first_name, first_entry = dataset_items[0]
            training_dataset_lower = first_name.lower()
            train_entry = first_entry

        train_X = train_entry.get("X")

        if train_X is None:
            raise ValueError(
                "X data is required for detector fitting when detectors are not supplied"
            )

        q_residuals_detector = QResiduals(model, confidence=confidence)
        q_residuals_detector.fit(train_X)

    # Plot each dataset
    for ds_name, data in dataset_items:
        X = data.get("X")
        y = data.get("y")
        y_true = data.get("y_true")
        y_pred = data.get("y_pred")

        if X is None:
            raise ValueError(f"X data is required for dataset '{ds_name}'")
        if y_true is None:
            raise ValueError(f"y_true data is required for dataset '{ds_name}'")
        if y_pred is None:
            raise ValueError(f"y_pred data is required for dataset '{ds_name}'")

        q = q_residuals_detector.predict_residuals(X)

        # Calculate Y residuals as simple difference
        # Handle both single-target and multi-target regression
        y_true_arr = np.asarray(y_true)
        y_pred_arr = np.asarray(y_pred)

        if y_true_arr.ndim == 1:
            y_true_arr = y_true_arr.reshape(-1, 1)
        if y_pred_arr.ndim == 1:
            y_pred_arr = y_pred_arr.reshape(-1, 1)

        # For multi-target, use L2 norm of residuals across targets
        residuals_matrix = y_true_arr - y_pred_arr
        if residuals_matrix.shape[1] == 1:
            # Single target: just flatten
            y_residuals = residuals_matrix.ravel()
        else:
            # Multi-target: compute L2 norm across targets
            y_residuals = np.linalg.norm(residuals_matrix, axis=1)

        # When multiple datasets, always color by dataset, not by y values
        if multi_dataset:
            color_values = None
            dataset_color = DATASET_COLORS.get(ds_name)
            marker = DATASET_MARKERS.get(ds_name, "o")
        else:
            # Single dataset: respect color_by parameter
            color_values = prepare_color_values(color_by, ds_name, y, q.shape[0])
            dataset_color = None
            marker = "o"

        # Only draw Q residuals confidence limit when plotting the training dataset
        should_draw_limits = (not multi_dataset) or (
            ds_name.lower() == training_dataset_lower
        )
        confidence_lines = (
            (
                None,  # No confidence limit for Y residuals (x-axis)
                q_residuals_detector.critical_value_,  # Q residuals limit (y-axis)
            )
            if should_draw_limits
            else None
        )

        dist_plot = DistancesPlot(
            y=q,  # Q residuals on y-axis
            x=y_residuals,  # Y residuals on x-axis
            color_by=color_values,
            label=ds_name.capitalize(),
            color=dataset_color,
            colormap=None,
            marker=marker,
            confidence_lines=confidence_lines,
            color_mode=color_mode,
        )
        dist_plot.render(ax)

        # Add annotations if requested
        labels = prepare_annotations(annotate_by, ds_name, X, y)
        if labels is not None:
            annotate_points(
                ax,
                y_residuals,
                q,
                labels,
                fontsize=8,
                alpha=0.7,
                xytext=(3, 3),
                textcoords="offset points",
            )

    # Add zero line for Y residuals (vertical now since Y residuals are on x-axis)
    ax.axvline(
        x=0,
        color="black",
        linestyle="-",
        linewidth=1,
        alpha=0.5,
        zorder=1,
    )

    ax.set_xlabel("Y Residuals (Prediction Error)", fontsize=10)
    ax.set_ylabel("Q Residuals (SPE)", fontsize=10)

    title_prefix = "Regression Distances: Q Residuals vs Y Residuals"
    if multi_dataset:
        ax.set_title(title_prefix, fontsize=12, fontweight="bold")
        ax.legend(loc="best")
    else:
        dataset_name = dataset_items[0][0].capitalize()
        ax.set_title(
            f"{title_prefix} ({dataset_name})",
            fontsize=12,
            fontweight="bold",
        )

    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig


def create_x_vs_y_scores_plots(
    x_scores: np.ndarray,
    y_scores: np.ndarray,
    y_train: Optional[np.ndarray],
    components: Union[int, Tuple[int, int], Sequence[Union[int, Tuple[int, int]]]],
    color_by: Optional[Union[str, Dict[str, np.ndarray]]],
    annotate_by: Optional[Union[str, Dict[str, np.ndarray]]],
    figsize: Tuple[float, float],
    component_label: str = "LV",
    color_mode: Literal["continuous", "categorical"] = "continuous",
) -> Dict[str, Figure]:
    """Create X-scores vs Y-scores plots (typically for PLS).

    Note: Only 2D component pairs (tuples) will be plotted. Single component
    specifications (ints) will be silently skipped since X vs Y scores
    requires two components.

    Parameters
    ----------
    x_scores : np.ndarray
        X-scores array
    y_scores : np.ndarray
        Y-scores array
    y_train : np.ndarray, optional
        Target values for coloring
    components : int, tuple, or sequence
        Component pairs to plot. Only tuple specifications will be used.
    color_by : str or dict, optional
        Coloring specification
    annotate_by : str or dict, optional
        Annotation specification
    figsize : tuple of float
        Figure size
    component_label : str, default="LV"
        Label for components (e.g., "LV", "PC")
    color_mode : Literal["continuous", "categorical"], default="continuous"
        Mode for coloring points.

    Returns
    -------
    dict
        Dictionary of figures with keys like 'x_vs_y_scores_1', 'x_vs_y_scores_2', etc.
    """
    components_list = normalize_components(components)
    figures: Dict[str, Figure] = {}

    for idx, component_spec in enumerate(components_list, start=1):
        # Only create 2D plots (component pairs)
        if isinstance(component_spec, tuple):
            fig, ax = plt.subplots(figsize=figsize)

            # Create combined scores array [X-score, Y-score]
            combined_scores = np.column_stack(
                [
                    x_scores[:, component_spec[0]],
                    y_scores[:, component_spec[1]],
                ]
            )

            # Determine color_by parameter
            color_values = prepare_color_values(
                color_by, "train", y_train, x_scores.shape[0]
            )

            # Create ScoresPlot
            plot = ScoresPlot(
                scores=combined_scores,
                components=(0, 1),  # We already selected the right columns
                color_by=color_values,
                label="Train",
                colormap=None,
                confidence_ellipse=None,
                color_mode=color_mode,
            )
            plot.render(ax)

            # Add annotations if requested
            labels = prepare_annotations(annotate_by, "train", x_scores, y_train)
            if labels is not None:
                annotate_points(
                    ax,
                    combined_scores[:, 0],
                    combined_scores[:, 1],
                    labels,
                    fontsize=8,
                    alpha=0.7,
                    xytext=(3, 3),
                    textcoords="offset points",
                )

            # Set custom labels
            ax.set_xlabel(f"X-{component_label}{component_spec[0] + 1}", fontsize=10)
            ax.set_ylabel(f"Y-{component_label}{component_spec[1] + 1}", fontsize=10)
            ax.set_title(
                f"X-scores vs Y-scores: {component_label}{component_spec[0] + 1} vs {component_label}{component_spec[1] + 1}",
                fontsize=12,
                fontweight="bold",
            )
            ax.grid(alpha=0.3)

            plt.tight_layout()
            figures[f"x_vs_y_scores_{idx}"] = fig

    return figures
