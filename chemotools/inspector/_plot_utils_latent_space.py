"""Core plot creation functions for inspectors.

This module contains reusable plotting functions for scores, loadings,
and variance plots that are common across PCA, PLS, ICA, and other
decomposition methods.
"""

from __future__ import annotations
from typing import Dict, Union, Optional, Tuple, Sequence, TYPE_CHECKING
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
from chemotools.plotting._utils import (
    add_colorbar,
    annotate_points,
    detect_categorical,
    get_colors_from_labels,
    get_default_colormap,
)
from chemotools.plotting._styles import DATASET_COLORS, DATASET_MARKERS
from chemotools.outliers import HotellingT2, QResiduals

from ._utils import (
    ComponentSpec,
    prepare_annotations,
    select_primary_target,
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
    )
    loadings_plot.render(ax=ax, linewidth=2, alpha=0.7)

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
    color_by_y: bool,
    annotate_by: Optional[Union[str, Dict[str, np.ndarray]]],
    figsize: Tuple[float, float],
    *,
    component_label: str = "PC",
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
    color_by_y : bool
        Whether to color points by y values
    annotate_by : Optional[Union[str, Dict]]
        Annotation specification ('sample_index', 'y', or dict)
    figsize : Tuple[float, float]
        Figure size (width, height) in inches

    component_label : str, optional
        Prefix used in axis labels and titles (default "PC").

    Returns
    -------
    Figure
        Matplotlib figure with scores plot

    Examples
    --------
    >>> scores = np.random.rand(50, 5)
    >>> var_ratios = np.array([0.45, 0.25, 0.15, 0.10, 0.05])
    >>> fig = create_scores_plot_single_dataset(
    ...     (0, 1), scores, None, var_ratios, 'train', False, None, (6, 6)
    ... )
    """
    fig, ax = plt.subplots(figsize=figsize)

    color_reference = select_primary_target(y) if color_by_y else None

    if isinstance(component_spec, int):
        # 1D plot: Single component vs sample index or y-value
        pc_scores = scores[:, component_spec]
        var_pct = explained_var[component_spec] * 100

        if color_reference is not None:
            x_values = color_reference

            if x_values.shape[0] != pc_scores.shape[0]:
                raise ValueError(
                    "Length of target values must match number of samples. "
                    f"Got {x_values.shape[0]} vs {pc_scores.shape[0]}."
                )

            color_array = np.asarray(x_values)
            is_categorical = detect_categorical(color_array)

            if is_categorical:
                colormap = get_default_colormap(True, None)
                colors = get_colors_from_labels(color_array, colormap)
                unique_values = np.unique(color_array)
                for value in unique_values:
                    mask = color_array == value
                    ax.scatter(
                        x_values[mask],
                        pc_scores[mask],
                        color=colors[mask][0],
                        alpha=0.7,
                        s=50,
                        label=f"{dataset_name.capitalize()} - {value}",
                    )
                ax.legend(loc="best")
            else:
                colormap = get_default_colormap(False, None)
                ax.scatter(
                    x_values,
                    pc_scores,
                    c=color_array,
                    cmap=colormap,
                    alpha=0.7,
                    s=50,
                )
                add_colorbar(ax, color_array, colormap, label="y-value")

            xlabel_text = "y-value"
        else:
            # Plot PC score vs sample index
            ax.scatter(range(len(pc_scores)), pc_scores, alpha=0.7, s=50)
            xlabel_text = "Sample Index"

        # Apply decorations
        ax.set_xlabel(xlabel_text, fontsize=10)
        ax.set_ylabel(
            f"{component_label}{component_spec + 1} ({var_pct:.1f}%)", fontsize=10
        )
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

        # Determine color_by parameter
        color_by = color_reference if color_reference is not None else None

        # Create and render ScoresPlot
        scores_plot = ScoresPlot(
            scores=scores,
            components=components_pair,
            color_by=color_by,
            label=dataset_name.capitalize(),
            colormap=None,
            confidence_ellipse=0.95,  # Always show 95% confidence ellipse
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
            f"{component_label}{components_pair[0] + 1} ({var_x:.1f}%)", fontsize=10
        )
        ax.set_ylabel(
            f"{component_label}{components_pair[1] + 1} ({var_y:.1f}%)", fontsize=10
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
    color_by_y: bool,
    annotate_by: Optional[Union[str, Dict[str, np.ndarray]]],
    figsize: Tuple[float, float],
    *,
    component_label: str = "PC",
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
    color_by_y : bool
        Whether to color by y values (when single dataset and y available)
    annotate_by : Optional[Union[str, Dict]]
        Annotation specification
    figsize : Tuple[float, float]
        Figure size (width, height) in inches

    component_label : str, optional
        Prefix used in axis labels and titles (default "PC").

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
    ...     (0, 1), data, var_ratios, False, None, (6, 6)
    ... )
    """
    fig, ax = plt.subplots(figsize=figsize)

    if isinstance(component_spec, int):
        # 1D plot: Single component vs sample index or y-value
        var_pct = explained_var[component_spec] * 100
        ylabel_text = f"{component_label}{component_spec + 1} ({var_pct:.1f}%)"
        xlabel_text = "Sample Index"

        for ds_name, data in datasets_data.items():
            scores = data["scores"]
            y = data["y"]

            # Scores should always be present
            assert scores is not None, f"Scores data is required for dataset {ds_name}"

            pc_scores = scores[:, component_spec]
            marker = DATASET_MARKERS.get(ds_name, "o")

            y_values = select_primary_target(y) if y is not None else None

            if color_by_y and y_values is not None:
                x_values = y_values
                xlabel_for_dataset = "y-value"
                xlabel_text = "y-value"
            else:
                x_values = np.arange(pc_scores.shape[0])
                xlabel_for_dataset = "Sample Index"

            scores_for_plot = np.column_stack([x_values, pc_scores])

            plot = ScoresPlot(
                scores=scores_for_plot,
                components=(0, 1),
                color_by=None,
                label=ds_name.capitalize(),
                color=DATASET_COLORS.get(ds_name, "#7f7f7f"),
                confidence_ellipse=None,
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

        # Compose multiple datasets on same axes
        for ds_name, data in datasets_data.items():
            scores = data["scores"]
            y = data["y"]

            # Scores should always be present
            assert scores is not None, f"Scores data is required for dataset {ds_name}"

            color = DATASET_COLORS.get(ds_name, "#7f7f7f")

            # Determine color_by parameter
            color_reference = (
                select_primary_target(y) if (color_by_y and y is not None) else None
            )
            color_by = color_reference

            # Create and render ScoresPlot for this dataset
            plot = ScoresPlot(
                scores=scores,
                components=components_pair,
                color_by=color_by,
                label=ds_name.capitalize(),
                color=color if color_by is None else None,
                colormap=None,
                confidence_ellipse=0.95,  # Always show 95% confidence ellipse
            )
            plot.render(ax)

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
            f"{component_label}{components_pair[0] + 1} ({var_x:.1f}%)", fontsize=10
        )
        ax.set_ylabel(
            f"{component_label}{components_pair[1] + 1} ({var_y:.1f}%)", fontsize=10
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
    color_by_y: bool,
    figsize: Tuple[float, float],
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
        ``color_by_y`` is False or target values are unavailable.
    model : fitted model
        Fitted decomposition model (PCA, PLS, etc.) that provides latent
        scores used by the distance detectors.
    confidence : float
        Confidence level for the Hotelling's T² and Q residual detectors.
    color_by_y : bool
        Whether to colour points using the provided ``y`` targets.
    figsize : Tuple[float, float]
        Figure size (width, height) in inches.

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

    for ds_name, data in dataset_items:
        X = data.get("X")
        y = data.get("y")

        if X is None:
            raise ValueError(f"X data is required for dataset '{ds_name}'")

        # Calculate Hotelling's T² residuals
        hotelling = HotellingT2(model, confidence=confidence)
        hotelling.fit(X)
        t2 = hotelling.predict_residuals(X)

        # Calculate Q residuals
        q_res_model = QResiduals(model, confidence=confidence)
        q_res_model.fit(X)
        q = q_res_model.predict_residuals(X)

        color_by = select_primary_target(y) if (color_by_y and y is not None) else None
        dataset_color = (
            DATASET_COLORS.get(ds_name, "#7f7f7f")
            if color_by is None and multi_dataset
            else None
        )

        # Only draw confidence limits when plotting the training dataset
        should_draw_limits = (not multi_dataset) or (ds_name.lower() == "train")
        confidence_lines = (
            (
                hotelling.critical_value_,
                q_res_model.critical_value_,
            )
            if should_draw_limits
            else None
        )

        dist_plot = DistancesPlot(
            y=q,
            x=t2,
            color_by=color_by,
            label=ds_name.capitalize(),
            color=dataset_color,
            colormap=None,
            confidence_lines=confidence_lines,
        )
        dist_plot.render(ax)

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
