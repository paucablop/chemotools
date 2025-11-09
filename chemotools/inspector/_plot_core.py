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

from chemotools.plotting import ExplainedVariancePlot, LoadingsPlot, ScoresPlot
from chemotools.plotting._utils import annotate_points
from chemotools.plotting._styles import DATASET_COLORS, DATASET_MARKERS

from ._utils import (
    ComponentSpec,
    prepare_annotations,
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

    if isinstance(component_spec, int):
        # 1D plot: Single component vs sample index or y-value
        pc_scores = scores[:, component_spec]
        var_pct = explained_var[component_spec] * 100

        if color_by_y and y is not None:
            # Plot PC score vs y-value
            scatter = ax.scatter(y, pc_scores, c=y, cmap="viridis", alpha=0.7, s=50)
            plt.colorbar(scatter, ax=ax, label="y-value")
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
        color_by = y if (color_by_y and y is not None) else None

        # Create and render ScoresPlot
        scores_plot = ScoresPlot(
            scores=scores,
            components=components_pair,
            color_by=color_by,
            label=dataset_name.capitalize(),
            colormap="viridis" if color_by is not None else None,
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

        for ds_name, data in datasets_data.items():
            scores = data["scores"]
            y = data["y"]

            # Scores should always be present
            assert scores is not None, f"Scores data is required for dataset {ds_name}"

            pc_scores = scores[:, component_spec]
            color = DATASET_COLORS.get(ds_name, "#7f7f7f")
            marker = DATASET_MARKERS.get(ds_name, "o")

            if color_by_y and y is not None:
                # Plot PC score vs y-value
                ax.scatter(
                    y,
                    pc_scores,
                    c=color,
                    marker=marker,
                    alpha=0.7,
                    s=50,
                    label=ds_name.capitalize(),
                )
                xlabel_text = "y-value"
            else:
                # Plot PC score vs sample index
                ax.scatter(
                    range(len(pc_scores)),
                    pc_scores,
                    c=color,
                    marker=marker,
                    alpha=0.7,
                    s=50,
                    label=ds_name.capitalize(),
                )
                xlabel_text = "Sample Index"

        # Apply decorations
        ax.set_xlabel(xlabel_text, fontsize=10)
        ax.set_ylabel(
            f"{component_label}{component_spec + 1} ({var_pct:.1f}%)", fontsize=10
        )
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
            color_by = y if (color_by_y and y is not None) else None

            # Create and render ScoresPlot for this dataset
            plot = ScoresPlot(
                scores=scores,
                components=components_pair,
                color_by=color_by,
                label=ds_name.capitalize(),
                color=color if color_by is None else None,
                colormap="viridis" if color_by is not None else None,
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
