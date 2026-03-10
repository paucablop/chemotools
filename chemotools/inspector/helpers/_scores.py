"""Scores plot creation functions for inspectors."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Literal, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PathCollection

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

from chemotools.plotting import ScoresPlot
from chemotools.plotting._styles import DATASET_COLORS, DATASET_MARKERS
from chemotools.plotting._utils import annotate_points

from ..core.utils import (
    ComponentSpec,
    normalize_components,
    prepare_annotations,
    prepare_color_values,
)

# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _validate_component_spec(component_spec: ComponentSpec, n_components: int) -> None:
    """Raise ``ValueError`` if component indices are invalid."""
    indices = (component_spec,) if isinstance(component_spec, int) else component_spec
    for idx in indices:
        if idx < 0 or idx >= n_components:
            raise ValueError(
                f"Component index {idx} is invalid. "
                f"Valid range: 0-{n_components - 1} "
                f"(have {n_components} components)"
            )
    if not isinstance(component_spec, int) and component_spec[0] == component_spec[1]:
        raise ValueError(
            f"Component indices must be different, got both as {component_spec[0]}"
        )


def _draw_confidence_ellipse(
    ax: Axes,
    scores: np.ndarray,
    components_pair: Tuple[int, int],
    confidence: float,
) -> None:
    """Draw a confidence ellipse on *ax* without leaving scatter points behind."""
    ellipse_plot = ScoresPlot(
        scores=scores,
        components=components_pair,
        color_by=None,
        label="",
        color="red",
        colormap=None,
        confidence_ellipse=confidence,
    )
    ellipse_plot.render(ax)
    # Remove the scatter points that ScoresPlot renders alongside the ellipse
    for collection in ax.collections:
        if isinstance(collection, PathCollection):
            collection.remove()
            break


def _render_scores_1d(
    ax: Axes,
    component: int,
    scores: np.ndarray,
    y: Optional[np.ndarray],
    explained_var: np.ndarray,
    dataset_name: str,
    color_by: Optional[Union[str, Dict[str, np.ndarray]]],
    component_label: str,
    dataset_color: Optional[str],
    color_mode: Optional[Literal["continuous", "categorical"]],
) -> None:
    """Render a 1-D scores plot (single component vs sample index / colour)."""
    color_values = prepare_color_values(color_by, dataset_name, y, scores.shape[0])

    pc_scores = scores[:, component]
    var_pct = explained_var[component] * 100
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

    scores_for_plot = np.column_stack([x_values, pc_scores])

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

    ax.set_xlabel(xlabel_text, fontsize=10)
    ax.set_ylabel(f"{component_label}{component + 1}{var_label}", fontsize=10)
    ax.set_title(
        f"Scores: {component_label}{component + 1} ({dataset_name.capitalize()})",
        fontsize=12,
        fontweight="bold",
    )
    ax.grid(alpha=0.3)


def _render_scores_2d(
    ax: Axes,
    components_pair: Tuple[int, int],
    scores: np.ndarray,
    y: Optional[np.ndarray],
    explained_var: np.ndarray,
    dataset_name: str,
    color_by: Optional[Union[str, Dict[str, np.ndarray]]],
    annotate_by: Optional[Union[str, Dict[str, np.ndarray]]],
    component_label: str,
    dataset_color: Optional[str],
    confidence: float,
    train_scores_for_ellipse: Optional[np.ndarray],
    color_mode: Optional[Literal["continuous", "categorical"]],
) -> None:
    """Render a 2-D scores scatter plot with optional confidence ellipse."""
    color_values = prepare_color_values(color_by, dataset_name, y, scores.shape[0])

    var_x = explained_var[components_pair[0]] * 100
    var_y = explained_var[components_pair[1]] * 100
    var_x_label = f" ({var_x:.1f}%)" if not np.isnan(var_x) else ""
    var_y_label = f" ({var_y:.1f}%)" if not np.isnan(var_y) else ""

    # Determine which scores to use for the confidence ellipse
    ellipse_scores = train_scores_for_ellipse
    if ellipse_scores is None and dataset_name.lower() == "train":
        ellipse_scores = scores

    if ellipse_scores is not None:
        _draw_confidence_ellipse(ax, ellipse_scores, components_pair, confidence)

    # Render actual dataset points (without ellipse — already drawn above)
    scores_plot = ScoresPlot(
        scores=scores,
        components=components_pair,
        color_by=color_values,
        label=dataset_name.capitalize(),
        color=dataset_color if color_values is None else None,
        colormap=None,
        confidence_ellipse=None,
        color_mode=color_mode,
    )
    scores_plot.render(ax=ax)

    # Annotations
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

    ax.set_xlabel(
        f"{component_label}{components_pair[0] + 1}{var_x_label}", fontsize=10
    )
    ax.set_ylabel(
        f"{component_label}{components_pair[1] + 1}{var_y_label}", fontsize=10
    )
    ax.set_title(
        f"Scores: {component_label}"
        f"{components_pair[0] + 1} vs "
        f"{component_label}"
        f"{components_pair[1] + 1}"
        f" ({dataset_name.capitalize()})",
        fontsize=12,
        fontweight="bold",
    )
    ax.grid(alpha=0.3)


def _render_multi_scores_1d(
    ax: Axes,
    component: int,
    datasets_data: Dict[str, Dict[str, Optional[np.ndarray]]],
    explained_var: np.ndarray,
    color_by: Optional[Union[str, Dict[str, np.ndarray]]],
    component_label: str,
    color_mode: Optional[Literal["continuous", "categorical"]],
) -> None:
    """Render 1-D scores for multiple datasets on *ax*."""
    var_pct = explained_var[component] * 100
    var_label = f" ({var_pct:.1f}%)" if not np.isnan(var_pct) else ""
    ylabel_text = f"{component_label}{component + 1}{var_label}"
    xlabel_text = "Sample Index"

    for ds_name, data in datasets_data.items():
        scores = data["scores"]
        y = data["y"]
        if scores is None:
            raise ValueError(f"Scores data is required for dataset {ds_name}")

        pc_scores = scores[:, component]
        marker = DATASET_MARKERS.get(ds_name, "o")
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

    ax.set_xlabel(xlabel_text, fontsize=10)
    ax.set_ylabel(ylabel_text, fontsize=10)
    ax.set_title(
        f"Scores: {component_label}{component + 1}",
        fontsize=12,
        fontweight="bold",
    )
    ax.grid(alpha=0.3)
    ax.legend(loc="best")


def _render_multi_scores_2d(
    ax: Axes,
    components_pair: Tuple[int, int],
    datasets_data: Dict[str, Dict[str, Optional[np.ndarray]]],
    explained_var: np.ndarray,
    color_by: Optional[Union[str, Dict[str, np.ndarray]]],
    annotate_by: Optional[Union[str, Dict[str, np.ndarray]]],
    component_label: str,
    train_scores_for_ellipse: Optional[np.ndarray],
    confidence: float,
    color_mode: Optional[Literal["continuous", "categorical"]],
) -> None:
    """Render 2-D scores scatter for multiple datasets on *ax*."""
    var_x = explained_var[components_pair[0]] * 100
    var_y = explained_var[components_pair[1]] * 100
    var_x_label = f" ({var_x:.1f}%)" if not np.isnan(var_x) else ""
    var_y_label = f" ({var_y:.1f}%)" if not np.isnan(var_y) else ""

    # Draw training confidence ellipse as reference
    ellipse_scores = train_scores_for_ellipse
    if ellipse_scores is None and "train" in datasets_data:
        ellipse_scores = datasets_data["train"]["scores"]

    if ellipse_scores is not None:
        _draw_confidence_ellipse(ax, ellipse_scores, components_pair, confidence)

    # Compose multiple datasets on same axes
    for ds_name, data in datasets_data.items():
        scores = data["scores"]
        y = data["y"]
        if scores is None:
            raise ValueError(f"Scores data is required for dataset {ds_name}")

        color = DATASET_COLORS.get(ds_name, "grey")
        marker = DATASET_MARKERS.get(ds_name, "o")
        color_values = prepare_color_values(color_by, ds_name, y, scores.shape[0])

        plot = ScoresPlot(
            scores=scores,
            components=components_pair,
            color_by=color_values,
            label=ds_name.capitalize(),
            color=color if color_values is None else None,
            colormap=None,
            confidence_ellipse=None,
            color_mode=color_mode,
        )
        plot.render(ax, marker=marker)

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

    ax.set_xlabel(
        f"{component_label}{components_pair[0] + 1}{var_x_label}", fontsize=10
    )
    ax.set_ylabel(
        f"{component_label}{components_pair[1] + 1}{var_y_label}", fontsize=10
    )
    ax.set_title(
        f"Scores: {component_label}"
        f"{components_pair[0] + 1} vs "
        f"{component_label}"
        f"{components_pair[1] + 1}",
        fontsize=12,
        fontweight="bold",
    )
    ax.grid(alpha=0.3)
    ax.legend(loc="best")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


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
        Fixed colour for the dataset when ``color_by`` is None.
    confidence : float, optional
        Confidence level for the ellipse (default 0.95).
    train_scores_for_ellipse : Optional[np.ndarray], optional
        Training scores to use for drawing confidence ellipse reference.
    color_mode : Literal["continuous", "categorical"], optional
        Mode for coloring points.

    Returns
    -------
    Figure
        Matplotlib figure with scores plot
    """
    fig, ax = plt.subplots(figsize=figsize)

    _validate_component_spec(component_spec, scores.shape[1])

    if isinstance(component_spec, int):
        _render_scores_1d(
            ax,
            component_spec,
            scores,
            y,
            explained_var,
            dataset_name,
            color_by,
            component_label,
            dataset_color,
            color_mode,
        )
    else:
        _render_scores_2d(
            ax,
            component_spec,
            scores,
            y,
            explained_var,
            dataset_name,
            color_by,
            annotate_by,
            component_label,
            dataset_color,
            confidence,
            train_scores_for_ellipse,
            color_mode,
        )

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
        Training scores for confidence ellipse reference.
    confidence : float, optional
        Confidence level for the ellipse (default 0.95).
    color_mode : Literal["continuous", "categorical"], optional
        Mode for coloring points.

    Returns
    -------
    Figure
        Matplotlib figure with scores plot showing all datasets
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Get number of available components from first dataset
    first_dataset_scores = next(iter(datasets_data.values()))["scores"]
    if first_dataset_scores is None:
        raise ValueError("At least one dataset must have scores data")

    _validate_component_spec(component_spec, first_dataset_scores.shape[1])

    if isinstance(component_spec, int):
        _render_multi_scores_1d(
            ax,
            component_spec,
            datasets_data,
            explained_var,
            color_by,
            component_label,
            color_mode,
        )
    else:
        _render_multi_scores_2d(
            ax,
            component_spec,
            datasets_data,
            explained_var,
            color_by,
            annotate_by,
            component_label,
            train_scores_for_ellipse,
            confidence,
            color_mode,
        )

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

    Only 2D component pairs (tuples) will be plotted. Single component
    specifications (ints) will be silently skipped.

    Parameters
    ----------
    x_scores : np.ndarray
        X-scores array
    y_scores : np.ndarray
        Y-scores array
    y_train : np.ndarray, optional
        Target values for coloring
    components : int, tuple, or sequence
        Component pairs to plot.
    color_by : str or dict, optional
        Coloring specification
    annotate_by : str or dict, optional
        Annotation specification
    figsize : tuple of float
        Figure size
    component_label : str, default="LV"
        Label for components
    color_mode : Literal["continuous", "categorical"], default="continuous"
        Mode for coloring points.

    Returns
    -------
    dict
        Dictionary of figures with keys like 'x_vs_y_scores_1', etc.
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

            color_values = prepare_color_values(
                color_by, "train", y_train, x_scores.shape[0]
            )

            plot = ScoresPlot(
                scores=combined_scores,
                components=(0, 1),
                color_by=color_values,
                label="Train",
                colormap=None,
                confidence_ellipse=None,
                color_mode=color_mode,
            )
            plot.render(ax)

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

            ax.set_xlabel(f"X-{component_label}{component_spec[0] + 1}", fontsize=10)
            ax.set_ylabel(f"Y-{component_label}{component_spec[1] + 1}", fontsize=10)
            ax.set_title(
                f"X-scores vs Y-scores: "
                f"{component_label}"
                f"{component_spec[0] + 1} vs "
                f"{component_label}"
                f"{component_spec[1] + 1}",
                fontsize=12,
                fontweight="bold",
            )
            ax.grid(alpha=0.3)

            plt.tight_layout()
            figures[f"x_vs_y_scores_{idx}"] = fig

    return figures
