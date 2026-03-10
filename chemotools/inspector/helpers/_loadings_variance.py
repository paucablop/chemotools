"""Loadings and variance plot creation functions for inspectors."""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from matplotlib.figure import Figure

from chemotools.plotting import ExplainedVariancePlot, LoadingsPlot

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def create_variance_plot(
    explained_variance_ratio: np.ndarray,
    variance_threshold: float,
    figsize: Tuple[float, float],
) -> Figure:
    """Create explained variance plot.

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

    Parameters
    ----------
    loadings : np.ndarray
        Loadings matrix of shape (n_features, n_components)
    feature_names : np.ndarray
        Feature names/wavenumbers/indices
    loadings_components : Union[int, Sequence[int]]
        Which component(s) to plot
    xlabel : str
        Label for x-axis
    figsize : Tuple[float, float]
        Figure size (width, height) in inches
    component_label : str, optional
        Prefix for component naming in titles (default "PC").

    Returns
    -------
    Figure
        Matplotlib figure with loadings plot
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
