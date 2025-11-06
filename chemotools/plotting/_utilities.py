"""Core plotting utilities for chemotools visualizations."""

from typing import Optional, Union
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes


def setup_figure(
    figsize: tuple = (10, 8),
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    **kwargs,
) -> tuple[Figure, Axes]:
    """Create a figure with consistent styling."""
    fig, ax = plt.subplots(figsize=figsize, **kwargs)
    if title:
        ax.set_title(title, fontsize=14, fontweight="bold")
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=12)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(alpha=0.3)
    return fig, ax


def get_colors_from_labels(
    labels: Union[np.ndarray, list], colormap: str = "tab10"
) -> np.ndarray:
    """Convert labels to colors using a colormap."""
    unique_labels = np.unique(labels)
    cmap = plt.get_cmap(colormap)
    label_to_color = {
        label: cmap(i / len(unique_labels)) for i, label in enumerate(unique_labels)
    }
    return np.array([label_to_color[label] for label in labels])


def add_confidence_ellipse(
    ax: Axes, x: np.ndarray, y: np.ndarray, confidence: float = 0.95, **kwargs
) -> None:
    """Add confidence ellipse to a scatter plot."""
    # Implementation for Hotelling T² ellipse
    pass


def annotate_points(
    ax: Axes, x: np.ndarray, y: np.ndarray, labels: Union[np.ndarray, list], **kwargs
) -> None:
    """Annotate points on a plot."""
    for xi, yi, label in zip(x, y, labels):
        ax.annotate(label, (xi, yi), **kwargs)
