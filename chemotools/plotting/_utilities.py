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
    ax: Axes,
    x: np.ndarray,
    y: np.ndarray,
    confidence: float = 0.95,
    n_std: Optional[float] = None,
    facecolor: str = "none",
    edgecolor: Optional[str] = None,
    **kwargs,
) -> None:
    """Add confidence ellipse to a scatter plot.

    Draws an ellipse representing the confidence region for bivariate data.
    Can be based on either confidence level (using chi-square distribution)
    or standard deviations.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes to draw the ellipse on.
    x : np.ndarray
        X-coordinates of the data points.
    y : np.ndarray
        Y-coordinates of the data points.
    confidence : float, optional
        Confidence level for the ellipse (default: 0.95 = 95%).
        Used only if n_std is None. Common values: 0.90, 0.95, 0.99.
    n_std : float, optional
        Number of standard deviations for the ellipse radius.
        If provided, overrides the confidence parameter.
        Common values: 1, 2, 3 (for 1σ, 2σ, 3σ ellipses).
    facecolor : str, optional
        Face color of the ellipse (default: "none" for transparent).
    edgecolor : str, optional
        Edge color of the ellipse. If None, uses the current color cycle.
    **kwargs : Any
        Additional keyword arguments passed to matplotlib.patches.Ellipse.
        Common options: linewidth, linestyle, alpha, label.

    Examples
    --------
    Add a 95% confidence ellipse:

    >>> add_confidence_ellipse(ax, x, y, confidence=0.95, edgecolor='red')

    Add a 2-sigma ellipse:

    >>> add_confidence_ellipse(ax, x, y, n_std=2, edgecolor='blue', linewidth=2)

    Add multiple ellipses with different confidence levels:

    >>> add_confidence_ellipse(ax, x, y, confidence=0.95, edgecolor='red', label='95%')
    >>> add_confidence_ellipse(ax, x, y, confidence=0.99, edgecolor='blue', label='99%')

    Notes
    -----
    The ellipse is computed using the covariance matrix of the data.
    For multivariate normal data, this represents the confidence region
    based on the chi-square distribution with 2 degrees of freedom.
    """
    from matplotlib.patches import Ellipse
    from scipy import stats

    if len(x) != len(y):
        raise ValueError(
            f"x and y must have the same length, got {len(x)} and {len(y)}"
        )

    if len(x) < 3:
        raise ValueError(f"Need at least 3 points to compute ellipse, got {len(x)}")

    # Calculate the mean
    mean_x = np.mean(x)
    mean_y = np.mean(y)

    # Calculate the covariance matrix
    cov = np.cov(x, y)

    # Calculate eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Sort eigenvalues and eigenvectors in descending order
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    # Calculate the angle of the ellipse
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))

    # Determine the scale factor
    if n_std is not None:
        # Use standard deviations
        scale_factor = n_std
    else:
        # Use confidence level with chi-square distribution (2 DOF for bivariate)
        scale_factor = np.sqrt(stats.chi2.ppf(confidence, df=2))

    # Width and height are 2 * scale_factor * sqrt(eigenvalue)
    width = 2 * scale_factor * np.sqrt(eigenvalues[0])
    height = 2 * scale_factor * np.sqrt(eigenvalues[1])

    # Create the ellipse
    ellipse = Ellipse(
        xy=(mean_x, mean_y),
        width=width,
        height=height,
        angle=angle,
        facecolor=facecolor,
        edgecolor=edgecolor,
        **kwargs,
    )

    ax.add_patch(ellipse)


def annotate_points(
    ax: Axes, x: np.ndarray, y: np.ndarray, labels: Union[np.ndarray, list], **kwargs
) -> None:
    """Annotate points on a plot."""
    for xi, yi, label in zip(x, y, labels):
        ax.annotate(label, (xi, yi), **kwargs)
