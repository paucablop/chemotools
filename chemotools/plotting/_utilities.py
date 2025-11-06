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


def detect_categorical(color_by: np.ndarray) -> bool:
    """Detect if color_by array should be treated as categorical.

    Parameters
    ----------
    color_by : np.ndarray
        The color reference array to analyze.

    Returns
    -------
    bool
        True if the array should be treated as categorical.

    Notes
    -----
    Detection logic:
    1. String types (U, S, O) → categorical
    2. Boolean type → categorical
    3. Integer type with ≤ 10 unique values → categorical
    4. Float type with ≤ 5 unique values AND all values repeat → categorical
    5. Otherwise → continuous

    Examples
    --------
    >>> classes = np.array(['A', 'B', 'A', 'C'])
    >>> detect_categorical(classes)
    True

    >>> concentrations = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    >>> detect_categorical(concentrations)
    False

    >>> levels = np.array([1, 1, 2, 2, 3, 3])
    >>> detect_categorical(levels)
    True
    """
    # String or object types are categorical
    if color_by.dtype.kind in ["U", "S", "O"]:
        return True

    # Boolean is categorical
    if color_by.dtype.kind == "b":
        return True

    unique_values = np.unique(color_by)
    n_unique = len(unique_values)

    # Integer types with reasonable number of unique values
    if color_by.dtype.kind in ["i", "u"]:  # signed or unsigned int
        return n_unique <= 10

    # Float types: only categorical if very few unique values AND repeated
    if color_by.dtype.kind == "f":
        if n_unique <= 5:
            counts = np.bincount(np.searchsorted(unique_values, color_by))
            has_repeats = bool(np.any(counts > 1))
            return has_repeats

    return False


def get_default_colormap(is_categorical: bool, colormap: Optional[str] = None) -> str:
    """Get appropriate colormap for categorical or continuous data.

    Parameters
    ----------
    is_categorical : bool
        Whether the data is categorical or continuous.
    colormap : str, optional
        User-specified colormap. If provided, this is returned as-is.

    Returns
    -------
    str
        The colormap name to use.

    Notes
    -----
    Defaults are colorblind-friendly:
    - "tab10" for categorical data
    - "viridis" for continuous data

    Examples
    --------
    >>> get_default_colormap(is_categorical=True)
    'tab10'

    >>> get_default_colormap(is_categorical=False)
    'viridis'

    >>> get_default_colormap(is_categorical=True, colormap='Set2')
    'Set2'
    """
    if colormap is not None:
        return colormap
    return "tab10" if is_categorical else "viridis"


def add_colorbar(
    ax: Axes,
    color_by: np.ndarray,
    colormap: str,
    label: str = "Reference Value",
) -> None:
    """Add a colorbar to the axes for continuous coloring.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes to add the colorbar to.
    color_by : np.ndarray
        The continuous values used for coloring.
    colormap : str
        Name of the colormap to use.
    label : str, optional
        Label for the colorbar (default: "Reference Value").

    Examples
    --------
    >>> add_colorbar(ax, concentrations, 'viridis', 'Concentration (mg/L)')
    """
    from matplotlib import cm
    import matplotlib.colors as mcolors

    norm = mcolors.Normalize(vmin=color_by.min(), vmax=color_by.max())
    sm = cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label(label, fontsize=10)
