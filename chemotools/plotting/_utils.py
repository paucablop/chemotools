"""Core plotting utilities for chemotools visualizations."""

from typing import Optional, Union, Iterable, cast, Any
import inspect
import numpy as np
from sklearn.utils import check_array
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from chemotools.plotting._styles import CUSTOM_CMAP

# Register custom colormaps
try:
    import matplotlib as mpl

    # Check if already registered to avoid errors on reload
    if "shap" not in mpl.colormaps:
        mpl.colormaps.register(name="shap", cmap=CUSTOM_CMAP)
except (ImportError, AttributeError):
    # Fallback for older matplotlib versions or if something goes wrong
    pass

# Determine check_array finite parameter name (sklearn 1.6+ rename)
_CHECK_ARRAY_SIG = inspect.signature(check_array)
if "ensure_all_finite" in _CHECK_ARRAY_SIG.parameters:
    _FINITE_PARAM_NAME = "ensure_all_finite"
else:
    _FINITE_PARAM_NAME = "force_all_finite"

# Keys that should be forwarded to ``plt.subplots`` via ``setup_figure``.
# Keys that should be forwarded to ``plt.subplots`` via ``setup_figure``.
FIGURE_SETUP_KEYS: frozenset[str] = frozenset(
    {"subplot_kw", "gridspec_kw", "sharex", "sharey"}
)


def split_figure_plot_kwargs(
    kwargs: dict[str, object],
    figure_keys: Iterable[str] = FIGURE_SETUP_KEYS,
) -> tuple[dict[str, object], dict[str, object]]:
    """Split keyword arguments between figure creation and plotting.

    Parameters
    ----------
    kwargs : dict[str, Any]
        Keyword arguments passed to the plotting entrypoint.
    figure_keys : Iterable[str], optional
        Keys that should be forwarded to ``plt.subplots``.

    Returns
    -------
    tuple[dict[str, Any], dict[str, Any]]
        A tuple containing the figure kwargs and plot kwargs respectively.
    """

    figure_key_set = set(figure_keys)
    figure_kwargs = {k: v for k, v in kwargs.items() if k in figure_key_set}
    plot_kwargs = {k: v for k, v in kwargs.items() if k not in figure_key_set}
    return figure_kwargs, plot_kwargs


def ensure_axes(
    ax: Optional[Axes] = None,
    *,
    figsize: tuple[float, float] = (8.0, 6.0),
) -> tuple[Figure, Axes]:
    """Return a valid figure/axes pair, creating one when needed."""

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        return fig, ax

    figure = cast(Figure, ax.get_figure())
    if figure is None:
        raise ValueError("Axes object has no associated figure")
    return figure, ax


def apply_limits(
    ax: Axes,
    *,
    xlim: Optional[tuple[float, float]] = None,
    ylim: Optional[tuple[float, float]] = None,
) -> None:
    """Apply optional axis limits in a single, reusable helper."""

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)


def set_default_axis_labels(
    ax: Axes,
    *,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
) -> None:
    """Set axis labels only when the axes do not already define them."""

    if xlabel and not ax.get_xlabel():
        ax.set_xlabel(xlabel)
    if ylabel and not ax.get_ylabel():
        ax.set_ylabel(ylabel)


def setup_figure(
    figsize: Optional[tuple] = (10, 8),
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    **kwargs,
) -> tuple[Figure, Axes]:
    """Create a figure with consistent styling."""
    if figsize is None:
        figsize = (10, 8)
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
    cmap = plt.colormaps.get_cmap(colormap)
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
    - "shap" (custom Red-Blue) for continuous data

    Examples
    --------
    >>> get_default_colormap(is_categorical=True)
    'tab10'

    >>> get_default_colormap(is_categorical=False)
    'shap'

    >>> get_default_colormap(is_categorical=True, colormap='Set2')
    'Set2'
    """
    if colormap is not None:
        return colormap
    return "tab10" if is_categorical else "shap"


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


def calculate_ylim_for_xlim(
    x: np.ndarray,
    y: np.ndarray,
    xlim: tuple[float, float],
    margin: float = 0.05,
) -> tuple[float, float]:
    """Calculate appropriate y-axis limits for the given x-axis range.

    This utility function automatically scales the y-axis to fit the data
    within a specified x-axis range, useful for zooming into spectral regions
    or feature ranges while maintaining optimal y-axis scaling.

    Parameters
    ----------
    x : np.ndarray
        X-axis data (e.g., wavelengths, wavenumbers, feature indices).
        Shape: (n_points,)
    y : np.ndarray
        Y-axis data (e.g., spectra, loadings). Can be 1D or 2D.
        - If 1D with shape (n_points,): single spectrum/loading
        - If 2D: can have either layout:
          * (n_spectra, n_points): x maps to columns (axis 1)
          * (n_points, n_components): x maps to rows (axis 0)
        Function automatically detects which layout based on x length.
    xlim : tuple[float, float]
        The x-axis limits (xmin, xmax) to calculate y-limits for.
    margin : float, optional
        Fraction of the data range to add as margin (default: 0.05 = 5%).
        This prevents data from touching the plot edges.

    Returns
    -------
    tuple[float, float]
        The calculated y-axis limits (ymin, ymax) with margin applied.
        Returns (0, 1) if no data points are found within xlim.

    Examples
    --------
    Auto-scale y-axis when zooming into a spectral region:

    >>> xlim = (2800, 3000)  # Focus on C-H stretch region
    >>> ylim = calculate_ylim_for_xlim(wavenumbers, spectra, xlim)
    >>> ax.set_xlim(xlim)
    >>> ax.set_ylim(ylim)

    With custom margin:

    >>> ylim = calculate_ylim_for_xlim(x, y, xlim=(100, 200), margin=0.1)

    Works with 2D data (multiple spectra/loadings):

    >>> # y can have shape (n_spectra, n_points) or (n_points, n_components)
    >>> ylim = calculate_ylim_for_xlim(wavelengths, all_spectra, xlim)
    """
    xmin, xmax = xlim
    # Find indices within the x-range
    mask = (x >= xmin) & (x <= xmax)

    if not np.any(mask):
        # No data in range, return default limits
        return (0, 1)

    # Handle both 1D and 2D y data
    if y.ndim == 1:
        y_in_range = y[mask]
    else:
        # For 2D data, determine if x corresponds to rows or columns
        # If x length matches axis 1 (columns), filter columns (SpectraPlot style)
        # If x length matches axis 0 (rows), filter rows (LoadingsPlot style)
        if len(x) == y.shape[1]:
            # x maps to columns: y has shape (n_spectra, n_points)
            y_in_range = y[:, mask]
        elif len(x) == y.shape[0]:
            # x maps to rows: y has shape (n_points, n_components)
            y_in_range = y[mask, :]
        else:
            raise ValueError(
                f"x length ({len(x)}) must match either dimension of y {y.shape}"
            )

    # Calculate min and max
    ymin = np.min(y_in_range)
    ymax = np.max(y_in_range)

    # Add margin
    y_range = ymax - ymin
    if y_range > 0:
        ymin -= margin * y_range
        ymax += margin * y_range
    else:
        # If all values are the same, add small margin
        ymin -= 0.1
        ymax += 0.1

    return (ymin, ymax)


def add_confidence_lines(
    ax: Axes,
    x_threshold: Optional[float] = None,
    y_threshold: Optional[float] = None,
    color: str = "red",
    linestyle: str = "--",
    linewidth: float = 2,
    alpha: float = 0.7,
    label_x: Optional[str] = None,
    label_y: Optional[str] = None,
) -> None:
    """Add vertical and/or horizontal confidence/threshold lines to a plot.

    Useful for showing control limits, confidence regions, or threshold values
    in diagnostic plots (e.g., Q residuals, Hotelling's T²).

    Parameters
    ----------
    ax : Axes
        Matplotlib axes to draw the lines on.
    x_threshold : float, optional
        X-coordinate for vertical line. If None, no vertical line is drawn.
    y_threshold : float, optional
        Y-coordinate for horizontal line. If None, no horizontal line is drawn.
    color : str, optional
        Color of the threshold lines (default: "red").
    linestyle : str, optional
        Line style (default: "--" for dashed).
    linewidth : float, optional
        Width of the lines (default: 2).
    alpha : float, optional
        Transparency of the lines (default: 0.7).
    label_x : str, optional
        Label for the vertical line (appears in legend).
    label_y : str, optional
        Label for the horizontal line (appears in legend).

    Examples
    --------
    >>> fig, ax = plt.subplots()
    >>> ax.scatter(x, y)
    >>> add_confidence_lines(
    ...     ax,
    ...     x_threshold=12.5,
    ...     y_threshold=5.2,
    ...     label_x="T² limit (95%)",
    ...     label_y="Q limit (95%)"
    ... )
    """
    if x_threshold is not None:
        ax.axvline(
            x=x_threshold,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            alpha=alpha,
            label=label_x,
        )

    if y_threshold is not None:
        ax.axhline(
            y=y_threshold,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            alpha=alpha,
            label=label_y,
        )


def validate_data(
    X: Any,
    name: str = "Input",
    ensure_2d: bool = False,
    numeric: bool = True,
) -> np.ndarray:
    """Validate input data using sklearn check_array.

    Parameters
    ----------
    X : array-like
        Input data to validate.
    name : str, default="Input"
        Name of the input for error messages.
    ensure_2d : bool, default=False
        Whether to force 2D array.
    """
    dtype = "numeric" if numeric else None
    kwargs = {_FINITE_PARAM_NAME: numeric}

    # check_array parameters
    return check_array(
        X,
        dtype=dtype,
        ensure_2d=ensure_2d,
        input_name=name,
        **kwargs,
    )


def scatter_with_colormap(
    ax: Axes,
    x: np.ndarray,
    y: np.ndarray,
    color_by: Optional[np.ndarray] = None,
    is_categorical: bool = False,
    colormap: Optional[str] = None,
    color: Optional[str] = None,
    label: Optional[str] = None,
    **kwargs: Any,
) -> None:
    """Scatter plot with automatic colormap handling for categorical/continuous data.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes to plot on.
    x : np.ndarray
        X-coordinates.
    y : np.ndarray
        Y-coordinates.
    color_by : np.ndarray, optional
        Values for coloring samples.
    is_categorical : bool, optional
        Whether color_by is categorical.
    colormap : str, optional
        Colormap name.
    color : str, optional
        Fallback color if color_by is None.
    label : str, optional
        Legend label.
    **kwargs : Any
        Additional arguments passed to ax.scatter.
    """
    if color_by is None:
        # Simple scatter with single color
        ax.scatter(
            x,
            y,
            c=color,
            label=label,
            **kwargs,
        )
    elif is_categorical:
        # Categorical coloring
        assert colormap is not None
        colors = get_colors_from_labels(color_by, colormap)
        unique_values = np.unique(color_by)

        # Plot each category
        for value in unique_values:
            mask = color_by == value
            ax.scatter(
                x[mask],
                y[mask],
                color=colors[mask][0],
                label=f"{label} - {value}" if label else f"{value}",
                **kwargs,
            )
    else:
        # Continuous coloring
        import matplotlib as mpl
        import matplotlib.colors as mcolors

        norm = mcolors.Normalize(vmin=color_by.min(), vmax=color_by.max())
        colormap_name = colormap if colormap is not None else "viridis"
        cmap = mpl.colormaps.get_cmap(colormap_name)

        ax.scatter(
            x,
            y,
            c=color_by,
            cmap=cmap,
            norm=norm,
            label=label,
            **kwargs,
        )
