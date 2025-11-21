"""Distances plot for visualizing diagnostic measures and outlier detection."""

from typing import Optional, Any
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from chemotools.plotting import Display
from chemotools.plotting._utils import (
    setup_figure,
    get_colors_from_labels,
    detect_categorical,
    get_default_colormap,
    add_colorbar,
    annotate_points,
    add_confidence_lines,
    ensure_axes,
    apply_limits,
    set_default_axis_labels,
)


class DistancesPlot(Display):
    """Simple, composable distances plot for a single dataset.

    This class creates scatter plots of distance measures (e.g., Q residuals, Hotelling's T²)
    for outlier detection. Supports plotting one distance vs another or distance vs sample index.
    Multiple datasets can be overlaid by using the render() method on shared axes.

    Parameters
    ----------
    x : np.ndarray, optional
        Explicit x-axis values. Must match the length of ``y``. When omitted,
        the sample index (0, 1, ..., n_samples-1) is used.
    y : np.ndarray, optional
        Y-axis values to plot. Accepts 1D arrays only.
    color_by : np.ndarray, optional
        Values for coloring samples. Can be either:
        - Continuous (numeric): shows colorbar
        - Categorical (strings/classes): shows legend with discrete colors
    annotations : list[str], optional
        Labels for annotating individual points.
    label : str, optional
        Legend label for this dataset (default: "Data").
    color : str, optional
        Color for all points when color_by is None (default: auto-assigned).
    colormap : str, optional
        Colormap name. Colorblind-friendly defaults:
        - "tab10" for categorical data
        - "viridis" for continuous data
    confidence_lines : bool or tuple[float | None, float | None], optional
        Whether to draw confidence/threshold lines.
        - If True: draws lines at distances using default method
        - If tuple: (x_threshold, y_threshold) values for lines
        - If False or None: no lines (default)
        Examples: True, (12.5, 5.2), (None, 5.2), (12.5, None)

    Raises
    ------
    ValueError
        If distances have invalid shapes or index selections.

    Examples
    --------
    **Simple single dataset plot (Q residuals vs sample index):**

    >>> plot = DistancesPlot(q_residuals, confidence_lines=(None, 5.2))
    >>> fig = plot.show(title="Q Residuals with Control Limit")

    **Multiple datasets composed together (T² vs Q):**

    >>> fig, ax = plt.subplots()
    >>> DistancesPlot(
    ...     y=train_q,
    ...     x=train_t2,
    ...     label="Train",
    ...     color="blue",
    ...     confidence_lines=(12.5, 5.2),
    ... ).render(ax)
    >>> DistancesPlot(
    ...     y=test_q,
    ...     x=test_t2,
    ...     label="Test",
    ...     color="red",
    ... ).render(ax)
    >>> ax.set_xlabel("Hotelling's T²")
    >>> ax.set_ylabel("Q Residuals")
    >>> ax.legend()
    >>> plt.show()

    **With categorical coloring:**

    >>> plot = DistancesPlot(
    ...     y=q_residuals,
    ...     x=t2_values,
    ...     color_by=classes,
    ...     confidence_lines=(12.5, 5.2),
    ... )
    >>> fig = plot.show(title="Outliers by Class")

    **With annotations for outliers:**

    >>> outliers = [5, 23, 47]
    >>> annotations = [f"S{i}" if i in outliers else "" for i in range(len(q_residuals))]
    >>> plot = DistancesPlot(
    ...     y=q_residuals,
    ...     annotations=annotations,
    ...     confidence_lines=(None, 5.2),
    ... )
    >>> fig = plot.show(title="Annotated Outliers")

    **Explicit x/y arrays:**

    >>> plot = DistancesPlot(
    ...     y=q_residuals,
    ...     x=t2_values,
    ...     confidence_lines=(9.35, 12.0),
    ... )
    >>> fig = plot.show(
    ...     title="T² vs Q",
    ...     xlabel="Hotelling's T²",
    ...     ylabel="Q Residuals",
    ... )
    """

    def __init__(
        self,
        y: np.ndarray,
        *,
        x: Optional[np.ndarray] = None,
        color_by: Optional[np.ndarray] = None,
        annotations: Optional[list[str]] = None,
        label: str = "Data",
        color: Optional[str] = None,
        colormap: Optional[str] = None,
        confidence_lines: Optional[bool | tuple[float | None, float | None]] = None,
    ):
        self._x: np.ndarray
        self._y: np.ndarray
        self.color_by = color_by
        self.annotations = annotations
        self.label = label
        self.color = color
        self.colormap: Optional[str]

        # Process confidence lines parameter
        if confidence_lines is True:
            # True means calculate from data - we'll implement later if needed
            self.x_threshold = None
            self.y_threshold = None
        elif isinstance(confidence_lines, tuple):
            self.x_threshold, self.y_threshold = confidence_lines
        else:
            self.x_threshold = None
            self.y_threshold = None

        self._default_xlabel: str
        self._default_ylabel: str
        self._init_from_xy(x, y)

        # Determine if coloring is categorical or continuous
        if self.color_by is not None:
            self.color_by = np.asarray(self.color_by)
            self.is_categorical = detect_categorical(self.color_by)
            self.colormap = get_default_colormap(self.is_categorical, colormap)
        else:
            self.is_categorical = False
            self.colormap = colormap

        self._validate_color_and_annotations()

    def _init_from_xy(
        self,
        x: Optional[np.ndarray],
        y: np.ndarray,
    ) -> None:
        """Initialize internal state from explicit x/y arrays."""

        y_arr = np.asarray(y)
        if y_arr.ndim != 1:
            raise ValueError("Explicit 'y' must be a 1D array.")

        if x is None:
            self._x = np.arange(y_arr.shape[0])
            auto_xlabel = "Sample Index"
        else:
            x_arr = np.asarray(x)
            if x_arr.ndim != 1:
                raise ValueError("Explicit 'x' must be a 1D array.")
            if x_arr.shape[0] != y_arr.shape[0]:
                raise ValueError("'x' and 'y' must have the same length.")
            self._x = x_arr
            auto_xlabel = "X"

        self._y = y_arr

        auto_ylabel = "Distance"

        self._default_xlabel = auto_xlabel
        self._default_ylabel = auto_ylabel

    def _validate_color_and_annotations(self) -> None:
        """Ensure optional color and annotation arrays align with the data length."""

        n_points = self._y.shape[0]

        if self.color_by is not None and len(self.color_by) != n_points:
            raise ValueError("color_by must have the same length as the plotted data.")

        if self.annotations is not None and len(self.annotations) != n_points:
            raise ValueError(
                "annotations must have the same length as the plotted data."
            )

    def show(
        self,
        *,
        figsize: Optional[tuple[float, float]] = None,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        xlim: Optional[tuple[float, float]] = None,
        ylim: Optional[tuple[float, float]] = None,
        **kwargs: Any,
    ) -> Figure:
        """Create and return a complete figure with the distances plot.

        Parameters
        ----------
        figsize : tuple[float, float], optional
            Figure size as (width, height) in inches. Default is (8, 8).
        title : str, optional
            Title for the plot.
        xlabel : str, optional
            Custom x-axis label. If None, uses the auto-detected default label
            (e.g., ``"Sample Index"`` or ``"X"``).
        ylabel : str, optional
            Custom y-axis label. If None, uses the auto-detected default label
            (e.g., ``"Distance"``).
        xlim : tuple[float, float], optional
            X-axis limits as (xmin, xmax).
        ylim : tuple[float, float], optional
            Y-axis limits as (ymin, ymax).
        **kwargs : Any
            Additional keyword arguments passed to ax.scatter().

        Returns
        -------
        Figure
            The matplotlib Figure object containing the plot.

        Examples
        --------
        >>> plot.show(title="Diagnostic Measures")
        >>> plot.show(figsize=(10, 10), xlim=(0, 20), ylim=(0, 10))
        >>> plot.show(title="Outliers", xlabel="Hotelling T²", ylabel="Q Residuals")
        """
        # Determine axis labels
        xlabel_text = xlabel if xlabel is not None else self._default_xlabel
        ylabel_text = ylabel if ylabel is not None else self._default_ylabel

        # Create figure
        fig, ax = setup_figure(
            figsize=figsize or (8, 8),
            title=title,
            xlabel=xlabel_text,
            ylabel=ylabel_text,
        )

        self._render_plot(ax, **kwargs)

        # Apply axis limits
        apply_limits(ax, xlim=xlim, ylim=ylim)

        # Add colorbar for continuous data
        if self.color_by is not None and not self.is_categorical:
            assert self.colormap is not None  # colormap is set by get_default_colormap
            add_colorbar(ax, self.color_by, self.colormap, "Value")

        # Add legend
        ax.legend()

        plt.tight_layout()
        return fig

    def render(
        self,
        ax: Optional[Axes] = None,
        *,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        xlim: Optional[tuple[float, float]] = None,
        ylim: Optional[tuple[float, float]] = None,
        **kwargs: Any,
    ) -> tuple[Figure, Axes]:
        """Render the plot on the given axes or create new ones.

        Use this method to compose multiple plots on the same axes.

        Parameters
        ----------
        ax : Axes, optional
            Matplotlib axes to plot on. If None, creates new figure and axes.
        xlabel : str, optional
            Custom x-axis label. If None, uses existing label or the default label
            configured at initialization.
        ylabel : str, optional
            Custom y-axis label. If None, uses existing label or the default label
            configured at initialization.
        xlim : tuple[float, float], optional
            X-axis limits as (xmin, xmax).
        ylim : tuple[float, float], optional
            Y-axis limits as (ymin, ymax).
        **kwargs : Any
            Additional keyword arguments passed to ax.scatter().

        Returns
        -------
        fig : Figure
            The matplotlib Figure object.
        ax : Axes
            The matplotlib Axes object with the rendered plot.

        Examples
        --------
        Compose multiple datasets:

        >>> fig, ax = plt.subplots()
        >>> DistancesPlot(train_dist, label="Train").render(ax)
        >>> DistancesPlot(test_dist, label="Test").render(ax)
        >>> ax.set_xlabel("Hotelling T²")
        >>> ax.set_ylabel("Q Residuals")
        >>> ax.legend()
        >>> plt.show()
        """
        fig, ax = ensure_axes(ax, figsize=(8, 8))

        self._render_plot(ax, **kwargs)

        # Set axis labels if provided
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        else:
            set_default_axis_labels(ax, xlabel=self._default_xlabel)

        if ylabel is not None:
            ax.set_ylabel(ylabel)
        else:
            set_default_axis_labels(ax, ylabel=self._default_ylabel)

        # Apply axis limits
        apply_limits(ax, xlim=xlim, ylim=ylim)

        return fig, ax

    def _render_plot(self, ax: Axes, **kwargs: Any) -> None:
        """Internal method to render the distances plot on given axes."""
        alpha = kwargs.pop("alpha", 0.7)
        s = kwargs.pop("s", 50)

        # Extract data for plotting
        x = self._x
        y = self._y

        if self.color_by is None:
            # Simple scatter with single color
            ax.scatter(
                x,
                y,
                c=self.color,
                label=self.label,
                alpha=alpha,
                s=s,
                **kwargs,
            )
        elif self.is_categorical:
            # Categorical coloring
            assert self.colormap is not None  # colormap is set by get_default_colormap
            colors = get_colors_from_labels(self.color_by, self.colormap)
            unique_values = np.unique(self.color_by)

            # Plot each category
            for value in unique_values:
                mask = self.color_by == value
                ax.scatter(
                    x[mask],
                    y[mask],
                    color=colors[mask][0],  # All same color for this category
                    label=f"{self.label} - {value}",
                    alpha=alpha,
                    s=s,
                    **kwargs,
                )
        else:
            # Continuous coloring
            import matplotlib as mpl
            import matplotlib.colors as mcolors

            norm = mcolors.Normalize(vmin=self.color_by.min(), vmax=self.color_by.max())
            # Ensure we have a valid colormap (should not be None here, but be defensive)
            colormap_name = self.colormap if self.colormap is not None else "viridis"
            cmap = mpl.colormaps.get_cmap(colormap_name)

            ax.scatter(
                x,
                y,
                c=self.color_by,
                cmap=cmap,
                norm=norm,
                label=self.label,
                alpha=alpha,
                s=s,
                **kwargs,
            )

        # Add confidence lines if requested
        if self.x_threshold is not None or self.y_threshold is not None:
            add_confidence_lines(
                ax,
                x_threshold=self.x_threshold,
                y_threshold=self.y_threshold,
                color="red",
                linestyle="--",
                linewidth=1,
                alpha=0.7,
            )

        # Add point annotations if provided
        if self.annotations is not None:
            annotate_points(
                ax,
                x,
                y,
                self.annotations,
                fontsize=8,
                xytext=(5, 5),
                textcoords="offset points",
            )
