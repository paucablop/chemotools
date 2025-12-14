"""Distances plot for visualizing diagnostic measures and outlier detection."""

from typing import Optional, Any, Tuple, Literal

import numpy as np
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from chemotools.plotting._base import BasePlot, ColoringMixin
from chemotools.plotting._utils import (
    annotate_points,
    add_confidence_lines,
    validate_data,
    scatter_with_colormap,
)


class DistancesPlot(BasePlot, ColoringMixin):
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

    marker : str, optional
        Marker style for scatter points (default: "o"). Examples: "o", "s", "^", "v", "D".
    confidence_lines : bool or tuple[float | None, float | None], optional
        Whether to draw confidence/threshold lines.

        - If True: draws lines at distances using default method
        - If tuple: (x_threshold, y_threshold) values for lines
        - If False or None: no lines (default)

        Examples: True, (12.5, 5.2), (None, 5.2), (12.5, None)
    color_mode : {"continuous", "categorical"}, optional
        Explicitly specify coloring mode. If None (default), automatically
        detects based on dtype and unique values of color_by.
    colorbar_label : str, optional
        Label for the colorbar when using continuous coloring.
        Default is "Value". Only applies when color_by is continuous.

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
    ...     x=t2_residuals,
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
    ...     x=t2_residuals,
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
        marker: str = "o",
        confidence_lines: Optional[bool | tuple[float | None, float | None]] = None,
        color_mode: Optional[Literal["continuous", "categorical"]] = None,
        colorbar_label: str = "Value",
    ):
        self.annotations = annotations
        self.label = label
        self.color = color
        self.marker = marker

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

        # Validate inputs
        y = validate_data(y, name="y", ensure_2d=False)
        if x is not None:
            x = validate_data(x, name="x", ensure_2d=False)

        if color_by is not None:
            color_by = validate_data(
                color_by, name="color_by", ensure_2d=False, numeric=False
            )

        self._default_xlabel: str
        self._default_ylabel: str
        self._init_from_xy(x, y)

        # Initialize coloring
        self._init_coloring(
            color_by, colormap, color_mode=color_mode, colorbar_label=colorbar_label
        )

        self._validate_color_and_annotations()

    def _init_from_xy(
        self,
        x: Optional[np.ndarray],
        y: np.ndarray,
    ) -> None:
        """Initialize internal state from explicit x/y arrays."""

        if y.ndim != 1:
            raise ValueError("Explicit 'y' must be a 1D array.")

        if x is None:
            self._x = np.arange(y.shape[0])
            auto_xlabel = "Sample Index"
        else:
            if x.ndim != 1:
                raise ValueError("Explicit 'x' must be a 1D array.")
            if x.shape[0] != y.shape[0]:
                raise ValueError("'x' and 'y' must have the same length.")
            self._x = x
            auto_xlabel = "X"

        self._y = y

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

    def _get_default_labels(self) -> dict[str, str]:
        return {
            "xlabel": self._default_xlabel,
            "ylabel": self._default_ylabel,
        }

    def show(
        self,
        *,
        figsize: Optional[Tuple[float, float]] = None,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        xlim: Optional[Tuple[float, float]] = None,
        ylim: Optional[Tuple[float, float]] = None,
        **kwargs: Any,
    ) -> Figure:
        """Create and return a complete figure with the distances plot.

        This method handles figure creation and then delegates to `render()`.

        Parameters
        ----------
        figsize : tuple[float, float], optional
            Figure size in inches (width, height).
        title : str, optional
            Figure title.
        xlabel : str, optional
            Custom x-axis label. If None, uses existing label or default.
        ylabel : str, optional
            Custom y-axis label. If None, uses existing label or default.
        xlim : tuple[float, float], optional
            X-axis limits as (xmin, xmax).
        ylim : tuple[float, float], optional
            Y-axis limits as (ymin, ymax).
        **kwargs : Any
            Additional keyword arguments passed to the render() method.

        Returns
        -------
        Figure
            The matplotlib Figure object containing the plot.
        """
        return super().show(
            figsize=figsize,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            xlim=xlim,
            ylim=ylim,
            **kwargs,
        )

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
        fig, ax = super().render(
            ax=ax,
            xlabel=xlabel,
            ylabel=ylabel,
            xlim=xlim,
            ylim=ylim,
            **kwargs,
        )

        # Add colorbar for continuous data
        self._add_colorbar_if_needed(ax)

        # Add legend
        ax.legend()

        return fig, ax

    def _render_plot(self, ax: Axes, **kwargs: Any) -> None:
        """Internal method to render the distances plot on given axes."""
        alpha = kwargs.pop("alpha", 0.7)
        s = kwargs.pop("s", 50)
        marker = kwargs.pop("marker", self.marker)

        # Extract data for plotting
        x = self._x
        y = self._y

        scatter_with_colormap(
            ax,
            x,
            y,
            color_by=self.color_by,
            is_categorical=self.is_categorical,
            colormap=self.colormap,
            color=self.color,
            label=self.label,
            alpha=alpha,
            s=s,
            marker=marker,
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
