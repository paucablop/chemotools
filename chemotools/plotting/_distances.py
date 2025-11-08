"""Distances plot for visualizing diagnostic measures and outlier detection."""

from typing import Optional, Any, cast
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from chemotools.plotting._utilities import (
    setup_figure,
    get_colors_from_labels,
    detect_categorical,
    get_default_colormap,
    add_colorbar,
    annotate_points,
    add_confidence_lines,
)


class DistancesPlot:
    """Simple, composable distances plot for a single dataset.

    This class creates scatter plots of distance measures (e.g., Q residuals, Hotelling's T²)
    for outlier detection. Supports plotting one distance vs another or distance vs sample index.
    Multiple datasets can be overlaid by using the render() method on shared axes.

    Parameters
    ----------
    distances : np.ndarray
        Distance array with shape (n_samples,) for single distance or
        (n_samples, n_distances) for multiple distances.
    distances_selection : tuple[int | None, int] or int, optional
        Which distances to plot on x and y axes.
        - For 1D arrays: single int (always 0, can be omitted)
        - For 2D+ arrays: tuple (x_index, y_index) to select which distances to plot
        - Use None as first element to plot against sample index: (None, y_index)
        Default is (0, 1) for 2D+ arrays, or (None, 0) for 1D arrays.
        Examples: (0, 1) plots distance 0 vs distance 1
                  (None, 0) plots sample index vs distance 0
                  (1, 2) plots distance 1 vs distance 2
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
    ...     train_distances,
    ...     distances_selection=(0, 1),
    ...     label="Train",
    ...     color="blue",
    ...     confidence_lines=(12.5, 5.2)
    ... ).render(ax)
    >>> DistancesPlot(
    ...     test_distances,
    ...     distances_selection=(0, 1),
    ...     label="Test",
    ...     color="red"
    ... ).render(ax)
    >>> ax.set_xlabel("Hotelling's T²")
    >>> ax.set_ylabel("Q Residuals")
    >>> ax.legend()
    >>> plt.show()

    **With categorical coloring:**

    >>> distances = np.column_stack([t2, q_residuals])
    >>> plot = DistancesPlot(
    ...     distances,
    ...     distances_selection=(0, 1),
    ...     color_by=classes,
    ...     confidence_lines=(12.5, 5.2)
    ... )
    >>> fig = plot.show(title="Outliers by Class")

    **With annotations for outliers:**

    >>> outliers = [5, 23, 47]
    >>> annotations = [f"S{i}" if i in outliers else "" for i in range(len(q_residuals))]
    >>> plot = DistancesPlot(
    ...     q_residuals,
    ...     annotations=annotations,
    ...     confidence_lines=(None, 5.2)
    ... )
    >>> fig = plot.show(title="Annotated Outliers")
    """

    def __init__(
        self,
        distances: np.ndarray,
        *,
        distances_selection: tuple[int | None, int] | int | None = None,
        color_by: Optional[np.ndarray] = None,
        annotations: Optional[list[str]] = None,
        label: str = "Data",
        color: Optional[str] = None,
        colormap: Optional[str] = None,
        confidence_lines: Optional[bool | tuple[float | None, float | None]] = None,
    ):
        self.distances = distances
        self.color_by = color_by
        self.annotations = annotations
        self.label = label
        self.color = color

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
        self._validate_distances()

        # Determine dimensionality
        n_distances = self.distances.shape[1] if self.distances.ndim == 2 else 1

        # Parse distances_selection parameter
        self.distances_selection: tuple[int | None, int]
        if distances_selection is None:
            # Smart defaults
            if self.distances.ndim == 1:
                # 1D: plot distance vs sample index
                self.distances_selection = (None, 0)
            else:
                # 2D+: plot first vs second distance
                self.distances_selection = (0, 1)
        elif isinstance(distances_selection, int):
            # Single int: treat as y-axis, x is sample index
            self.distances_selection = (None, distances_selection)
        else:
            # Tuple provided
            self.distances_selection = distances_selection

        # Extract x_axis and y_axis
        self.x_axis, self.y_axis = self.distances_selection

        # Validate axes indices
        self._validate_axes(n_distances)

        # Determine if coloring is categorical or continuous
        if color_by is not None:
            self.is_categorical = detect_categorical(color_by)
            self.colormap: Optional[str] = get_default_colormap(
                self.is_categorical, colormap
            )
        else:
            self.is_categorical = False
            self.colormap = colormap

    def _validate_distances(self) -> None:
        """Validate that distances array has correct shape."""
        if self.distances.ndim not in (1, 2):
            raise ValueError(
                f"Distances must be 1D or 2D array, got shape {self.distances.shape}"
            )

    def _validate_axes(self, n_distances: int) -> None:
        """Validate that axis indices are valid for the number of distances.

        Parameters
        ----------
        n_distances : int
            Number of distance measures available.

        Raises
        ------
        ValueError
            If axis indices are invalid for the number of distances.
        """
        # Validate y_axis
        if self.y_axis < 0 or self.y_axis >= n_distances:
            raise ValueError(
                f"y_axis index {self.y_axis} is invalid. "
                f"Valid range: 0-{n_distances - 1} (have {n_distances} distance(s))"
            )

        # Validate x_axis (can be None for sample index)
        if self.x_axis is not None:
            if self.x_axis < 0 or self.x_axis >= n_distances:
                raise ValueError(
                    f"x_axis index {self.x_axis} is invalid. "
                    f"Valid range: 0-{n_distances - 1} (have {n_distances} distance(s))"
                )

            # Prevent plotting same distance on both axes
            if self.x_axis == self.y_axis:
                raise ValueError(
                    f"x_axis and y_axis cannot be the same (both are {self.x_axis}). "
                    f"Please select different distances to plot."
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
            Custom x-axis label. If None, auto-generates based on x_axis.
        ylabel : str, optional
            Custom y-axis label. If None, auto-generates based on y_axis.
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
        if xlabel is None:
            if self.x_axis is None:
                xlabel_text = "Sample Index"
            else:
                xlabel_text = f"Distance {self.x_axis + 1}"
        else:
            xlabel_text = xlabel

        if ylabel is None:
            ylabel_text = f"Distance {self.y_axis + 1}"
        else:
            ylabel_text = ylabel

        # Create figure
        fig, ax = setup_figure(
            figsize=figsize or (8, 8),
            title=title,
            xlabel=xlabel_text,
            ylabel=ylabel_text,
        )

        self._render_plot(ax, **kwargs)

        # Apply axis limits
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)

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
            Custom x-axis label. If None, uses existing label or auto-generates.
        ylabel : str, optional
            Custom y-axis label. If None, uses existing label or auto-generates.
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
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
        else:
            figure = ax.get_figure()
            if figure is None:
                raise ValueError("Axes object has no associated figure")
            fig = cast(Figure, figure)

        self._render_plot(ax, **kwargs)

        # Set axis labels if provided
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        elif not ax.get_xlabel():  # Only set default if no label exists
            if self.x_axis is None:
                ax.set_xlabel("Sample Index")
            else:
                ax.set_xlabel(f"Distance {self.x_axis + 1}")

        if ylabel is not None:
            ax.set_ylabel(ylabel)
        elif not ax.get_ylabel():  # Only set default if no label exists
            ax.set_ylabel(f"Distance {self.y_axis + 1}")

        # Apply axis limits
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)

        return fig, ax

    def _render_plot(self, ax: Axes, **kwargs: Any) -> None:
        """Internal method to render the distances plot on given axes."""
        alpha = kwargs.pop("alpha", 0.7)
        s = kwargs.pop("s", 50)

        # Extract data for plotting
        x: np.ndarray
        y: np.ndarray

        if self.x_axis is None:
            # Plot vs sample index
            x = np.arange(len(self.distances))
        else:
            # Plot distance vs distance
            if self.distances.ndim == 1:
                # Should not happen due to validation, but handle gracefully
                x = np.arange(len(self.distances))
            else:
                x = self.distances[:, self.x_axis]

        # Extract y data
        if self.distances.ndim == 1:
            y = self.distances
        else:
            y = self.distances[:, self.y_axis]

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
            from matplotlib import cm
            import matplotlib.colors as mcolors

            norm = mcolors.Normalize(vmin=self.color_by.min(), vmax=self.color_by.max())
            cmap = cm.get_cmap(self.colormap)

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
                linewidth=2,
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
