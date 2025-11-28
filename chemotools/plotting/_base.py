"""Base classes and mixins for chemotools plotting."""

from typing import Optional, Any, Tuple, Protocol, runtime_checkable, Literal
from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from chemotools.plotting._utils import (
    setup_figure,
    split_figure_plot_kwargs,
    ensure_axes,
    apply_limits,
    set_default_axis_labels,
    detect_categorical,
    get_default_colormap,
    add_colorbar,
)


@runtime_checkable
class Display(Protocol):
    """Protocol for objects that can be displayed as plots.

    This protocol defines a consistent interface for visualization across
    chemotools. Any class implementing these methods can be used polymorphically
    for plotting operations.

    The protocol supports flexible plotting with optional figure/axes injection,
    making it easy to create subplots and composite visualizations.

    Examples
    --------
    >>> class MyPlot:
    ...     def show(self, **kwargs):
    ...         fig, ax = plt.subplots()
    ...         ax.plot([1, 2, 3])
    ...         return fig
    ...
    ...     def render(self, ax=None, **kwargs):
    ...         if ax is None:
    ...             fig, ax = plt.subplots()
    ...         else:
    ...             fig = ax.figure
    ...         ax.plot([1, 2, 3])
    ...         return fig, ax
    ...
    >>> plot = MyPlot()
    >>> isinstance(plot, Display)  # True
    >>> fig = plot.show()
    """

    def show(self, **kwargs: Any) -> Figure:
        """Create and return a complete figure with the plot.

        This method creates a new figure and displays the plot on it.
        Use this when you want a standalone visualization.

        Parameters
        ----------
        **kwargs : Any
            Keyword arguments for customizing the plot. Common parameters include:

            - figsize : tuple[float, float], optional
                Figure size as (width, height) in inches.
            - title : str, optional
                Title for the plot.
            - xlabel : str, optional
                X-axis label.
            - ylabel : str, optional
                Y-axis label.
            - xlim : tuple[float, float], optional
                X-axis limits.
            - ylim : tuple[float, float], optional
                Y-axis limits.

        Returns
        -------
        Figure
            The matplotlib Figure object containing the plot.

        Examples
        --------
        >>> fig = plotter.show(figsize=(10, 6), title="My Plot")
        >>> fig.savefig("output.png")
        """
        ...

    def render(self, ax: Optional[Axes] = None, **kwargs: Any) -> Any:
        """Render the plot on the given axes or create new ones.

        This method is more flexible than `show()` as it allows plotting
        on existing axes, making it perfect for creating subplots and
        composite visualizations.

        Parameters
        ----------
        ax : Axes, optional
            Matplotlib axes to plot on. If None, creates new figure and axes.
        **kwargs : Any
            Additional keyword arguments for customizing the plot.

        Returns
        -------
        Any
            Typically returns tuple[Figure, Axes] or just Axes, depending on implementation.
            Implementations may vary in their return type based on specific needs.

        Examples
        --------
        Plot on existing axes:

        >>> fig, axes = plt.subplots(2, 2)
        >>> fig, ax = plotter.render(ax=axes[0, 0])

        Create new figure:

        >>> fig, ax = plotter.render()
        >>> ax.set_xlabel("Custom label")
        """
        ...


def is_displayable(obj: Any) -> bool:
    """Check if an object implements the Display protocol.

    Parameters
    ----------
    obj : Any
        Object to check.

    Returns
    -------
    bool
        True if the object implements Display protocol.

    Examples
    --------
    >>> is_displayable(my_plotter)
    True
    """
    return isinstance(obj, Display)


class BasePlot(Display, ABC):
    """Base class for all plots implementing the Display protocol.

    This class reduces boilerplate by implementing the standard show/render pattern.
    Subclasses should implement `_render_plot` and optionally override `render`
    if they need custom logic before/after the standard rendering pipeline.
    """

    def _get_default_labels(self) -> dict[str, str]:
        """Return default labels for the plot.

        Returns
        -------
        dict[str, str]
            Dictionary with keys 'xlabel', 'ylabel', and optionally 'title'.
        """
        return {}

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
        """Create and return a complete figure with the plot.

        This method handles figure creation and then delegates to `render()`.
        """
        # Get defaults
        defaults = self._get_default_labels()

        # Resolve labels: argument > default > None
        title = title if title is not None else defaults.get("title")
        xlabel = xlabel if xlabel is not None else defaults.get("xlabel")
        ylabel = ylabel if ylabel is not None else defaults.get("ylabel")

        # Split kwargs into figure setup (e.g. subplot_kw) and plotting kwargs
        figure_kwargs, plot_kwargs = split_figure_plot_kwargs(kwargs)

        # Create figure with consistent styling
        fig, ax = setup_figure(
            figsize=figsize,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            **figure_kwargs,
        )

        # Delegate to render for the actual plotting
        self.render(
            ax=ax,
            xlabel=xlabel,
            ylabel=ylabel,
            xlim=xlim,
            ylim=ylim,
            **plot_kwargs,
        )

        plt.tight_layout()
        return fig

    def render(
        self,
        ax: Optional[Axes] = None,
        *,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        xlim: Optional[Tuple[float, float]] = None,
        ylim: Optional[Tuple[float, float]] = None,
        **kwargs: Any,
    ) -> Tuple[Figure, Axes]:
        """Render the plot on the given axes or create new ones."""
        # Get defaults
        defaults = self._get_default_labels()

        # Resolve labels
        xlabel = xlabel if xlabel is not None else defaults.get("xlabel")
        ylabel = ylabel if ylabel is not None else defaults.get("ylabel")

        fig, ax = ensure_axes(ax)

        # Hook for actual plotting logic
        self._render_plot(ax, **kwargs)

        # Apply labels if provided (and not already set by setup_figure/ax)
        # We pass them here to ensure they are applied even if render is called directly
        if xlabel or ylabel:
            set_default_axis_labels(ax, xlabel=xlabel, ylabel=ylabel)

        # Apply limits
        apply_limits(ax, xlim=xlim, ylim=ylim)

        return fig, ax

    @abstractmethod
    def _render_plot(self, ax: Axes, **kwargs: Any) -> None:
        """Implement the actual plotting logic here.

        Parameters
        ----------
        ax : Axes
            The axes to plot on.
        **kwargs : Any
            Plotting keyword arguments.
        """
        pass


class ColoringMixin:
    """Mixin for handling consistent coloring logic (categorical vs continuous)."""

    color_by: Optional[np.ndarray]
    is_categorical: bool
    colormap: Optional[str]
    colorbar_label: str

    def _init_coloring(
        self,
        color_by: Optional[np.ndarray],
        colormap: Optional[str],
        color_mode: Optional[Literal["continuous", "categorical"]] = None,
        colorbar_label: str = "Value",
    ) -> None:
        """Initialize coloring attributes."""
        self.color_by = color_by
        self.colorbar_label = colorbar_label

        if color_mode == "categorical":
            self.is_categorical = True
        elif color_mode == "continuous":
            self.is_categorical = False
        elif color_by is not None:
            self.is_categorical = detect_categorical(color_by)
        else:
            self.is_categorical = False

        self.colormap = get_default_colormap(self.is_categorical, colormap)

    def _add_colorbar_if_needed(self, ax: Axes) -> None:
        """Add a colorbar if the data is continuous."""
        if self.color_by is not None and not self.is_categorical:
            if self.colormap is None:
                self.colormap = get_default_colormap(self.is_categorical, None)
            add_colorbar(ax, self.color_by, self.colormap, self.colorbar_label)
