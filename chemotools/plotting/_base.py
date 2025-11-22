"""Base classes and mixins for chemotools plotting."""

from typing import Optional, Any, Tuple
from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from chemotools.plotting._display import Display
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


class BasePlot(Display, ABC):
    """Base class for all plots implementing the Display protocol.

    This class reduces boilerplate by implementing the standard show/render pattern.
    Subclasses should implement `_render_plot` and optionally override `render`
    if they need custom logic before/after the standard rendering pipeline.
    """

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
        categorical: Optional[bool] = None,
        colorbar_label: str = "Value",
    ) -> None:
        """Initialize coloring attributes."""
        self.color_by = color_by
        self.colorbar_label = colorbar_label

        if categorical is not None:
            self.is_categorical = categorical
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
