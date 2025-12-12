"""Loadings plot for visualizing model feature weights."""

from typing import Optional, Any
import numpy as np
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from chemotools.plotting._base import BasePlot
from chemotools.plotting._utils import (
    calculate_ylim_for_xlim,
    validate_data,
)


class LoadingsPlot(BasePlot):
    """Loadings plot implementing Display protocol for model inspection.

    This class creates line plots of model loadings (feature weights),
    following the same design pattern as SpectraPlot and ScoresPlot.
    Supports plotting single or multiple components overlaid on the same plot.

    Parameters
    ----------
    loadings : np.ndarray
        Loadings array with shape (n_features, n_components).
    feature_names : np.ndarray or list, optional
        Names/values for features (x-axis). Can be wavelengths, wavenumbers,
        feature indices, etc. If None, uses feature indices [0, 1, 2, ...].
    components : int or list[int], optional
        Which component(s) to plot. Can be:
        - Single int (default 0): plots one component
        - List of ints: plots multiple components overlaid with legend
        Uses 0-based indexing.
    component_label : str, optional
        Prefix for component naming in legend and titles (default "PC").
        Use "LV" for PLS models, "PC" for PCA models, "IC" for ICA, etc.

    Raises
    ------
    ValueError
        If component index exceeds the available components in the loadings array.

    Examples
    --------
    Basic usage with single component:

    >>> loadings = model.components_.T  # Shape: (n_features, n_components)
    >>> wavenumbers = np.linspace(400, 2500, n_features)
    >>> plot = LoadingsPlot(loadings, feature_names=wavenumbers, components=0)
    >>> fig = plot.show(title="PC1 Loadings")

    Plot multiple components overlaid:

    >>> plot = LoadingsPlot(
    ...     loadings,
    ...     feature_names=wavenumbers,
    ...     components=[0, 1, 2],  # Plot PC1, PC2, PC3 together
    ... )
    >>> fig = plot.show(
    ...     title="First 3 Principal Components",
    ...     xlabel='Wavenumber (cm⁻¹)',
    ...     ylabel='Loading Coefficient'
    ... )

    With custom axis labels:

    >>> plot = LoadingsPlot(
    ...     loadings,
    ...     feature_names=wavenumbers,
    ...     components=0
    ... )
    >>> fig = plot.show(
    ...     title="PLS LV1 Loadings",
    ...     xlabel="Wavenumber (cm⁻¹)",
    ...     ylabel="Loading Coefficient"
    ... )

    Zoom into a spectral region:

    >>> plot = LoadingsPlot(loadings, feature_names=wavenumbers, components=[0, 1])
    >>> fig = plot.show(title="C-H Region", xlim=(2800, 3000))

    Create subplots for different component groups:

    >>> fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    >>> plot1 = LoadingsPlot(loadings, feature_names=wavelengths, components=[0, 1])
    >>> plot1.render(ax=axes[0])
    >>> plot2 = LoadingsPlot(loadings, feature_names=wavelengths, components=[0, 1])
    >>> plot2.render(ax=axes[1], xlim=(2800, 3000))
    >>> plt.tight_layout()

    Custom styling:

    >>> plot.show(
    ...     title="Styled Loadings",
    ...     linewidth=2,
    ...     alpha=0.8
    ... )
    """

    def __init__(
        self,
        loadings: np.ndarray,
        *,
        feature_names: Optional[np.ndarray | list] = None,
        components: int | list[int] = 0,
        component_label: str = "PC",
    ):
        self.loadings = validate_data(loadings, name="loadings", ensure_2d=True)
        self.n_features, self.n_components = self.loadings.shape
        self.component_label = component_label

        # Handle components parameter - convert to list
        if isinstance(components, int):
            self.components = [components]
        else:
            self.components = list(components)

        # Validate all component indices
        for comp in self.components:
            if comp < 0 or comp >= self.n_components:
                raise ValueError(
                    f"Component index {comp} is out of bounds. "
                    f"loadings has {self.n_components} components (valid range: 0-{self.n_components - 1})"
                )

        # Set up feature names/values for x-axis
        if feature_names is not None:
            self.feature_names = validate_data(
                feature_names, name="feature_names", ensure_2d=False, numeric=False
            )
            if len(self.feature_names) != self.n_features:
                raise ValueError(
                    f"feature_names length ({len(self.feature_names)}) must match "
                    f"number of features ({self.n_features})"
                )
        else:
            self.feature_names = np.arange(self.n_features)

    def _get_default_labels(self) -> dict[str, str]:
        # Auto-generate title if not provided
        if len(self.components) == 1:
            title = f"{self.component_label}{self.components[0] + 1} Loadings"
        else:
            comp_names = ", ".join(
                [f"{self.component_label}{c + 1}" for c in self.components]
            )
            title = f"Loadings: {comp_names}"

        return {
            "xlabel": "Feature",
            "ylabel": "Loading",
            "title": title,
        }

    def show(
        self,
        figsize=(12, 4),
        title=None,
        xlabel="X-axis",
        ylabel="Y-axis",
        xlim=None,
        ylim=None,
        **kwargs,
    ) -> Figure:
        """Show the loadings plot with optional customization.

        Parameters
        ----------
        figsize : tuple, optional
            Figure size as (width, height) in inches. Default is (12, 4).
        title : str, optional
            Title for the plot. If None, a default title is generated.
        xlabel : str, optional
            X-axis label. Default is "X-axis".
        ylabel : str, optional
            Y-axis label. Default is "Y-axis".
        xlim : tuple, optional
            X-axis limits as (xmin, xmax). Default is None (auto).
        ylim : tuple, optional
            Y-axis limits as (ymin, ymax). Default is None (auto).
        **kwargs : Any
            Additional keyword arguments passed to the plot function.

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

        Parameters
        ----------
        ax : Axes, optional
            Matplotlib axes to plot on. If None, creates new figure and axes.
        xlabel : str, optional
            X-axis label. Default is "Feature".
        ylabel : str, optional
            Y-axis label. Default is "Loading".
        xlim : tuple[float, float], optional
            X-axis limits as (xmin, xmax).
        ylim : tuple[float, float], optional
            Y-axis limits as (ymin, ymax).
        **kwargs : Any
            Additional keyword arguments passed to the plot function.

        Returns
        -------
        fig : Figure
            The matplotlib Figure object.
        ax : Axes
            The matplotlib Axes object with the rendered plot.

        Examples
        --------
        Render on existing axes:

        >>> fig, ax = plt.subplots()
        >>> plot.render(ax=ax)

        Create subplots:

        >>> fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        >>> plot1.render(ax=axes[0])
        >>> plot2.render(ax=axes[1])
        """
        # Apply axis limits with auto-scaling
        if xlim is not None and ylim is None:
            # Collect all y-data for components being plotted
            y_data = self.loadings[:, self.components]
            ylim = calculate_ylim_for_xlim(self.feature_names, y_data, xlim)

        fig, ax = super().render(
            ax=ax,
            xlabel=xlabel,
            ylabel=ylabel,
            xlim=xlim,
            ylim=ylim,
            **kwargs,
        )

        # Add zero reference line
        ax.axhline(y=0, color="k", linestyle="-", linewidth=0.5, alpha=0.3)

        # Add legend if multiple components
        if len(self.components) > 1:
            ax.legend()

        return fig, ax

    def _render_plot(self, ax: Axes, **kwargs: Any) -> None:
        """Internal method to render the loadings plot.

        Parameters
        ----------
        ax : Axes
            Matplotlib axes to plot on.
        **kwargs : Any
            Additional keyword arguments passed to the plot function.
        """
        # Set default styling
        linewidth = kwargs.pop("linewidth", kwargs.pop("lw", 1.5))
        alpha = kwargs.pop("alpha", 0.8)

        x = self.feature_names

        # Plot each component
        for comp_idx in self.components:
            loadings = self.loadings[:, comp_idx]
            label = f"{self.component_label}{comp_idx + 1}"

            ax.plot(
                x,
                loadings,
                label=label,
                linewidth=linewidth,
                alpha=alpha,
                **kwargs,
            )
