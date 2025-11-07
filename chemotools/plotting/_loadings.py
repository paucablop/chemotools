"""Loadings plot for visualizing model feature weights."""

from typing import Optional, Any, cast
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from chemotools.plotting._utilities import setup_figure, calculate_ylim_for_xlim


class LoadingsPlot:
    """Loadings plot implementing Display protocol for model inspection.

    This class creates line plots of model loadings (feature weights),
    following the same design pattern as SpectrumPlot and ScoresPlot.
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
    xlabel : str, optional
        X-axis label. Default is "Feature".
    ylabel : str, optional
        Y-axis label. Default is "Loading".

    Raises
    ------
    ValueError
        If component index exceeds the available components in the loadings array.

    Examples
    --------
    Basic usage with single component:

    >>> loadings = model.components_.T  # Shape: (n_features, n_components)
    >>> wavelengths = np.linspace(400, 2500, n_features)
    >>> plot = LoadingsPlot(loadings, feature_names=wavelengths, components=0)
    >>> fig = plot.show(title="PC1 Loadings")

    Plot multiple components overlaid:

    >>> plot = LoadingsPlot(
    ...     loadings,
    ...     feature_names=wavelengths,
    ...     components=[0, 1, 2],  # Plot PC1, PC2, PC3 together
    ...     xlabel='Wavenumber (cm⁻¹)',
    ...     ylabel='Loading Coefficient'
    ... )
    >>> fig = plot.show(title="First 3 Principal Components")

    With custom axis labels:

    >>> plot = LoadingsPlot(
    ...     loadings,
    ...     feature_names=wavenumbers,
    ...     components=0,
    ...     xlabel="Wavenumber (cm⁻¹)",
    ...     ylabel="Loading Coefficient"
    ... )
    >>> fig = plot.show(title="PLS LV1 Loadings")

    Zoom into a spectral region:

    >>> plot = LoadingsPlot(loadings, feature_names=wavenumbers, components=[0, 1])
    >>> fig = plot.show(title="C-H Region", xlim=(2800, 3000))

    Create subplots for different component groups:

    >>> fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    >>> plot1 = LoadingsPlot(loadings, feature_names=wavelengths, components=[0, 1])
    >>> plot1.render(ax=axes[0])
    >>> plot2 = LoadingsPlot(loadings, feature_names=wavelengths, components=[2, 3])
    >>> plot2.render(ax=axes[1])
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
        feature_names: Optional[np.ndarray | list] = None,
        components: int | list[int] = 0,
        xlabel: str = "Feature",
        ylabel: str = "Loading",
    ):
        # Validate loadings shape
        if loadings.ndim != 2:
            raise ValueError(
                f"loadings must be 2D array with shape (n_features, n_components), "
                f"got shape {loadings.shape}"
            )

        self.loadings = loadings
        self.n_features, self.n_components = loadings.shape
        self.xlabel = xlabel
        self.ylabel = ylabel

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
            feature_names = np.asarray(feature_names)
            if len(feature_names) != self.n_features:
                raise ValueError(
                    f"feature_names length ({len(feature_names)}) must match "
                    f"number of features ({self.n_features})"
                )
            self.feature_names = feature_names
        else:
            self.feature_names = np.arange(self.n_features)

    def show(
        self,
        figsize: Optional[tuple[float, float]] = None,
        title: Optional[str] = None,
        xlim: Optional[tuple[float, float]] = None,
        ylim: Optional[tuple[float, float]] = None,
        **kwargs: Any,
    ) -> Figure:
        """Create and return a complete figure with the loadings plot.

        Parameters
        ----------
        figsize : tuple[float, float], optional
            Figure size as (width, height) in inches. Default is (12, 4).
        title : str, optional
            Title for the plot. If None, auto-generates title based on components.
        xlim : tuple[float, float], optional
            X-axis limits as (xmin, xmax). Useful for zooming into specific
            feature regions (e.g., spectral bands).
        ylim : tuple[float, float], optional
            Y-axis limits as (ymin, ymax). Useful for focusing on certain
            loading magnitudes.
        **kwargs : Any
            Additional keyword arguments split into:
            - Figure setup kwargs (subplot_kw, gridspec_kw) → setup_figure()
            - Plot kwargs (linewidth, linestyle, alpha, etc.) → ax.plot()

            Common plot kwargs:
            - linewidth or lw : float, optional (default: 1.5)
            - linestyle or ls : str, optional (default: '-')
            - alpha : float, optional (default: 0.8)
            - marker : str, optional (e.g., 'o', 's', '^')
            - markersize : float, optional

        Returns
        -------
        Figure
            The matplotlib Figure object containing the plot.

        Examples
        --------
        Basic usage:

        >>> plot.show()

        With custom title and figure size:

        >>> plot.show(figsize=(15, 5), title="Principal Component Loadings")

        Zoom into a spectral region:

        >>> plot.show(title="C-H Stretch Region", xlim=(2800, 3000))

        Custom styling:

        >>> plot.show(
        ...     title="Custom Styled Loadings",
        ...     linewidth=2.5,
        ...     alpha=0.7
        ... )
        """
        # Separate kwargs for setup_figure vs plot
        figure_kwargs_keys = {"subplot_kw", "gridspec_kw", "sharex", "sharey"}
        figure_kwargs = {k: v for k, v in kwargs.items() if k in figure_kwargs_keys}
        plot_kwargs = {k: v for k, v in kwargs.items() if k not in figure_kwargs_keys}

        # Auto-generate title if not provided
        if title is None:
            if len(self.components) == 1:
                title = f"PC{self.components[0] + 1} Loadings"
            else:
                comp_names = ", ".join([f"PC{c + 1}" for c in self.components])
                title = f"Loadings: {comp_names}"

        # Use setup_figure utility for consistent styling
        fig, ax = setup_figure(
            figsize=figsize or (12, 4),
            title=title,
            xlabel=self.xlabel,
            ylabel=self.ylabel,
            **figure_kwargs,
        )

        # Render the actual plot
        self._render_plot(ax, **plot_kwargs)

        # Add zero reference line
        ax.axhline(y=0, color="k", linestyle="-", linewidth=0.5, alpha=0.3)

        # Add legend if multiple components
        if len(self.components) > 1:
            ax.legend()

        # Apply axis limits with auto-scaling
        if xlim is not None:
            ax.set_xlim(xlim)

            # Auto-scale y-axis to data within xlim if ylim not provided
            if ylim is None:
                # Collect all y-data for components being plotted
                y_data = self.loadings[:, self.components]
                ylim = calculate_ylim_for_xlim(self.feature_names, y_data, xlim)

        if ylim is not None:
            ax.set_ylim(ylim)

        plt.tight_layout()
        return fig

    def render(
        self,
        ax: Optional[Axes] = None,
        xlim: Optional[tuple[float, float]] = None,
        ylim: Optional[tuple[float, float]] = None,
        **kwargs: Any,
    ) -> tuple[Figure, Axes]:
        """Render the plot on the given axes or create new ones.

        Parameters
        ----------
        ax : Axes, optional
            Matplotlib axes to plot on. If None, creates new figure and axes.
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
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 4))
        else:
            figure = ax.get_figure()
            if figure is None:
                raise ValueError("Axes object has no associated figure")
            fig = cast(Figure, figure)

        self._render_plot(ax, **kwargs)

        # Add zero reference line
        ax.axhline(y=0, color="k", linestyle="-", linewidth=0.5, alpha=0.3)

        # Add legend if multiple components
        if len(self.components) > 1:
            ax.legend()

        # Apply axis limits with auto-scaling
        if xlim is not None:
            ax.set_xlim(xlim)

            # Auto-scale y-axis to data within xlim if ylim not provided
            if ylim is None:
                # Collect all y-data for components being plotted
                y_data = self.loadings[:, self.components]
                ylim = calculate_ylim_for_xlim(self.feature_names, y_data, xlim)

        if ylim is not None:
            ax.set_ylim(ylim)

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
            label = f"PC{comp_idx + 1}"

            ax.plot(
                x,
                loadings,
                label=label,
                linewidth=linewidth,
                alpha=alpha,
                **kwargs,
            )
