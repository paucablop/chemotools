"""
The :mod:`chemotools.plotting._spectrum` module implements the SpectrumPlot class for visualizing spectral data.
"""

# Authors: Pau Cabaneros
# License: MIT

from typing import Optional, Any, cast
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from chemotools.plotting._utilities import setup_figure, get_colors_from_labels


class SpectrumPlot:
    """Plot class for visualizing spectral data.

    This class implements the Display protocol and provides flexible options
    for plotting spectral data with categorical or continuous coloring.

    Parameters
    ----------
    x : np.ndarray
        X-axis data (e.g., wavelengths, wavenumbers).
    y : np.ndarray
        Y-axis data (e.g., spectra intensities). Can be 1D or 2D.
    labels : list[str], optional
        Labels for each spectrum (used for legend).
    xlabel : str, optional
        X-axis label. Default is "Wavelength (nm)".
    ylabel : str, optional
        Y-axis label. Default is "Absorbance".
    color_by : np.ndarray, optional
        Reference vector for coloring spectra. Can be:
        - Categorical (class labels): uses discrete colormap
        - Continuous (numeric values): uses continuous colormap
    colormap : str, optional
        Colormap name. Colorblind-friendly defaults:
        - "tab10" for categorical data (default)
        - "viridis" for continuous data
        Other options: "plasma", "cividis", "coolwarm"
    categorical : bool, optional
        Explicitly specify whether color_by should be treated as categorical.
        If None (default), automatically detects based on dtype and unique values.
        Use this to override automatic detection for edge cases.
    colorbar_label : str, optional
        Label for the colorbar when using continuous coloring.
        Default is "Reference Value". Only applies when color_by is continuous.

    Examples
    --------
    Basic usage:

    >>> x = np.linspace(400, 2500, 100)
    >>> y = np.random.randn(5, 100)
    >>> plotter = SpectrumPlot(x, y)
    >>> fig = plotter.show(title="NIR Spectra")

    With categorical coloring:

    >>> classes = np.array(['A', 'A', 'B', 'B', 'C'])
    >>> plotter = SpectrumPlot(x, y, color_by=classes)
    >>> fig = plotter.show(title="Spectra by Class")

    With continuous coloring:

    >>> concentrations = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    >>> plotter = SpectrumPlot(x, y, color_by=concentrations, colormap="viridis")
    >>> fig = plotter.show(title="Spectra by Concentration")

    With custom colorbar label:

    >>> plotter = SpectrumPlot(
    ...     x, y, color_by=concentrations,
    ...     colormap="viridis", colorbar_label="Concentration (mg/L)"
    ... )
    >>> fig = plotter.show(title="Spectra by Concentration")

    Override categorical detection for small numeric datasets:

    >>> levels = np.array([1, 2, 3, 4])  # 4 unique values - might be detected as categorical
    >>> plotter = SpectrumPlot(x, y, color_by=levels, categorical=False)
    >>> fig = plotter.show(title="4 Concentration Levels")

    With custom axis labels:

    >>> plotter = SpectrumPlot(x, y, xlabel="Wavenumber (cm⁻¹)", ylabel="Intensity")
    >>> fig = plotter.show(title="Raman Spectra")

    Creating subplots:

    >>> fig, axes = plt.subplots(2, 1)
    >>> plotter1.render(ax=axes[0])
    >>> plotter2.render(ax=axes[1])
    >>> plt.tight_layout()
    """

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        labels: Optional[list[str]] = None,
        xlabel: str = "Wavelength (nm)",
        ylabel: str = "Absorbance",
        color_by: Optional[np.ndarray] = None,
        colormap: Optional[str] = None,
        categorical: Optional[bool] = None,
        colorbar_label: str = "Reference Value",
    ):
        self.x = x
        self.y = y if y.ndim == 2 else y.reshape(1, -1)
        self.labels = labels or [f"Spectrum {i}" for i in range(len(self.y))]
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.color_by = color_by
        self.colorbar_label = colorbar_label

        # Determine if color_by is categorical or continuous
        if categorical is not None:
            # User explicitly specified the type
            self.is_categorical = categorical
        elif color_by is not None:
            # Automatic detection with improved logic
            self.is_categorical = self._detect_categorical(color_by)
        else:
            self.is_categorical = False

        # Set colormap with colorblind-friendly defaults
        if colormap is None:
            self.colormap = "tab10" if self.is_categorical else "viridis"
        else:
            self.colormap = colormap

    def _detect_categorical(self, color_by: np.ndarray) -> bool:
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
                # Check if values repeat (each value appears more than once)
                # This distinguishes [1.0, 2.0, 3.0, 4.0] from [1.0, 1.0, 2.0, 2.0]
                counts = np.bincount(np.searchsorted(unique_values, color_by))
                has_repeats = bool(np.any(counts > 1))
                return has_repeats

        return False

    def show(
        self,
        figsize: Optional[tuple[float, float]] = None,
        title: Optional[str] = None,
        xlim: Optional[tuple[float, float]] = None,
        ylim: Optional[tuple[float, float]] = None,
        **kwargs: Any,
    ) -> Figure:
        """Create and return a complete figure with the spectrum plot.

        Parameters
        ----------
        figsize : tuple[float, float], optional
            Figure size as (width, height) in inches. Default is (10, 3).
        title : str, optional
            Title for the plot.
        xlim : tuple[float, float], optional
            X-axis limits as (xmin, xmax). Useful for zooming into spectral regions.
            When xlim is set without ylim, the y-axis automatically scales to fit
            the data within the x-range.
        ylim : tuple[float, float], optional
            Y-axis limits as (ymin, ymax). When provided, disables automatic y-scaling.
            Use this for manual control over the y-axis range.
        **kwargs : Any
            Additional keyword arguments. These are split into:
            - Figure setup kwargs (subplot_kw, gridspec_kw, etc.) passed to setup_figure
            - Plot kwargs (alpha, linewidth, linestyle, marker, etc.) passed to ax.plot()

            Common figure setup kwargs:
            - subplot_kw : dict, optional
                Dict with keywords passed to the add_subplot call
            - gridspec_kw : dict, optional
                Dict with keywords passed to the GridSpec constructor

            Common plot kwargs:
            - alpha : float, optional (default: 0.7)
            - linewidth or lw : float, optional (default: 1.5)
            - linestyle or ls : str, optional (default: '-')
            - marker : str, optional
            - markersize : float, optional

        Returns
        -------
        Figure
            The matplotlib Figure object containing the plot.

        Examples
        --------
        Zoom into a spectral region (y-axis auto-scales):

        >>> plot = SpectrumPlot(wavenumbers, spectra, xlabel="Wavenumber (cm⁻¹)")
        >>> plot.show(title="C-H Stretch Region", xlim=(2800, 3000))

        Manual control over both axes:

        >>> plot.show(title="Custom Range", xlim=(2800, 3000), ylim=(0, 0.5))
        """
        # Separate kwargs for setup_figure vs plot
        # These are kwargs that should go to plt.subplots() via setup_figure
        figure_kwargs_keys = {"subplot_kw", "gridspec_kw", "sharex", "sharey"}
        figure_kwargs = {k: v for k, v in kwargs.items() if k in figure_kwargs_keys}
        plot_kwargs = {k: v for k, v in kwargs.items() if k not in figure_kwargs_keys}

        # Use setup_figure utility for consistent styling
        fig, ax = setup_figure(
            figsize=figsize or (10, 3),
            title=title,
            xlabel=self.xlabel,
            ylabel=self.ylabel,
            **figure_kwargs,
        )

        # Render the actual plot
        self._render_plot(ax, **plot_kwargs)

        # Apply axis limits
        if xlim is not None:
            ax.set_xlim(xlim)

            # Auto-scale y-axis to data within xlim if ylim not provided
            if ylim is None:
                ylim = self._calculate_ylim_for_xlim(xlim)

        if ylim is not None:
            ax.set_ylim(ylim)

        # Add legend or colorbar
        if self.color_by is None or self.is_categorical:
            ax.legend()
        else:
            # Add colorbar for continuous data
            from matplotlib import cm
            import matplotlib.colors as mcolors

            norm = mcolors.Normalize(vmin=self.color_by.min(), vmax=self.color_by.max())
            sm = cm.ScalarMappable(cmap=self.colormap, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax)
            cbar.set_label(self.colorbar_label, fontsize=10)

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
            X-axis limits as (xmin, xmax). When set without ylim, the y-axis
            automatically scales to fit the data within the x-range.
        ylim : tuple[float, float], optional
            Y-axis limits as (ymin, ymax). When provided, disables automatic y-scaling.
        **kwargs : Any
            Additional keyword arguments passed to the plot function.

        Returns
        -------
        fig : Figure
            The matplotlib Figure object.
        ax : Axes
            The matplotlib Axes object with the rendered plot.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        else:
            figure = ax.get_figure()
            if figure is None:
                raise ValueError("Axes object has no associated figure")
            fig = cast(Figure, figure)

        self._render_plot(ax, **kwargs)

        # Apply axis limits
        if xlim is not None:
            ax.set_xlim(xlim)

            # Auto-scale y-axis to data within xlim if ylim not provided
            if ylim is None:
                ylim = self._calculate_ylim_for_xlim(xlim)

        if ylim is not None:
            ax.set_ylim(ylim)

        return fig, ax

    def _render_plot(self, ax: Axes, **kwargs: Any) -> None:
        """Internal method to render the actual plot on given axes.

        Parameters
        ----------
        ax : Axes
            Matplotlib axes to plot on.
        **kwargs : Any
            Additional keyword arguments passed to the plot function.
        """
        alpha = kwargs.pop("alpha", 0.7)
        linewidth = kwargs.pop("linewidth", 1.5)

        if self.color_by is None:
            # No color reference: use default colors
            for spectrum, label in zip(self.y, self.labels):
                ax.plot(
                    self.x,
                    spectrum,
                    label=label,
                    alpha=alpha,
                    linewidth=linewidth,
                    **kwargs,
                )
        elif self.is_categorical:
            # Categorical coloring: use discrete colors
            colors = get_colors_from_labels(self.color_by, self.colormap)
            unique_values = np.unique(self.color_by)

            # Plot each category
            for value in unique_values:
                mask = self.color_by == value
                indices = np.where(mask)[0]

                for idx in indices:
                    # Use label only for first spectrum of each category
                    category_label: Optional[str] = (
                        str(value) if idx == indices[0] else None
                    )
                    ax.plot(
                        self.x,
                        self.y[idx],
                        color=colors[idx],
                        label=category_label,
                        alpha=alpha,
                        linewidth=linewidth,
                        **kwargs,
                    )
        else:
            # Continuous coloring: use colormap
            from matplotlib import cm
            import matplotlib.colors as mcolors

            norm = mcolors.Normalize(vmin=self.color_by.min(), vmax=self.color_by.max())
            cmap = cm.get_cmap(self.colormap)

            for i, (spectrum, value) in enumerate(zip(self.y, self.color_by)):
                color = cmap(norm(value))
                ax.plot(
                    self.x,
                    spectrum,
                    color=color,
                    alpha=alpha,
                    linewidth=linewidth,
                    **kwargs,
                )

    def _calculate_ylim_for_xlim(
        self, xlim: tuple[float, float], margin: float = 0.05
    ) -> tuple[float, float]:
        """Calculate appropriate y-axis limits for the given x-axis range.

        Parameters
        ----------
        xlim : tuple[float, float]
            The x-axis limits (xmin, xmax).
        margin : float, optional
            Fraction of the data range to add as margin (default: 0.05 = 5%).

        Returns
        -------
        tuple[float, float]
            The calculated y-axis limits (ymin, ymax).
        """
        xmin, xmax = xlim

        # Find indices within the x-range
        mask = (self.x >= xmin) & (self.x <= xmax)

        if not np.any(mask):
            # No data in range, return default limits
            return (0, 1)

        # Get y-values within the x-range
        y_in_range = self.y[:, mask]

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
