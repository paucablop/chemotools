"""
The :mod:`chemotools.plotting._spectra` module implements the SpectraPlot class for visualizing spectral data.
"""

# Authors: Pau Cabaneros
# License: MIT

from typing import Literal, Optional, Any, Sequence
import numpy as np
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from chemotools.plotting._base import BasePlot, ColoringMixin
from chemotools.plotting._utils import (
    get_colors_from_labels,
    calculate_ylim_for_xlim,
    validate_data,
)


class SpectraPlot(BasePlot, ColoringMixin):
    """Plot class for visualizing spectral data.

    This class implements the Display protocol and provides flexible options
    for plotting spectral data with categorical or continuous coloring.

    Parameters
    ----------
    x : np.ndarray
        X-axis data (e.g., wavelengths, wavenumbers).
    y : np.ndarray
        Y-axis data (e.g., spectra intensities). Can be 1D or 2D.
    labels : list[str] or list[str | None], optional
        Labels for each spectrum (used for legend).
    color_by : np.ndarray, optional
        Reference vector for coloring spectra. Can be:

        - Categorical (class labels): uses discrete colormap
        - Continuous (numeric values): uses continuous colormap

    colormap : str, optional
        Colormap name. Colorblind-friendly defaults:

        - "tab10" for categorical data (default)
        - "shap" for continuous data

        Other options: "plasma", "cividis", "coolwarm"
    color_mode : str, optional
        Explicitly specify coloring mode ("continuous" or "categorical").
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
    >>> plotter = SpectraPlot(x, y)
    >>> fig = plotter.show(title="NIR Spectra", xlabel="Wavelength (nm)", ylabel="Absorbance")

    With categorical coloring:

    >>> classes = np.array(['A', 'A', 'B', 'B', 'C'])
    >>> plotter = SpectraPlot(x, y, color_by=classes)
    >>> fig = plotter.show(title="Spectra by Class")

    With continuous coloring:

    >>> concentrations = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    >>> plotter = SpectraPlot(x, y, color_by=concentrations, colormap="viridis")
    >>> fig = plotter.show(title="Spectra by Concentration")

    With custom colorbar label:

    >>> plotter = SpectraPlot(
    ...     x, y, color_by=concentrations,
    ...     colormap="viridis", colorbar_label="Concentration (mg/L)"
    ... )
    >>> fig = plotter.show(title="Spectra by Concentration")

    Override categorical detection for small numeric datasets:

    >>> levels = np.array([1, 2, 3, 4])  # 4 unique values - might be detected as categorical
    >>> plotter = SpectraPlot(x, y, color_by=levels, color_mode="continuous")
    >>> fig = plotter.show(title="4 Concentration Levels")

    With custom axis labels:

    >>> plotter = SpectraPlot(x, y)
    >>> fig = plotter.show(
    ...     title="Raman Spectra",
    ...     xlabel="Wavenumber (cm⁻¹)",
    ...     ylabel="Intensity"
    ... )

    Creating subplots:

    >>> fig, axes = plt.subplots(2, 1)
    >>> plotter.render(ax=axes[0])
    >>> plotter.render(ax=axes[1])
    >>> plt.tight_layout()
    """

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *,
        labels: Optional[Sequence[Optional[str]]] = None,
        color_by: Optional[np.ndarray] = None,
        colormap: Optional[str] = None,
        color_mode: Optional[Literal["continuous", "categorical"]] = None,
        colorbar_label: str = "Reference Value",
    ):
        self.x = validate_data(x, name="x", ensure_2d=False)
        y = validate_data(y, name="y", ensure_2d=False)
        self.y = y if y.ndim == 2 else y.reshape(1, -1)

        # Store whether labels were explicitly provided
        self._labels_provided = labels is not None
        self.labels = labels or [f"Spectrum {i}" for i in range(len(self.y))]

        if color_by is not None:
            color_by = validate_data(
                color_by, name="color_by", ensure_2d=False, numeric=False
            )

        self._init_coloring(
            color_by, colormap, color_mode=color_mode, colorbar_label=colorbar_label
        )

    def _get_default_labels(self) -> dict[str, str]:
        return {
            "xlabel": "X-axis",
            "ylabel": "Y-axis",
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
        """Show the spectra plot with given figure size and labels.

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
            X-axis label. Default is "X-axis".
        ylabel : str, optional
            Y-axis label. Default is "Y-axis".
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
        # Auto-scale y-axis to data within xlim if ylim not provided
        if xlim is not None and ylim is None:
            ylim = calculate_ylim_for_xlim(self.x, self.y, xlim)

        fig, ax = super().render(
            ax=ax,
            xlabel=xlabel,
            ylabel=ylabel,
            xlim=xlim,
            ylim=ylim,
            **kwargs,
        )

        # Add legend or colorbar
        if self.color_by is None or self.is_categorical:
            # Only add legend if:
            # 1. Labels were explicitly provided by user, OR
            # 2. Number of spectra is small (≤ 10) and there are labeled artists, OR
            # 3. Coloring is categorical (shows categories in legend)
            handles, labels = ax.get_legend_handles_labels()
            should_show_legend = (
                (self._labels_provided or len(self.y) <= 10 or self.is_categorical)
                and handles
                and any(label for label in labels)
            )

            if should_show_legend:
                ax.legend()
        else:
            # Add colorbar for continuous data
            self._add_colorbar_if_needed(ax)

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
            # Only use labels if explicitly provided or small number of spectra
            use_labels = self._labels_provided or len(self.y) <= 10

            for spectrum, label in zip(self.y, self.labels):
                ax.plot(
                    self.x,
                    spectrum,
                    label=label if use_labels else None,
                    alpha=alpha,
                    linewidth=linewidth,
                    **kwargs,
                )
        elif self.is_categorical:
            # Categorical coloring: use discrete colors
            assert self.colormap is not None
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
            import matplotlib as mpl
            import matplotlib.colors as mcolors

            norm = mcolors.Normalize(vmin=self.color_by.min(), vmax=self.color_by.max())
            # Ensure we have a valid colormap (should not be None here, but be defensive)
            colormap_name = self.colormap if self.colormap is not None else "viridis"
            cmap = mpl.colormaps.get_cmap(colormap_name)

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
