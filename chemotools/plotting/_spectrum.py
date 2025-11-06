"""Spectrum plot class for spectral data visualization."""

from typing import Optional, Any
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
    ):
        self.x = x
        self.y = y if y.ndim == 2 else y.reshape(1, -1)
        self.labels = labels or [f"Spectrum {i}" for i in range(len(self.y))]
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.color_by = color_by
        
        # Determine if color_by is categorical or continuous
        self.is_categorical = False
        if color_by is not None:
            # Check if categorical (strings or small number of unique values)
            if color_by.dtype.kind in ['U', 'S', 'O']:  # String types
                self.is_categorical = True
            elif len(np.unique(color_by)) < 10:  # Heuristic: < 10 unique values
                self.is_categorical = True
        
        # Set colormap with colorblind-friendly defaults
        if colormap is None:
            self.colormap = "tab10" if self.is_categorical else "viridis"
        else:
            self.colormap = colormap
        
    def show(
        self,
        figsize: Optional[tuple[float, float]] = None,
        title: Optional[str] = None,
        **kwargs: Any,
    ) -> Figure:
        """Create and return a complete figure with the spectrum plot.
        
        Parameters
        ----------
        figsize : tuple[float, float], optional
            Figure size as (width, height) in inches. Default is (10, 6).
        title : str, optional
            Title for the plot.
        **kwargs : Any
            Additional keyword arguments passed to the plot function.
            
        Returns
        -------
        Figure
            The matplotlib Figure object containing the plot.
        """
        # Use setup_figure utility for consistent styling
        fig, ax = setup_figure(
            figsize=figsize or (10, 3),
            title=title,
            xlabel=self.xlabel,
            ylabel=self.ylabel,
        )
        
        # Render the actual plot
        self._render_plot(ax, **kwargs)
        
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
            cbar.set_label('Reference Value', fontsize=10)
        
        plt.tight_layout()
        return fig
    
    def render(
        self,
        ax: Optional[Axes] = None,
        **kwargs: Any,
    ) -> tuple[Figure, Axes]:
        """Render the plot on the given axes or create new ones.
        
        Parameters
        ----------
        ax : Axes, optional
            Matplotlib axes to plot on. If None, creates new figure and axes.
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
            fig = ax.figure
            
        self._render_plot(ax, **kwargs)
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
                    label = str(value) if idx == indices[0] else None
                    ax.plot(
                        self.x,
                        self.y[idx],
                        color=colors[idx],
                        label=label,
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
