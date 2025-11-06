"""Display protocol for consistent plotting interface across chemotools."""

from typing import Protocol, Optional, Any, runtime_checkable
from matplotlib.figure import Figure
from matplotlib.axes import Axes


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
    
    def show(
        self,
        figsize: Optional[tuple[float, float]] = None,
        title: Optional[str] = None,
        **kwargs: Any,
    ) -> Figure:
        """Create and return a complete figure with the plot.
        
        This method creates a new figure and displays the plot on it.
        Use this when you want a standalone visualization.
        
        Parameters
        ----------
        figsize : tuple[float, float], optional
            Figure size as (width, height) in inches.
        title : str, optional
            Title for the plot.
        **kwargs : Any
            Additional keyword arguments for customizing the plot.
            
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
    
    def render(
        self,
        ax: Optional[Axes] = None,
        **kwargs: Any,
    ) -> tuple[Figure, Axes]:
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
        fig : Figure
            The matplotlib Figure object.
        ax : Axes
            The matplotlib Axes object with the rendered plot.
            
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
