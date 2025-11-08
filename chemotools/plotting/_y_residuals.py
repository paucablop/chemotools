"""Y residuals plot for regression diagnostics and homoscedasticity analysis."""

from typing import Optional, Any
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
)


class YResidualsPlot:
    """Plot of residuals to assess homoscedasticity and model fit quality.

    This class creates scatter plots of Y residuals (observed - predicted) versus
    sample index or a given vector (e.g., predicted values, experimental conditions).
    Useful for detecting heteroscedasticity, patterns in residuals, and model issues.

    Parameters
    ----------
    residuals : np.ndarray
        Residual values with shape (n_samples,) for univariate or
        (n_samples, n_targets) for multivariate regression.
        Residuals should be calculated as (y_true - y_pred).
    x_values : np.ndarray, optional
        Values for the x-axis. If None, uses sample indices (0, 1, 2, ...).
        Common choices: predicted values, experimental conditions, time points.
        Shape should be (n_samples,) or broadcastable to residuals shape.
    target_index : int, optional
        For multivariate residuals, which target to plot (default: 0).
        Ignored if residuals is 1D.
    color_by : np.ndarray, optional
        Values for coloring samples. Can be either:
        - Continuous (numeric): shows colorbar
        - Categorical (strings/classes): shows legend with discrete colors
    annotations : list[str], optional
        Labels for annotating individual points.
    label : str, optional
        Legend label for this dataset (default: "Residuals").
    color : str, optional
        Color for all points when color_by is None (default: auto-assigned).
    colormap : str, optional
        Colormap name. Colorblind-friendly defaults:
        - "tab10" for categorical data
        - "viridis" for continuous data
    add_zero_line : bool, optional
        Whether to add a horizontal line at y=0 (default: True).
    add_confidence_band : bool or float, optional
        Whether to add confidence bands (±n*std) around zero.
        - If True: uses ±2*std (95% for normal distribution)
        - If float: uses ±value*std
        - If False or None: no bands (default: None)

    Raises
    ------
    ValueError
        If residuals have invalid shapes or x_values shape mismatch.

    Examples
    --------
    **Simple residuals plot vs sample index:**

    >>> residuals = y_true - y_pred
    >>> plot = YResidualsPlot(residuals)
    >>> fig = plot.show(title="Residuals vs Sample Index")

    **Residuals vs predicted values (check for heteroscedasticity):**

    >>> plot = YResidualsPlot(residuals, x_values=y_pred)
    >>> fig = plot.show(
    ...     title="Residuals vs Predicted",
    ...     xlabel="Predicted Values",
    ...     ylabel="Residuals"
    ... )

    **With confidence bands:**

    >>> plot = YResidualsPlot(
    ...     residuals,
    ...     x_values=y_pred,
    ...     add_confidence_band=2.0  # ±2 standard deviations
    ... )
    >>> fig = plot.show(title="Residuals with 95% Confidence Band")

    **Multiple datasets composed together:**

    >>> fig, ax = plt.subplots()
    >>> YResidualsPlot(train_residuals, label="Train", color="blue").render(ax)
    >>> YResidualsPlot(test_residuals, label="Test", color="red").render(ax)
    >>> ax.legend()
    >>> plt.show()

    **With categorical coloring:**

    >>> plot = YResidualsPlot(residuals, x_values=y_pred, color_by=classes)
    >>> fig = plot.show(title="Residuals by Class")

    **Multivariate regression - plot specific target:**

    >>> residuals = y_true - y_pred  # shape (n_samples, n_targets)
    >>> plot = YResidualsPlot(residuals, target_index=1)  # Second target
    >>> fig = plot.show(title="Residuals for Target 2")
    """

    def __init__(
        self,
        residuals: np.ndarray,
        *,
        x_values: Optional[np.ndarray] = None,
        target_index: int = 0,
        color_by: Optional[np.ndarray] = None,
        annotations: Optional[list[str]] = None,
        label: str = "Residuals",
        color: Optional[str] = None,
        colormap: Optional[str] = None,
        add_zero_line: bool = True,
        add_confidence_band: Optional[bool | float] = None,
    ):
        self.residuals = np.asarray(residuals)
        self.x_values = x_values if x_values is None else np.asarray(x_values)
        self.target_index = target_index
        self.color_by = color_by
        self.annotations = annotations
        self.label = label
        self.color = color
        self.add_zero_line = add_zero_line
        self.add_confidence_band = add_confidence_band

        # Validate inputs
        self._validate_residuals()

        # Extract the specific target's residuals if multivariate
        if self.residuals.ndim == 2:
            if target_index >= self.residuals.shape[1]:
                raise ValueError(
                    f"target_index {target_index} is out of bounds for "
                    f"residuals with {self.residuals.shape[1]} targets"
                )
            self.residuals_1d = self.residuals[:, target_index]
        elif self.residuals.ndim == 1:
            self.residuals_1d = self.residuals
        else:
            raise ValueError("residuals must be 1D or 2D array")

        # Set up x-axis values
        if self.x_values is None:
            self.x_axis = np.asarray(np.arange(len(self.residuals_1d)))
            self.x_label = "Sample Index"
        else:
            if len(self.x_values) != len(self.residuals_1d):
                raise ValueError(
                    f"x_values length ({len(self.x_values)}) must match "
                    f"residuals length ({len(self.residuals_1d)})"
                )
            self.x_axis = self.x_values
            self.x_label = "X"

        # Detect if color_by is categorical
        self.is_categorical = (
            detect_categorical(color_by) if color_by is not None else False
        )

        # Get colormap
        self.colormap = get_default_colormap(self.is_categorical, colormap)

    def _validate_residuals(self) -> None:
        """Validate residuals array."""
        if self.residuals.size == 0:
            raise ValueError("residuals array cannot be empty")

    def show(
        self,
        *,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        figsize: Optional[tuple[float, float]] = None,
        xlim: Optional[tuple[float, float]] = None,
        ylim: Optional[tuple[float, float]] = None,
        **kwargs: Any,
    ) -> Figure:
        """Create and display the residuals plot.

        Parameters
        ----------
        title : str, optional
            Plot title (default: auto-generated based on configuration).
        xlabel : str, optional
            X-axis label (default: auto-generated).
        ylabel : str, optional
            Y-axis label (default: "Residuals").
        figsize : tuple[float, float], optional
            Figure size (width, height) in inches (default: (10, 6)).
        xlim : tuple[float, float], optional
            X-axis limits (min, max).
        ylim : tuple[float, float], optional
            Y-axis limits (min, max).
        **kwargs : Any
            Additional keyword arguments passed to setup_figure.

        Returns
        -------
        Figure
            The matplotlib Figure object containing the plot.
        """
        # Separate kwargs for setup_figure vs plot
        figure_kwargs_keys = {"subplot_kw", "gridspec_kw", "sharex", "sharey"}
        figure_kwargs = {k: v for k, v in kwargs.items() if k in figure_kwargs_keys}
        plot_kwargs = {k: v for k, v in kwargs.items() if k not in figure_kwargs_keys}

        # Auto-generate labels if not provided
        if xlabel is None:
            xlabel = self.x_label
        if ylabel is None:
            ylabel = "Residuals"
        if title is None:
            if self.residuals.ndim == 2:
                title = f"Residuals for Target {self.target_index + 1}"
            else:
                title = "Residuals Plot"

        # Use setup_figure utility for consistent styling
        fig, ax = setup_figure(
            figsize=figsize or (10, 6),
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            **figure_kwargs,
        )

        # Render the actual plot
        self._render_plot(ax, **plot_kwargs)

        # Apply axis limits
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)

        # Add grid
        ax.grid(alpha=0.3, linestyle="--")

        plt.tight_layout()
        return fig

    def render(
        self,
        ax: Optional[Axes] = None,
        *,
        xlim: Optional[tuple[float, float]] = None,
        ylim: Optional[tuple[float, float]] = None,
        **kwargs: Any,
    ) -> tuple[Figure, Axes]:
        """Render the plot on existing or new axes.

        Parameters
        ----------
        ax : Axes, optional
            Matplotlib axes to render on. If None, creates new figure/axes.
        xlim : tuple[float, float], optional
            X-axis limits (min, max).
        ylim : tuple[float, float], optional
            Y-axis limits (min, max).
        **kwargs : Any
            Additional keyword arguments passed to scatter plot.

        Returns
        -------
        tuple[Figure, Axes]
            The Figure and Axes objects containing the plot.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        else:
            _fig = ax.get_figure()
            if _fig is None:
                raise ValueError("Provided axes must be attached to a figure")
            # Type narrowing for mypy
            assert isinstance(_fig, Figure)
            fig = _fig

        # Render the plot
        self._render_plot(ax, **kwargs)

        # Set default labels if axes don't have them
        if not ax.get_xlabel():
            ax.set_xlabel(self.x_label)
        if not ax.get_ylabel():
            ax.set_ylabel("Residuals")

        # Apply axis limits
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)

        return fig, ax

    def _render_plot(self, ax: Axes, **kwargs: Any) -> None:
        """Internal method to render the plot on given axes."""
        # Determine colors for points
        colors: np.ndarray | str
        if self.color_by is not None:
            if self.is_categorical:
                colors = get_colors_from_labels(self.color_by, colormap=self.colormap)
            else:
                # Will use scatter's c parameter for continuous coloring
                colors = self.color_by
        elif self.color is not None:
            colors = self.color
        else:
            colors = "steelblue"

        # Create scatter plot
        scatter_kwargs = {
            "alpha": kwargs.get("alpha", 0.6),
            "s": kwargs.get("s", 50),
            "edgecolors": kwargs.get("edgecolors", "black"),
            "linewidths": kwargs.get("linewidths", 0.5),
            "label": self.label,
        }

        if isinstance(colors, np.ndarray) and not self.is_categorical:
            # Continuous coloring
            ax.scatter(
                self.x_axis,
                self.residuals_1d,
                c=colors,
                cmap=self.colormap,
                **scatter_kwargs,
            )
            # Add colorbar
            if ax.get_figure() is not None:
                add_colorbar(ax, colors, self.colormap)
        else:
            # Single color or categorical
            ax.scatter(self.x_axis, self.residuals_1d, c=colors, **scatter_kwargs)

        # Add zero reference line
        if self.add_zero_line:
            ax.axhline(y=0, color="black", linestyle="-", linewidth=1.5, alpha=0.7)

        # Add confidence bands if requested
        if self.add_confidence_band is not None:
            std = np.std(self.residuals_1d)
            if isinstance(self.add_confidence_band, bool):
                n_std = 2.0  # Default to ±2σ (95% for normal)
            else:
                n_std = float(self.add_confidence_band)

            ax.axhline(
                y=n_std * std,
                color="red",
                linestyle="--",
                linewidth=1.5,
                alpha=0.5,
                label=f"±{n_std:.1f}σ",
            )
            ax.axhline(
                y=-n_std * std,
                color="red",
                linestyle="--",
                linewidth=1.5,
                alpha=0.5,
            )
            ax.fill_between(
                [ax.get_xlim()[0], ax.get_xlim()[1]],
                -n_std * std,
                n_std * std,
                color="red",
                alpha=0.1,
            )

        # Add annotations if provided
        if self.annotations:
            annotate_points(ax, self.x_axis, self.residuals_1d, self.annotations)
