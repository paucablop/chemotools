"""Y residuals plot for regression diagnostics and homoscedasticity analysis."""

from typing import Optional, Any
import numpy as np
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from chemotools.plotting._base import BasePlot, ColoringMixin
from chemotools.plotting._utils import (
    get_colors_from_labels,
    annotate_points,
)


class YResidualsPlot(BasePlot, ColoringMixin):
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
        self.residuals = residuals
        self.x_values = x_values
        self.target_index = target_index
        self.annotations = annotations
        self.label = label
        self.color = color
        self.add_zero_line = add_zero_line
        self.add_confidence_band = add_confidence_band

        self._validate_residuals()
        self._init_xy_data()

        # Initialize coloring
        self._init_coloring(color_by, colormap)

    def _validate_residuals(self) -> None:
        """Validate residuals shape and target index."""
        if self.residuals.size == 0:
            raise ValueError("residuals array cannot be empty")

        if self.residuals.ndim == 1:
            self.residuals_1d = self.residuals
        elif self.residuals.ndim == 2:
            n_targets = self.residuals.shape[1]
            if self.target_index < 0 or self.target_index >= n_targets:
                raise ValueError(
                    f"Invalid target_index {self.target_index}. "
                    f"Residuals have {n_targets} targets."
                )
            self.residuals_1d = self.residuals[:, self.target_index]
        else:
            raise ValueError(
                f"Residuals must be 1D or 2D array, got shape {self.residuals.shape}"
            )

    def _init_xy_data(self) -> None:
        """Initialize x and y data for plotting."""
        n_samples = self.residuals_1d.shape[0]

        if self.x_values is None:
            self.x_axis = np.arange(n_samples)
            self.x_label = "Sample Index"
        else:
            if self.x_values.shape[0] != n_samples:
                raise ValueError(
                    f"x_values length ({self.x_values.shape[0]}) must match "
                    f"residuals length ({n_samples})"
                )
            self.x_axis = self.x_values
            self.x_label = "X Values"

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
        """Create and return a complete figure with the residuals plot.

        Parameters
        ----------
        figsize : tuple[float, float], optional
            Figure size as (width, height) in inches. Default is (10, 6).
        title : str, optional
            Title for the plot.
        xlabel : str, optional
            Custom x-axis label. If None, uses "Sample Index" or "X Values".
        ylabel : str, optional
            Custom y-axis label. If None, uses "Residuals".
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
        """
        xlabel_text = xlabel if xlabel is not None else self.x_label
        ylabel_text = ylabel if ylabel is not None else "Residuals"

        if title is None:
            if self.residuals.ndim == 2:
                title = f"Residuals Plot - Target {self.target_index + 1}"
            else:
                title = "Residuals Plot"

        return super().show(
            figsize=figsize or (10, 6),
            title=title,
            xlabel=xlabel_text,
            ylabel=ylabel_text,
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
        """Render the plot on existing or new axes.

        Parameters
        ----------
        ax : Axes, optional
            Matplotlib axes to render on. If None, creates new figure/axes.
        xlabel : str, optional
            Custom x-axis label. If None, uses existing label or default.
        ylabel : str, optional
            Custom y-axis label. If None, uses existing label or default.
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
        xlabel_text = xlabel if xlabel is not None else self.x_label
        ylabel_text = ylabel if ylabel is not None else "Residuals"

        fig, ax = super().render(
            ax=ax,
            xlabel=xlabel_text,
            ylabel=ylabel_text,
            xlim=xlim,
            ylim=ylim,
            **kwargs,
        )

        # Add colorbar for continuous data
        self._add_colorbar_if_needed(ax)

        # Add legend if categorical or if label is provided
        if self.is_categorical or self.label:
            # Only add legend if there are labeled artists
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend()

        return fig, ax

    def _render_plot(self, ax: Axes, **kwargs: Any) -> None:
        """Internal method to render the plot on given axes."""
        alpha = kwargs.pop("alpha", 0.6)
        s = kwargs.pop("s", 50)
        edgecolors = kwargs.pop("edgecolors", "black")
        linewidths = kwargs.pop("linewidths", 0.5)

        # Determine colors
        if self.color_by is not None:
            if self.is_categorical:
                assert self.colormap is not None
                colors = get_colors_from_labels(self.color_by, colormap=self.colormap)
                unique_values = np.unique(self.color_by)

                for value in unique_values:
                    mask = self.color_by == value
                    ax.scatter(
                        self.x_axis[mask],
                        self.residuals_1d[mask],
                        color=colors[mask][0],
                        label=f"{self.label} - {value}",
                        alpha=alpha,
                        s=s,
                        edgecolors=edgecolors,
                        linewidths=linewidths,
                        **kwargs,
                    )
            else:
                # Continuous
                import matplotlib as mpl
                import matplotlib.colors as mcolors

                norm = mcolors.Normalize(
                    vmin=self.color_by.min(), vmax=self.color_by.max()
                )
                colormap_name = (
                    self.colormap if self.colormap is not None else "viridis"
                )
                cmap = mpl.colormaps.get_cmap(colormap_name)

                ax.scatter(
                    self.x_axis,
                    self.residuals_1d,
                    c=self.color_by,
                    cmap=cmap,
                    norm=norm,
                    label=self.label,
                    alpha=alpha,
                    s=s,
                    edgecolors=edgecolors,
                    linewidths=linewidths,
                    **kwargs,
                )
        else:
            # Single color
            color = self.color if self.color is not None else "steelblue"
            ax.scatter(
                self.x_axis,
                self.residuals_1d,
                c=color,
                label=self.label,
                alpha=alpha,
                s=s,
                edgecolors=edgecolors,
                linewidths=linewidths,
                **kwargs,
            )

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
