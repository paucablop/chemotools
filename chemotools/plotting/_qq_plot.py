"""Q-Q plot for assessing normality of residuals."""

from typing import Optional, Any
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from scipy import stats

from chemotools.plotting._utils import (
    setup_figure,
    annotate_points,
    split_figure_plot_kwargs,
    ensure_axes,
    apply_limits,
    set_default_axis_labels,
)


class QQPlot:
    """Quantile-Quantile plot to assess if residuals follow a normal distribution.

    This class creates Q-Q plots comparing the quantiles of residuals against
    theoretical quantiles from a normal distribution. Points falling on the
    diagonal line indicate normality, while deviations suggest non-normality.

    Parameters
    ----------
    residuals : np.ndarray
        Residual values with shape (n_samples,) for univariate or
        (n_samples, n_targets) for multivariate regression.
    target_index : int, optional
        For multivariate residuals, which target to plot (default: 0).
        Ignored if residuals is 1D.
    annotations : list[str], optional
        Labels for annotating individual points (e.g., outliers).
    label : str, optional
        Legend label for this dataset (default: "Residuals").
    color : str, optional
        Color for the points (default: "steelblue").
    add_reference_line : bool, optional
        Whether to add the diagonal reference line (default: True).
    add_confidence_band : bool or float, optional
        Whether to add confidence bands around the reference line.
        - If True: uses 95% confidence band
        - If float: uses specified confidence level (0 < level < 1)
        - If False or None: no bands (default: None)

    Raises
    ------
    ValueError
        If residuals have invalid shapes.

    Examples
    --------
    **Basic Q-Q plot:**

    >>> residuals = y_true - y_pred
    >>> plot = QQPlot(residuals)
    >>> fig = plot.show(title="Q-Q Plot of Residuals")

    **With confidence bands:**

    >>> plot = QQPlot(residuals, add_confidence_band=0.95)
    >>> fig = plot.show(title="Q-Q Plot with 95% Confidence Band")

    **Multiple datasets compared:**

    >>> fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    >>> QQPlot(train_residuals, label="Train").render(axes[0])
    >>> QQPlot(test_residuals, label="Test").render(axes[1])
    >>> plt.show()

    **Multivariate regression - plot specific target:**

    >>> residuals = y_true - y_pred  # shape (n_samples, n_targets)
    >>> plot = QQPlot(residuals, target_index=1)
    >>> fig = plot.show(title="Q-Q Plot for Target 2")

    **With outlier annotations:**

    >>> outlier_indices = [5, 23, 47]
    >>> annotations = [f"S{i}" if i in outlier_indices else "" for i in range(len(residuals))]
    >>> plot = QQPlot(residuals, annotations=annotations)
    >>> fig = plot.show(title="Q-Q Plot with Outliers")

    Notes
    -----
    The Q-Q plot compares:
    - X-axis: Theoretical quantiles from standard normal distribution N(0,1)
    - Y-axis: Sample quantiles (standardized residuals)

    Points should fall approximately on the diagonal line y=x if residuals
    are normally distributed. Common patterns:
    - S-curve: Heavy or light tails
    - Points above line: Right skew
    - Points below line: Left skew
    """

    def __init__(
        self,
        residuals: np.ndarray,
        *,
        target_index: int = 0,
        annotations: Optional[list[str]] = None,
        label: str = "Residuals",
        color: str = "steelblue",
        add_reference_line: bool = True,
        add_confidence_band: Optional[bool | float] = None,
    ):
        self.residuals = np.asarray(residuals)
        self.target_index = target_index
        self.annotations = annotations
        self.label = label
        self.color = color
        self.add_reference_line = add_reference_line
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

        # Calculate Q-Q plot data
        self._calculate_qq_data()

    def _validate_residuals(self) -> None:
        """Validate residuals array."""
        if self.residuals.size == 0:
            raise ValueError("residuals array cannot be empty")
        if self.residuals.size < 3:
            raise ValueError("Need at least 3 residuals for Q-Q plot")

    def _calculate_qq_data(self) -> None:
        """Calculate theoretical and sample quantiles for Q-Q plot."""
        # Use scipy.stats.probplot to get the Q-Q plot data
        # probplot returns ((theoretical_quantiles, ordered_values), (slope, intercept, r))
        (
            (self.theoretical_quantiles, self.sample_quantiles),
            (
                self.slope,
                self.intercept,
                self.r_value,
            ),
        ) = stats.probplot(self.residuals_1d, dist="norm")

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
        """Create and display the Q-Q plot.

        Parameters
        ----------
        title : str, optional
            Plot title (default: "Q-Q Plot").
        xlabel : str, optional
            X-axis label (default: "Theoretical Quantiles").
        ylabel : str, optional
            Y-axis label (default: "Sample Quantiles").
        figsize : tuple[float, float], optional
            Figure size (width, height) in inches (default: (8, 8)).
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
        figure_kwargs, plot_kwargs = split_figure_plot_kwargs(kwargs)

        # Auto-generate labels if not provided
        if xlabel is None:
            xlabel = "Theoretical Quantiles"
        if ylabel is None:
            ylabel = "Sample Quantiles"
        if title is None:
            if self.residuals.ndim == 2:
                title = f"Q-Q Plot for Target {self.target_index + 1}"
            else:
                title = "Q-Q Plot"

        # Use setup_figure utility for consistent styling
        fig, ax = setup_figure(
            figsize=figsize or (8, 8),
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            **figure_kwargs,
        )

        # Render the actual plot
        self._render_plot(ax, **plot_kwargs)

        # Apply axis limits
        apply_limits(ax, xlim=xlim, ylim=ylim)

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
        fig, ax = ensure_axes(ax, figsize=(10, 6))

        # Render the plot
        self._render_plot(ax, **kwargs)

        # Set default labels if axes don't have them
        set_default_axis_labels(
            ax,
            xlabel="Theoretical Quantiles",
            ylabel="Sample Quantiles",
        )

        # Apply axis limits
        apply_limits(ax, xlim=xlim, ylim=ylim)

        return fig, ax

    def _render_plot(self, ax: Axes, **kwargs: Any) -> None:
        """Internal method to render the plot on given axes."""
        # Create scatter plot of theoretical vs sample quantiles
        scatter_kwargs = {
            "alpha": kwargs.get("alpha", 0.7),
            "s": kwargs.get("s", 50),
            "edgecolors": kwargs.get("edgecolors", "black"),
            "linewidths": kwargs.get("linewidths", 0.5),
            "label": self.label,
        }

        ax.scatter(
            self.theoretical_quantiles,
            self.sample_quantiles,
            c=self.color,
            **scatter_kwargs,
        )

        # Add reference line (diagonal)
        if self.add_reference_line:
            # Calculate the line based on the fit
            line_x = np.array(
                [self.theoretical_quantiles.min(), self.theoretical_quantiles.max()]
            )
            line_y = self.slope * line_x + self.intercept

            ax.plot(
                line_x,
                line_y,
                "r-",
                linewidth=2,
                alpha=0.8,
                label=f"Reference Line (R²={self.r_value**2:.3f})",
            )

        # Add confidence bands if requested
        if self.add_confidence_band is not None:
            if isinstance(self.add_confidence_band, bool):
                confidence_level = 0.95
            else:
                confidence_level = float(self.add_confidence_band)

            # Calculate confidence bands using standard error
            n = len(self.residuals_1d)
            se = np.std(self.residuals_1d) * np.sqrt(
                (1 / n)
                + (self.theoretical_quantiles**2)
                / np.sum(self.theoretical_quantiles**2)
            )

            # Critical value for the confidence level
            z_crit = stats.norm.ppf((1 + confidence_level) / 2)

            upper_band = (
                self.slope * self.theoretical_quantiles + self.intercept + z_crit * se
            )
            lower_band = (
                self.slope * self.theoretical_quantiles + self.intercept - z_crit * se
            )

            ax.fill_between(
                self.theoretical_quantiles,
                lower_band,
                upper_band,
                color="red",
                alpha=0.2,
                label=f"{confidence_level * 100:.0f}% Confidence Band",
            )

        # Add annotations if provided
        if self.annotations:
            annotate_points(
                ax, self.theoretical_quantiles, self.sample_quantiles, self.annotations
            )

        # Enforce equal scaling so the reference line is visually meaningful
        ax.set_aspect("equal", adjustable="box")

        # When available make the axes box square to avoid tiny drawing areas
        set_box_aspect = getattr(ax, "set_box_aspect", None)
        if callable(set_box_aspect):
            set_box_aspect(1)
