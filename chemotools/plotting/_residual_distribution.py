"""Residual distribution plot for visualizing residual histograms and normality."""

from typing import Optional, Any, Tuple
import numpy as np
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from scipy import stats

from chemotools.plotting._base import BasePlot
from chemotools.plotting._utils import validate_data


class ResidualDistributionPlot(BasePlot):
    """Histogram plot of residuals to assess normality and distribution shape.

    This class creates histogram plots of residuals with optional overlay of
    the theoretical normal distribution. Useful for visually assessing if
    residuals follow a normal distribution and detecting skewness or outliers.

    Parameters
    ----------
    residuals : np.ndarray
        Residual values with shape (n_samples,) for univariate or
        (n_samples, n_targets) for multivariate regression.
    target_index : int, optional
        For multivariate residuals, which target to plot (default: 0).
        Ignored if residuals is 1D.
    bins : int or str, optional
        Number of histogram bins or binning strategy (default: "auto").
        Can be int or any value accepted by np.histogram_bin_edges.
    density : bool, optional
        If True, normalize histogram to form probability density (default: True).
        Required for overlaying theoretical normal distribution.
    add_normal_curve : bool, optional
        Whether to overlay theoretical normal distribution curve (default: True).
    add_stats : bool, optional
        Whether to add text box with distribution statistics (default: True).
        Shows mean, std, skewness, and kurtosis.
    color : str, optional
        Color for the histogram bars (default: "#008BFB").
    alpha : float, optional
        Transparency of histogram bars (default: 0.6).

    Raises
    ------
    ValueError
        If residuals have invalid shapes.

    Examples
    --------
    **Basic histogram:**

    >>> residuals = y_true - y_pred
    >>> plot = ResidualDistributionPlot(residuals)
    >>> fig = plot.show(title="Distribution of Residuals")

    **Without normal curve overlay:**

    >>> plot = ResidualDistributionPlot(residuals, add_normal_curve=False)
    >>> fig = plot.show(title="Residual Histogram")

    **Custom number of bins:**

    >>> plot = ResidualDistributionPlot(residuals, bins=30)
    >>> fig = plot.show(title="Residual Distribution (30 bins)")

    **Multiple targets side by side:**

    >>> residuals = y_true - y_pred  # shape (n_samples, n_targets)
    >>> fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    >>> for i in range(3):
    ...     ResidualDistributionPlot(residuals, target_index=i).render(axes[i])
    ...     axes[i].set_title(f"Target {i+1}")
    >>> plt.tight_layout()
    >>> plt.show()

    **Without statistics text box:**

    >>> plot = ResidualDistributionPlot(residuals, add_stats=False)
    >>> fig = plot.show(title="Clean Histogram")

    **Count histogram instead of density:**

    >>> plot = ResidualDistributionPlot(residuals, density=False, add_normal_curve=False)
    >>> fig = plot.show(title="Residual Counts", ylabel="Count")

    Notes
    -----
    The statistics shown when add_stats=True include:
    - Mean: Should be close to 0 for good regression
    - Std: Standard deviation of residuals
    - Skewness: Measure of asymmetry (0 for normal)
    - Kurtosis: Measure of tail heaviness (0 for normal, excess kurtosis)

    For normally distributed residuals:
    - Histogram should be bell-shaped
    - Should match the overlaid normal curve
    - Skewness ≈ 0, Kurtosis ≈ 0
    """

    def __init__(
        self,
        residuals: np.ndarray,
        *,
        target_index: int = 0,
        bins: int | str = "auto",
        density: bool = True,
        add_normal_curve: bool = True,
        add_stats: bool = True,
        color: str = "#008BFB",
        alpha: float = 0.6,
    ):
        self.residuals = validate_data(residuals, name="residuals", ensure_2d=False)
        self.target_index = target_index
        self.bins = bins
        self.density = density
        self.add_normal_curve = add_normal_curve
        self.add_stats = add_stats
        self.color = color
        self.alpha = alpha

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

        # Calculate statistics
        self._calculate_statistics()

    def _validate_residuals(self) -> None:
        """Validate residuals array."""
        if self.residuals.size == 0:
            raise ValueError("residuals array cannot be empty")
        if self.residuals.size < 3:
            raise ValueError("Need at least 3 residuals for distribution plot")

    def _calculate_statistics(self) -> None:
        """Calculate distribution statistics."""
        self.mean = np.mean(self.residuals_1d)
        self.std = np.std(self.residuals_1d, ddof=1)  # Sample std
        self.skewness = stats.skew(self.residuals_1d)
        self.kurtosis = stats.kurtosis(self.residuals_1d)  # Excess kurtosis

    def _get_default_labels(self) -> dict[str, str]:
        if self.residuals.ndim == 2:
            title = f"Residual Distribution for Target {self.target_index + 1}"
        else:
            title = "Residual Distribution"

        return {
            "xlabel": "Residuals",
            "ylabel": "Density" if self.density else "Count",
            "title": title,
        }

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
        """Create and return a complete figure with the residual distribution plot.

        This method handles figure creation and then delegates to `render()`.

        Parameters
        ----------
        figsize : tuple[float, float], optional
            Figure size in inches (width, height).
        title : str, optional
            Figure title.
        xlabel : str, optional
            Custom x-axis label. If None, uses existing label or default.
        ylabel : str, optional
            Custom y-axis label. If None, uses existing label or default.
        xlim : tuple[float, float], optional
            X-axis limits as (xmin, xmax).
        ylim : tuple[float, float], optional
            Y-axis limits as (ymin, ymax).
        **kwargs : Any
            Additional keyword arguments passed to the render() method.

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
            Additional keyword arguments passed to histogram.

        Returns
        -------
        tuple[Figure, Axes]
            The Figure and Axes objects containing the plot.
        """
        fig, ax = super().render(
            ax=ax,
            xlabel=xlabel,
            ylabel=ylabel,
            xlim=xlim,
            ylim=ylim,
            **kwargs,
        )

        return fig, ax

    def _render_plot(self, ax: Axes, **kwargs: Any) -> None:
        """Internal method to render the plot on given axes."""
        # Create histogram
        hist_kwargs = {
            "bins": kwargs.get("bins", self.bins),
            "density": self.density,
            "alpha": self.alpha,
            "color": self.color,
            "edgecolor": "black",
            "linewidth": 0.5,
        }

        counts, bins, patches = ax.hist(self.residuals_1d, **hist_kwargs)

        # Add normal distribution curve if requested
        if self.add_normal_curve and self.density:
            x = np.linspace(self.residuals_1d.min(), self.residuals_1d.max(), 200)
            normal_pdf = stats.norm.pdf(x, loc=self.mean, scale=self.std)
            ax.plot(
                x,
                normal_pdf,
                "r-",
                linewidth=2.0,
                alpha=0.8,
                label=f"Normal(μ={self.mean:.3f}, σ={self.std:.3f})",
            )
            ax.legend()

        # Add statistics text box if requested
        if self.add_stats:
            stats_text = (
                f"Mean: {self.mean:.4f}\n"
                f"Std: {self.std:.4f}\n"
                f"Skewness: {self.skewness:.4f}\n"
                f"Kurtosis: {self.kurtosis:.4f}"
            )
            # Place text box in upper right
            ax.text(
                0.95,
                0.95,
                stats_text,
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment="top",
                horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )
