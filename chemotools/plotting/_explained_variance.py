"""Explained Variance plot for PCA/PLS model diagnostics."""

from typing import Optional, Any
import numpy as np
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from chemotools.plotting._base import BasePlot
from chemotools.plotting._utils import validate_data


class ExplainedVariancePlot(BasePlot):
    """Visualize explained variance by component with cumulative variance.

    Shows both individual and cumulative explained variance ratios across
    components. Works with any decomposition method (PCA, PLS, ICA, etc.)
    to help determine the optimal number of components.

    **Works with:**
    - PCA: Use `pca.explained_variance_ratio_` directly
    - PLS: Use `chemotools.models.PLSRegression` for automatic variance calculation
    - Any method: Just provide an array of variance ratios per component

    Parameters
    ----------
    explained_variance_ratio : np.ndarray
        Array of explained variance ratios for each component.
        Should be 1D array with values between 0 and 1.
        For PCA, use `model.explained_variance_ratio_` directly.
        For PLS, use `chemotools.models.PLSRegression` which provides
        `explained_x_variance_ratio_` and `explained_y_variance_ratio_`.
    threshold : float or None, optional
        If provided, draws a horizontal dashed line at this variance level.
        Common values are 0.90, 0.95, 0.99. Default is 0.95.

    Attributes
    ----------
    cumulative_variance : np.ndarray
        Cumulative sum of explained variance ratios.

    Examples
    --------
    **Example 1: PCA (simplest case)**

    >>> from sklearn.decomposition import PCA
    >>> pca = PCA(n_components=10)
    >>> pca.fit(X)
    >>> plot = ExplainedVariancePlot(pca.explained_variance_ratio_)
    >>> plot.show(title="PCA Explained Variance")

    **Example 2: PLS - Now just as simple!**

    >>> from chemotools.models import PLSRegression
    >>> from chemotools.plotting import ExplainedVariancePlot
    >>>
    >>> pls = PLSRegression(n_components=5)
    >>> pls.fit(X, y)
    >>>
    >>> # Variance ratios automatically available!
    >>> plot_x = ExplainedVariancePlot(pls.explained_x_variance_ratio_)
    >>> plot_x.show(title="PLS Explained Variance in X-space")
    >>>
    >>> plot_y = ExplainedVariancePlot(pls.explained_y_variance_ratio_)
    >>> plot_y.show(title="PLS Explained Variance in Y-space")

    **Example 3: Side-by-side PLS comparison**

    >>> import matplotlib.pyplot as plt
    >>> fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    >>>
    >>> ExplainedVariancePlot(pls.explained_x_variance_ratio_).render(ax=axes[0])
    >>> axes[0].set_title('X-space (Predictors)')
    >>>
    >>> ExplainedVariancePlot(pls.explained_y_variance_ratio_).render(ax=axes[1])
    >>> axes[1].set_title('Y-space (Response)')

    **Example 4: Custom threshold and labels**

    >>> plot = ExplainedVariancePlot(pca.explained_variance_ratio_, threshold=0.90)
    >>> plot.show(xlabel="PC Number", ylabel="Variance Explained")
    """

    def __init__(
        self,
        explained_variance_ratio: np.ndarray,
        *,
        threshold: Optional[float] = 0.95,
    ):
        # Validate input
        self.explained_variance_ratio = validate_data(
            explained_variance_ratio,
            name="explained_variance_ratio",
            ensure_2d=False,
        )

        if self.explained_variance_ratio.ndim != 1:
            raise ValueError(
                f"explained_variance_ratio must be 1D, got shape {self.explained_variance_ratio.shape}"
            )

        # Validate threshold if provided
        if threshold is not None and not (0 < threshold <= 1):
            raise ValueError(f"threshold must be between 0 and 1, got {threshold}")

        self.cumulative_variance = np.cumsum(self.explained_variance_ratio)
        self.threshold = threshold

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
        """Create and display a complete figure with the variance plot.

        Parameters
        ----------
        figsize : tuple[float, float], optional
            Figure size (width, height) in inches.
        title : str, optional
            Plot title.
        xlabel : str, optional
            Label for x-axis. Default is "Component".
        ylabel : str, optional
            Label for y-axis. Default is "Explained Variance Ratio".
        xlim : tuple[float, float], optional
            X-axis limits (min, max).
        ylim : tuple[float, float], optional
            Y-axis limits (min, max).
        **kwargs : Any
            Additional keyword arguments passed to the plotting functions.
            Can include bar plot kwargs (alpha, color, edgecolor) and
            line plot kwargs (linewidth, markersize).

        Returns
        -------
        Figure
            The matplotlib Figure object.
        """
        if xlabel is None:
            xlabel = "Component"
        if ylabel is None:
            ylabel = "Explained Variance Ratio"

        fig = super().show(
            figsize=figsize or (10, 6),
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            xlim=xlim,
            ylim=ylim,
            **kwargs,
        )

        # Add legend (BasePlot.show calls render, but render returns fig, ax)
        # Wait, BasePlot.show returns fig.
        # But BasePlot.show calls self.render.
        # If I add legend in render, it will be there.

        return fig

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
            Matplotlib axes to render on. If None, current axes are used.
        xlabel : str, optional
            Label for x-axis. Default is "Component".
        ylabel : str, optional
            Label for y-axis. Default is "Explained Variance Ratio".
        xlim : tuple[float, float], optional
            X-axis limits (min, max).
        ylim : tuple[float, float], optional
            Y-axis limits (min, max).
        **kwargs : Any
            Additional keyword arguments for plot customization.

        Returns
        -------
        Axes
            The matplotlib Axes object.
        """
        if xlabel is None:
            xlabel = "Component"
        if ylabel is None:
            ylabel = "Explained Variance Ratio"

        fig, ax = super().render(
            ax=ax,
            xlabel=xlabel,
            ylabel=ylabel,
            xlim=xlim,
            ylim=ylim,
            **kwargs,
        )

        ax.legend()

        return fig, ax

    def _render_plot(self, ax: Axes, **kwargs: Any) -> None:
        """Internal method to render the variance plot.

        Parameters
        ----------
        ax : Axes
            Matplotlib axes to render on.
        **kwargs : Any
            Additional keyword arguments for customization.
            - bar_kwargs: dict for bar plot styling
            - line_kwargs: dict for line plot styling
            - threshold_kwargs: dict for threshold line styling
        """
        n_components = len(self.explained_variance_ratio)
        components = np.arange(1, n_components + 1)

        # Extract specific kwargs for different plot elements
        bar_kwargs = kwargs.pop("bar_kwargs", {})
        line_kwargs = kwargs.pop("line_kwargs", {})
        threshold_kwargs = kwargs.pop("threshold_kwargs", {})

        # Default bar plot settings
        bar_defaults: dict[str, Any] = {
            "alpha": 0.6,
            "color": "#008BFB",
            "edgecolor": "#008BFB",
        }
        bar_defaults.update(bar_kwargs)  # type: ignore[arg-type]

        # Bar plot for individual variance
        ax.bar(
            components,
            self.explained_variance_ratio,
            label="Individual",
            **bar_defaults,  # type: ignore[arg-type]
        )

        # Default line plot settings
        line_defaults: dict[str, Any] = {
            "color": "#FF0051",
            "marker": "o",
            "linestyle": "-",
            "linewidth": 1,
            "markersize": 2,
        }
        line_defaults.update(line_kwargs)  # type: ignore[arg-type]

        # Line plot for cumulative variance
        ax.plot(
            components,
            self.cumulative_variance,
            label="Cumulative",
            **line_defaults,  # type: ignore[arg-type]
        )

        # Add threshold line if specified
        if self.threshold is not None:
            threshold_defaults: dict[str, Any] = {
                "color": "green",
                "linestyle": "--",
                "alpha": 0.5,
            }
            threshold_defaults.update(threshold_kwargs)  # type: ignore[arg-type]
            ax.axhline(
                y=self.threshold,
                label=f"{self.threshold * 100:.0f}% Threshold",
                **threshold_defaults,  # type: ignore[arg-type]
            )

        # Grid for better readability
        ax.grid(alpha=0.3, axis="y")
