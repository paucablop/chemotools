"""Example integration of Display protocol with inspector module.

This demonstrates how to create plot classes that implement the Display
protocol for use with your inspector classes.
"""

from typing import Optional, Any, cast
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from chemotools.plotting._utilities import setup_figure
from chemotools.plotting._scores import ScoresPlot


class LoadingsPlot:
    """Loadings plot implementing Display protocol for model inspection.

    This class creates line or stem plots of model loadings (feature weights).

    Parameters
    ----------
    loadings : np.ndarray
        Loadings array with shape (n_features, n_components).
    feature_names : list[str], optional
        Names of features for x-axis labeling.
    component : int, optional
        Which component to plot (default is 0 for PC1).
    plot_type : {'line', 'stem'}, optional
        Type of plot (default is 'stem').

    Examples
    --------
    >>> plot = LoadingsPlot(loadings, feature_names=wavelengths, component=0)
    >>> fig = plot.show(title="PC1 Loadings")
    """

    def __init__(
        self,
        loadings: np.ndarray,
        feature_names: Optional[list[str]] = None,
        component: int = 0,
        plot_type: str = "stem",
    ):
        self.loadings = loadings
        self.feature_names = feature_names
        self.component = component
        self.plot_type = plot_type

    def show(
        self,
        figsize: Optional[tuple[float, float]] = None,
        title: Optional[str] = None,
        **kwargs: Any,
    ) -> Figure:
        """Create and return a complete figure with the loadings plot."""
        # Use setup_figure utility for consistent styling
        fig, ax = setup_figure(
            figsize=figsize or (12, 4),
            title=title or f"PC{self.component + 1} Loadings",
            xlabel="Feature",
            ylabel="Loading",
        )

        self._render_plot(ax, **kwargs)
        ax.axhline(y=0, color="k", linestyle="-", linewidth=0.5)

        plt.tight_layout()
        return fig

    def render(
        self,
        ax: Optional[Axes] = None,
        **kwargs: Any,
    ) -> tuple[Figure, Axes]:
        """Render the plot on the given axes or create new ones."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 4))
        else:
            figure = ax.get_figure()
            if figure is None:
                raise ValueError("Axes object has no associated figure")
            fig = cast(Figure, figure)

        self._render_plot(ax, **kwargs)
        return fig, ax

    def _render_plot(self, ax: Axes, **kwargs: Any) -> None:
        """Internal method to render the loadings plot."""
        loadings = self.loadings[:, self.component]

        if self.feature_names is not None:
            x: Any = self.feature_names
        else:
            x = np.arange(len(loadings))

        if self.plot_type == "stem":
            ax.stem(x, loadings, **kwargs)
        elif self.plot_type == "line":
            ax.plot(x, loadings, **kwargs)
        else:
            raise ValueError(f"Unknown plot_type: {self.plot_type}")


class ExplainedVariancePlot:
    """Explained variance plot implementing Display protocol.

    Shows cumulative and individual explained variance by component.

    Parameters
    ----------
    explained_variance_ratio : np.ndarray
        Array of explained variance ratios for each component.

    Examples
    --------
    >>> plot = ExplainedVariancePlot(model.explained_variance_ratio_)
    >>> fig = plot.show(title="Explained Variance")
    """

    def __init__(self, explained_variance_ratio: np.ndarray):
        self.explained_variance_ratio = explained_variance_ratio
        self.cumulative_variance = np.cumsum(explained_variance_ratio)

    def show(
        self,
        figsize: Optional[tuple[float, float]] = None,
        title: Optional[str] = None,
        **kwargs: Any,
    ) -> Figure:
        """Create and return a complete figure with the variance plot."""
        # Use setup_figure utility for consistent styling
        fig, ax = setup_figure(
            figsize=figsize or (10, 6),
            title=title or "Explained Variance",
            xlabel="Component",
            ylabel="Explained Variance Ratio",
        )

        self._render_plot(ax, **kwargs)
        ax.legend()

        plt.tight_layout()
        return fig

    def render(
        self,
        ax: Optional[Axes] = None,
        **kwargs: Any,
    ) -> tuple[Figure, Axes]:
        """Render the plot on the given axes or create new ones."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        else:
            figure = ax.get_figure()
            if figure is None:
                raise ValueError("Axes object has no associated figure")
            fig = cast(Figure, figure)

        self._render_plot(ax, **kwargs)
        return fig, ax

    def _render_plot(self, ax: Axes, **kwargs: Any) -> None:
        """Internal method to render the variance plot."""
        n_components = len(self.explained_variance_ratio)
        components = np.arange(1, n_components + 1)

        # Bar plot for individual variance
        ax.bar(
            components,
            self.explained_variance_ratio,
            alpha=0.6,
            label="Individual",
            **kwargs,
        )

        # Line plot for cumulative variance
        ax.plot(
            components,
            self.cumulative_variance,
            color="red",
            marker="o",
            linestyle="-",
            linewidth=2,
            markersize=6,
            label="Cumulative",
        )

        # Add horizontal line at 95% explained variance
        ax.axhline(y=0.95, color="green", linestyle="--", alpha=0.5, label="95%")


# Example usage with inspector
def example_integration():
    """Example showing how to use Display plots with inspector module."""
    from chemotools.plotting import is_displayable

    # Simulate some data
    train_scores = np.random.randn(100, 5)
    test_scores = np.random.randn(50, 5)
    loadings = np.random.randn(200, 5)
    explained_var = np.array([0.45, 0.25, 0.15, 0.10, 0.05])

    # Create plot objects
    scores_plot = ScoresPlot(
        {"train": train_scores, "test": test_scores}, components=(0, 1)
    )
    loadings_plot = LoadingsPlot(loadings, component=0)
    variance_plot = ExplainedVariancePlot(explained_var)

    # Verify they all implement Display
    assert is_displayable(scores_plot)
    assert is_displayable(loadings_plot)
    assert is_displayable(variance_plot)

    # Create a dashboard with all plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    scores_plot.render(ax=axes[0, 0])
    axes[0, 0].set_title("Scores Plot")

    loadings_plot.render(ax=axes[0, 1])
    axes[0, 1].set_title("Loadings Plot")

    variance_plot.render(ax=axes[1, 0])
    axes[1, 0].set_title("Explained Variance")

    axes[1, 1].axis("off")  # Empty subplot

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    fig = example_integration()
    plt.show()
