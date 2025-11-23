"""Scores plot for visualizing model projections and latent space."""

from typing import Optional, Any
import numpy as np
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from chemotools.plotting._base import BasePlot, ColoringMixin
from chemotools.plotting._utils import (
    get_colors_from_labels,
    annotate_points,
    add_confidence_ellipse,
)


class ScoresPlot(BasePlot, ColoringMixin):
    """Simple, composable scores plot for a single dataset.

    This class creates scatter plots of model scores (projections) for one dataset.
    Multiple datasets can be overlaid by using the render() method on shared axes.

    Parameters
    ----------
    scores : np.ndarray
        Score array with shape (n_samples, n_components).
    components : tuple[int, int], optional
        Component indices to plot (default is (0, 1) for PC1 vs PC2).
        Uses 0-based indexing (e.g., (0, 1) plots PC1 vs PC2).
    color_by : np.ndarray, optional
        Values for coloring samples. Can be either:
        - Continuous (numeric): shows colorbar (e.g., concentration, temperature)
        - Categorical (strings/classes): shows legend with discrete colors
    annotations : list[str], optional
        Labels for annotating individual points.
    label : str, optional
        Legend label for this dataset (default: "Data").
    color : str, optional
        Color for all points when color_by is None (default: auto-assigned).
    colormap : str, optional
        Colormap name. Colorblind-friendly defaults:
        - "tab10" for categorical data
        - "viridis" for continuous data
    confidence_ellipse : bool or float, optional
        Whether to draw a confidence ellipse around the data.
        - If True: draws 95% confidence ellipse
        - If float: draws ellipse at specified confidence level (e.g., 0.90, 0.99)
        - If False or None: no ellipse (default)

    Raises
    ------
    ValueError
        If components tuple contains invalid component indices.

    Examples
    --------
    **Simple single dataset plot:**

    >>> plot = ScoresPlot(train_scores)
    >>> fig = plot.show(title="PCA Scores")

    **Multiple datasets composed together:**

    >>> fig, ax = plt.subplots()
    >>> ScoresPlot(train_scores, label="Train", color="blue").render(ax)
    >>> ScoresPlot(test_scores, label="Test", color="red").render(ax)
    >>> ax.legend()
    >>> plt.show()

    **With categorical coloring:**

    >>> plot = ScoresPlot(train_scores, color_by=train_classes)
    >>> fig = plot.show(title="Scores by Class")

    **With continuous coloring:**

    >>> plot = ScoresPlot(train_scores, color_by=concentrations, colormap='viridis')
    >>> fig = plot.show(title="Scores by Concentration")

    **Custom components and labels:**

    >>> plot = ScoresPlot(scores, components=(1, 2))
    >>> fig = plot.show(
    ...     title="PC2 vs PC3",
    ...     xlabel="Second Component",
    ...     ylabel="Third Component"
    ... )

    **With annotations:**

    >>> annotations = [f"S{i}" if i in outliers else "" for i in range(len(scores))]
    >>> plot = ScoresPlot(scores, annotations=annotations)
    >>> fig = plot.show(title="Annotated Scores")

    **With confidence ellipse:**

    >>> plot = ScoresPlot(train_scores, confidence_ellipse=True)
    >>> fig = plot.show(title="Scores with 95% Confidence Ellipse")

    >>> plot = ScoresPlot(train_scores, confidence_ellipse=0.99, color="blue")
    >>> fig = plot.show(title="Scores with 99% Confidence Ellipse")
    """

    def __init__(
        self,
        scores: np.ndarray,
        *,
        components: tuple[int, int] = (0, 1),
        color_by: Optional[np.ndarray] = None,
        annotations: Optional[list[str]] = None,
        label: str = "Data",
        color: Optional[str] = None,
        colormap: Optional[str] = None,
        confidence_ellipse: Optional[bool | float] = None,
    ):
        self.scores = scores
        self.components = components
        self.annotations = annotations
        self.label = label
        self.color = color

        # Process confidence ellipse parameter
        self.confidence_level: Optional[float]
        if confidence_ellipse is True:
            self.confidence_level = 0.95
        elif isinstance(confidence_ellipse, (int, float)) and confidence_ellipse:
            self.confidence_level = float(confidence_ellipse)
        else:
            self.confidence_level = None

        # Validate inputs
        self._validate_scores()
        self._validate_components()

        # Initialize coloring
        self._init_coloring(color_by, colormap)

    def _validate_scores(self) -> None:
        """Validate that scores array has correct shape."""
        if self.scores.ndim != 2:
            raise ValueError(f"Scores must be 2D array, got shape {self.scores.shape}")

    def _validate_components(self) -> None:
        """Validate that component indices are valid.

        Raises
        ------
        ValueError
            If components are invalid for the scores array.
        """
        comp1, comp2 = self.components
        n_components = self.scores.shape[1]

        if comp1 < 0 or comp1 >= n_components:
            raise ValueError(
                f"Component index {comp1} is invalid. "
                f"Valid range: 0-{n_components - 1} (have {n_components} components)"
            )

        if comp2 < 0 or comp2 >= n_components:
            raise ValueError(
                f"Component index {comp2} is invalid. "
                f"Valid range: 0-{n_components - 1} (have {n_components} components)"
            )

        if comp1 == comp2:
            raise ValueError(
                f"Component indices must be different, got both as {comp1}"
            )

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
        """Create and return a complete figure with the scores plot.

        Parameters
        ----------
        figsize : tuple[float, float], optional
            Figure size as (width, height) in inches. Default is (8, 8).
        title : str, optional
            Title for the plot.
        xlabel : str, optional
            Custom x-axis label. If None, defaults to "PC{comp1+1}".
        ylabel : str, optional
            Custom y-axis label. If None, defaults to "PC{comp2+1}".
        xlim : tuple[float, float], optional
            X-axis limits as (xmin, xmax).
        ylim : tuple[float, float], optional
            Y-axis limits as (ymin, ymax).
        **kwargs : Any
            Additional keyword arguments passed to ax.scatter().
            Common options: alpha, s (marker size), marker, edgecolors, linewidths.

        Returns
        -------
        Figure
            The matplotlib Figure object containing the plot.

        Examples
        --------
        >>> plot.show(title="PCA Scores")
        >>> plot.show(figsize=(10, 10), xlim=(-5, 5), ylim=(-3, 3))
        >>> plot.show(alpha=0.8, s=100, edgecolors='black')
        >>> plot.show(title="Custom", xlabel="First PC", ylabel="Second PC")
        """
        comp1, comp2 = self.components

        # Determine default axis labels if not provided
        if xlabel is None:
            xlabel = f"PC{comp1 + 1}"
        if ylabel is None:
            ylabel = f"PC{comp2 + 1}"

        return super().show(
            figsize=figsize or (8, 8),
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

        Use this method to compose multiple plots on the same axes.

        Parameters
        ----------
        ax : Axes, optional
            Matplotlib axes to plot on. If None, creates new figure and axes.
        xlabel : str, optional
            Custom x-axis label. If None, uses existing label or defaults to "PC{comp1+1}".
        ylabel : str, optional
            Custom y-axis label. If None, uses existing label or defaults to "PC{comp2+1}".
        xlim : tuple[float, float], optional
            X-axis limits as (xmin, xmax).
        ylim : tuple[float, float], optional
            Y-axis limits as (ymin, ymax).
        **kwargs : Any
            Additional keyword arguments passed to ax.scatter().

        Returns
        -------
        fig : Figure
            The matplotlib Figure object.
        ax : Axes
            The matplotlib Axes object with the rendered plot.

        Examples
        --------
        Compose multiple datasets:

        >>> fig, ax = plt.subplots()
        >>> ScoresPlot(train_scores, label="Train").render(ax)
        >>> ScoresPlot(test_scores, label="Test").render(ax)
        >>> ax.set_xlabel("PC1")
        >>> ax.set_ylabel("PC2")
        >>> ax.legend()
        >>> plt.show()
        """
        comp1, comp2 = self.components

        # Determine default axis labels if not provided
        if xlabel is None:
            xlabel = f"PC{comp1 + 1}"
        if ylabel is None:
            ylabel = f"PC{comp2 + 1}"

        fig, ax = super().render(
            ax=ax,
            xlabel=xlabel,
            ylabel=ylabel,
            xlim=xlim,
            ylim=ylim,
            **kwargs,
        )

        # Add colorbar for continuous data
        self._add_colorbar_if_needed(ax)

        # Add legend
        ax.legend()

        return fig, ax

    def _render_plot(self, ax: Axes, **kwargs: Any) -> None:
        """Internal method to render the scores plot on given axes."""
        comp1, comp2 = self.components
        alpha = kwargs.pop("alpha", 0.7)
        s = kwargs.pop("s", 50)

        x = self.scores[:, comp1]
        y = self.scores[:, comp2]

        if self.color_by is None:
            # Simple scatter with single color
            ax.scatter(
                x,
                y,
                c=self.color,
                label=self.label,
                alpha=alpha,
                s=s,
                **kwargs,
            )
        elif self.is_categorical:
            # Categorical coloring
            assert self.colormap is not None  # colormap is set by get_default_colormap
            colors = get_colors_from_labels(self.color_by, self.colormap)
            unique_values = np.unique(self.color_by)

            # Plot each category
            for value in unique_values:
                mask = self.color_by == value
                ax.scatter(
                    x[mask],
                    y[mask],
                    color=colors[mask][0],  # All same color for this category
                    label=f"{self.label} - {value}",
                    alpha=alpha,
                    s=s,
                    **kwargs,
                )
        else:
            # Continuous coloring
            import matplotlib as mpl
            import matplotlib.colors as mcolors

            norm = mcolors.Normalize(vmin=self.color_by.min(), vmax=self.color_by.max())
            # Ensure we have a valid colormap (should not be None here, but be defensive)
            colormap_name = self.colormap if self.colormap is not None else "viridis"
            cmap = mpl.colormaps.get_cmap(colormap_name)

            ax.scatter(
                x,
                y,
                c=self.color_by,
                cmap=cmap,
                norm=norm,
                label=self.label,
                alpha=alpha,
                s=s,
                **kwargs,
            )

        # Add confidence ellipse if requested
        if self.confidence_level is not None:
            # Default to black if no color specified
            edgecolor = self.color if self.color is not None else "black"
            add_confidence_ellipse(
                ax,
                x,
                y,
                confidence=self.confidence_level,
                edgecolor=edgecolor,
                linewidth=1,
                linestyle="--",
                alpha=0.8,
            )

        # Add point annotations if provided
        if self.annotations is not None:
            annotate_points(
                ax,
                x,
                y,
                self.annotations,
                fontsize=8,
                xytext=(5, 5),
                textcoords="offset points",
            )
