"""Scores plot for visualizing model projections and latent space."""

from typing import Literal, Optional, Any, Tuple
import numpy as np
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from chemotools.plotting._base import BasePlot, ColoringMixin
from chemotools.plotting._utils import (
    annotate_points,
    add_confidence_ellipse,
    validate_data,
    scatter_with_colormap,
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

    color_mode : {"continuous", "categorical"}, optional
        Explicitly specify coloring mode. If None (default), automatically
        detects based on dtype and unique values of color_by.
    colorbar_label : str, optional
        Label for the colorbar when using continuous coloring.
        Default is "Value". Only applies when color_by is continuous.

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
        color_mode: Optional[Literal["continuous", "categorical"]] = None,
        colorbar_label: str = "Value",
    ):
        self.scores = validate_data(scores, name="scores", ensure_2d=True)
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
        self._validate_components()

        if color_by is not None:
            color_by = validate_data(
                color_by, name="color_by", ensure_2d=False, numeric=False
            )

        # Initialize coloring
        self._init_coloring(
            color_by, colormap, color_mode=color_mode, colorbar_label=colorbar_label
        )

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

    def _get_default_labels(self) -> dict[str, str]:
        comp1, comp2 = self.components
        return {
            "xlabel": f"PC{comp1 + 1}",
            "ylabel": f"PC{comp2 + 1}",
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
        """Create and return a complete figure with the scores plot.

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

        # Add legend only if there are labeled artists
        handles, _ = ax.get_legend_handles_labels()
        if handles:
            ax.legend()

        return fig, ax

    def _render_plot(self, ax: Axes, **kwargs: Any) -> None:
        """Internal method to render the scores plot on given axes."""
        comp1, comp2 = self.components
        alpha = kwargs.pop("alpha", 0.7)
        s = kwargs.pop("s", 50)

        x = self.scores[:, comp1]
        y = self.scores[:, comp2]

        scatter_with_colormap(
            ax,
            x,
            y,
            color_by=self.color_by,
            is_categorical=self.is_categorical,
            colormap=self.colormap,
            color=self.color,
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
