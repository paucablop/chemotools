"""Scores plot for visualizing model projections and latent space."""

from typing import Optional, Any, cast
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
    add_confidence_ellipse,
)
from chemotools.plotting._styles import DATASET_COLORS


class ScoresPlot:
    """Scores plot implementing Display protocol for model inspection.

    This class creates scatter plots of model scores (projections) with
    support for multiple datasets, custom styling, and automatic validation.

    Parameters
    ----------
    scores_dict : dict[str, np.ndarray]
        Dictionary mapping dataset names to score arrays.
        Each array should have shape (n_samples, n_components).
    components : tuple[int, int], optional
        Component indices to plot (default is (0, 1) for PC1 vs PC2).
        Uses 0-based indexing (e.g., (0, 1) plots PC1 vs PC2).
    labels_dict : dict[str, np.ndarray], optional
        Dictionary mapping dataset names to sample labels for coloring by class/group.
        Creates legend entries for each unique label. DEPRECATED: Use color_by_dict instead.
    color_by_dict : dict[str, np.ndarray], optional
        Dictionary mapping dataset names to values for coloring samples.
        Can be either continuous (numeric) or categorical (strings/classes).
        - Continuous: shows colorbar (e.g., concentration, temperature)
        - Categorical: shows legend with discrete colors (e.g., class labels)
    annotations_dict : dict[str, list[str]], optional
        Dictionary mapping dataset names to labels for annotating individual points.
        If provided, each point will be labeled with its corresponding text.
    dataset_colors : dict[str, str], optional
        Dictionary mapping dataset names to colors. Only used when color_by_dict is None.
    colormap : str, optional
        Colormap name. Colorblind-friendly defaults:
        - "tab10" for categorical data (default)
        - "viridis" for continuous data
    colorbar_label : str, optional
        Label for the colorbar when using continuous coloring.
        Default is "Reference Value".
    categorical : bool, optional
        Explicitly specify whether color_by values should be treated as categorical.
        If None (default), automatically detects based on dtype and unique values.
    xlabel : str, optional
        Custom x-axis label. If None, defaults to "PC{comp1+1}".
    ylabel : str, optional
        Custom y-axis label. If None, defaults to "PC{comp2+1}".
    confidence_ellipse : bool, float, or list[str], optional
        Controls which datasets get confidence ellipses:
        - If True: adds 95% ellipse only for 'train' dataset
        - If float (e.g., 0.90, 0.95, 0.99): specifies confidence level for 'train' only
        - If list of dataset names: draws ellipses for those specific datasets
        - If False or None (default): no ellipses are drawn
        Examples: True, 0.95, ['train'], ['train', 'test']

    Raises
    ------
    ValueError
        If components tuple contains invalid component indices that exceed
        the available components in the data.

    Examples
    --------
    Basic usage:

    >>> scores = {
    ...     'train': train_scores,
    ...     'test': test_scores
    ... }
    >>> plot = ScoresPlot(scores, components=(0, 1))
    >>> fig = plot.show(title="PCA Scores")

    With custom labels:

    >>> plot = ScoresPlot(
    ...     scores,
    ...     components=(1, 2),
    ...     xlabel="Second Principal Component",
    ...     ylabel="Third Principal Component"
    ... )
    >>> fig = plot.show(title="PC2 vs PC3")

    With point annotations:

    >>> annotations = {
    ...     'train': ['Sample 1', 'Sample 2', 'Sample 3'],
    ...     'test': ['Test 1', 'Test 2']
    ... }
    >>> plot = ScoresPlot(scores, annotations_dict=annotations)
    >>> fig = plot.show(title="Annotated Scores")

    With confidence ellipses:

    >>> plot = ScoresPlot(scores, confidence_ellipse=0.95)
    >>> fig = plot.show(title="Scores with 95% Confidence Ellipse for Training")

    >>> plot = ScoresPlot(scores, confidence_ellipse=['train', 'test'])
    >>> fig = plot.show(title="Ellipses for Multiple Datasets")

    With continuous coloring by concentration:

    >>> color_by = {'train': train_concentrations}
    >>> plot = ScoresPlot(scores, color_by_dict=color_by, colormap='viridis')
    >>> fig = plot.show(title="Scores Colored by Concentration")

    With categorical coloring by class:

    >>> color_by = {'train': train_classes}
    >>> plot = ScoresPlot(scores, color_by_dict=color_by, colormap='tab10')
    >>> fig = plot.show(title="Scores Colored by Class")

    Create subplots:

    >>> fig, axes = plt.subplots(1, 2)
    >>> plot1.render(ax=axes[0])
    >>> plot2.render(ax=axes[1])

    Using xlim/ylim to zoom:

    >>> plot.show(title="Zoomed Scores", xlim=(-2, 2), ylim=(-3, 3))
    """

    def __init__(
        self,
        scores_dict: dict[str, np.ndarray],
        components: tuple[int, int] = (0, 1),
        labels_dict: Optional[dict[str, np.ndarray]] = None,
        color_by_dict: Optional[dict[str, np.ndarray]] = None,
        annotations_dict: Optional[dict[str, list[str]]] = None,
        dataset_colors: Optional[dict[str, str]] = None,
        colormap: Optional[str] = None,
        colorbar_label: str = "Reference Value",
        categorical: Optional[bool] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        confidence_ellipse: Optional[float | bool | list[str]] = None,
    ):
        self.scores_dict = scores_dict
        self.components = components

        # Handle backwards compatibility: labels_dict -> color_by_dict
        if color_by_dict is None and labels_dict is not None:
            color_by_dict = labels_dict

        self.color_by_dict = color_by_dict or {}
        self.annotations_dict = annotations_dict or {}
        self.dataset_colors = dataset_colors or DATASET_COLORS
        self.colorbar_label = colorbar_label
        self.xlabel = xlabel
        self.ylabel = ylabel

        # Determine if we have categorical or continuous coloring for each dataset
        self.is_categorical_dict = {}
        self.colormap_dict = {}

        for dataset_name, color_by in self.color_by_dict.items():
            if categorical is not None:
                # User explicitly specified
                is_cat = categorical
            else:
                # Auto-detect using utility function
                is_cat = detect_categorical(color_by)

            self.is_categorical_dict[dataset_name] = is_cat

            # Set colormap using utility function
            self.colormap_dict[dataset_name] = get_default_colormap(is_cat, colormap)

        # Handle confidence_ellipse parameter
        self.confidence_ellipse: Optional[float]
        self.ellipse_datasets: list[str]

        if confidence_ellipse is True:
            self.confidence_ellipse = 0.95  # Default to 95%
            self.ellipse_datasets = ["train"]  # Only train by default
        elif confidence_ellipse is False or confidence_ellipse is None:
            self.confidence_ellipse = None
            self.ellipse_datasets = []
        elif isinstance(confidence_ellipse, list):
            self.confidence_ellipse = 0.95  # Default confidence level
            self.ellipse_datasets = confidence_ellipse
        else:
            self.confidence_ellipse = float(confidence_ellipse)
            self.ellipse_datasets = ["train"]  # Only train by default

        # Validate components at initialization
        self._validate_components()

    def _validate_components(self) -> None:
        """Validate that component indices are valid for all datasets.

        Raises
        ------
        ValueError
            If components are invalid for any dataset in scores_dict.
        """
        if not self.scores_dict:
            raise ValueError("scores_dict cannot be empty")

        comp1, comp2 = self.components

        for dataset_name, scores in self.scores_dict.items():
            if scores.ndim != 2:
                raise ValueError(
                    f"Scores for '{dataset_name}' must be 2D array, "
                    f"got shape {scores.shape}"
                )

            n_components = scores.shape[1]

            if comp1 < 0 or comp1 >= n_components:
                raise ValueError(
                    f"Component index {comp1} is invalid for dataset '{dataset_name}' "
                    f"which has {n_components} components (valid range: 0-{n_components - 1})"
                )

            if comp2 < 0 or comp2 >= n_components:
                raise ValueError(
                    f"Component index {comp2} is invalid for dataset '{dataset_name}' "
                    f"which has {n_components} components (valid range: 0-{n_components - 1})"
                )

            if comp1 == comp2:
                raise ValueError(
                    f"Component indices must be different, got both as {comp1}"
                )

    def show(
        self,
        figsize: Optional[tuple[float, float]] = None,
        title: Optional[str] = None,
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
        xlim : tuple[float, float], optional
            X-axis limits as (xmin, xmax).
        ylim : tuple[float, float], optional
            Y-axis limits as (ymin, ymax).
        **kwargs : Any
            Additional keyword arguments split into:
            - Figure setup kwargs (subplot_kw, gridspec_kw) passed to setup_figure
            - Plot kwargs (alpha, s, marker, etc.) passed to ax.scatter()

            Common plot kwargs:
            - alpha : float, optional (default: 0.7)
            - s : float, optional (default: 50) - marker size
            - marker : str, optional (default: 'o')
            - edgecolors : str, optional
            - linewidths : float, optional

        Returns
        -------
        Figure
            The matplotlib Figure object containing the plot.

        Examples
        --------
        Basic plot with custom size:

        >>> plot.show(figsize=(10, 10), title="PCA Scores")

        Zoom into a region:

        >>> plot.show(xlim=(-5, 5), ylim=(-3, 3))

        Custom styling:

        >>> plot.show(alpha=0.8, s=100, edgecolors='black', linewidths=0.5)
        """
        # Separate kwargs for setup_figure vs scatter
        figure_kwargs_keys = {"subplot_kw", "gridspec_kw", "sharex", "sharey"}
        figure_kwargs = {k: v for k, v in kwargs.items() if k in figure_kwargs_keys}
        plot_kwargs = {k: v for k, v in kwargs.items() if k not in figure_kwargs_keys}

        comp1, comp2 = self.components

        # Determine axis labels
        xlabel = self.xlabel if self.xlabel is not None else f"PC{comp1 + 1}"
        ylabel = self.ylabel if self.ylabel is not None else f"PC{comp2 + 1}"

        # Use setup_figure utility for consistent styling
        fig, ax = setup_figure(
            figsize=figsize or (8, 8),
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            **figure_kwargs,
        )

        self._render_plot(ax, **plot_kwargs)

        # Apply axis limits
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)

        # Add legend or colorbar
        has_continuous = any(
            not self.is_categorical_dict.get(name, True)
            for name in self.color_by_dict.keys()
        )

        if has_continuous:
            # Add colorbar for continuous data
            # Get the first continuous dataset for colorbar
            for dataset_name, color_by in self.color_by_dict.items():
                if not self.is_categorical_dict.get(dataset_name, True):
                    add_colorbar(
                        ax,
                        color_by,
                        self.colormap_dict[dataset_name],
                        self.colorbar_label,
                    )
                    break  # Only one colorbar

        # Always show legend (for datasets or categories)
        if self.color_by_dict or not has_continuous:
            ax.legend()

        plt.tight_layout()
        return fig

    def render(
        self,
        ax: Optional[Axes] = None,
        xlim: Optional[tuple[float, float]] = None,
        ylim: Optional[tuple[float, float]] = None,
        **kwargs: Any,
    ) -> tuple[Figure, Axes]:
        """Render the plot on the given axes or create new ones.

        Parameters
        ----------
        ax : Axes, optional
            Matplotlib axes to plot on. If None, creates new figure and axes.
        xlim : tuple[float, float], optional
            X-axis limits as (xmin, xmax).
        ylim : tuple[float, float], optional
            Y-axis limits as (ymin, ymax).
        **kwargs : Any
            Additional keyword arguments passed to the scatter function.

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
        >>> fig, ax = plot.render(ax=axes[0, 0])

        Create new figure:

        >>> fig, ax = plot.render()
        >>> ax.set_xlabel("Custom label")
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
        else:
            figure = ax.get_figure()
            if figure is None:
                raise ValueError("Axes object has no associated figure")
            fig = cast(Figure, figure)

        self._render_plot(ax, **kwargs)

        # Apply axis limits
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)

        return fig, ax

    def _render_plot(self, ax: Axes, **kwargs: Any) -> None:
        """Internal method to render the scores plot."""
        comp1, comp2 = self.components
        alpha = kwargs.pop("alpha", 0.7)
        s = kwargs.pop("s", 50)  # marker size

        for dataset_name, scores in self.scores_dict.items():
            color_by = self.color_by_dict.get(dataset_name, None)
            annotations = self.annotations_dict.get(dataset_name, None)

            if color_by is None:
                # No color_by - use dataset colors
                dataset_color = self.dataset_colors.get(dataset_name, None)
                ax.scatter(
                    scores[:, comp1],
                    scores[:, comp2],
                    c=dataset_color,
                    label=dataset_name.capitalize(),
                    alpha=alpha,
                    s=s,
                    **kwargs,
                )
            elif self.is_categorical_dict.get(dataset_name, True):
                # Categorical coloring
                colors = get_colors_from_labels(
                    color_by, self.colormap_dict[dataset_name]
                )
                unique_values = np.unique(color_by)

                # Plot each category
                for value in unique_values:
                    mask = color_by == value
                    indices = np.where(mask)[0]

                    for idx in indices:
                        # Use label only for first sample of each category
                        category_label: Optional[str] = (
                            f"{dataset_name.capitalize()} - {value}"
                            if idx == indices[0]
                            else None
                        )
                        ax.scatter(
                            scores[idx, comp1],
                            scores[idx, comp2],
                            color=colors[idx],
                            label=category_label,
                            alpha=alpha,
                            s=s,
                            **kwargs,
                        )
            else:
                # Continuous coloring
                from matplotlib import cm
                import matplotlib.colors as mcolors

                norm = mcolors.Normalize(vmin=color_by.min(), vmax=color_by.max())
                cmap = cm.get_cmap(self.colormap_dict[dataset_name])

                ax.scatter(
                    scores[:, comp1],
                    scores[:, comp2],
                    c=color_by,
                    cmap=cmap,
                    norm=norm,
                    label=dataset_name.capitalize(),
                    alpha=alpha,
                    s=s,
                    **kwargs,
                )

            # Add point annotations if provided
            if annotations is not None:
                annotate_points(
                    ax,
                    scores[:, comp1],
                    scores[:, comp2],
                    annotations,
                    fontsize=8,
                    xytext=(5, 5),
                    textcoords="offset points",
                )

            # Add confidence ellipse if requested for this dataset
            if (
                self.confidence_ellipse is not None
                and dataset_name in self.ellipse_datasets
            ):
                # Determine ellipse color - use dataset color or first continuous color
                if color_by is None:
                    ellipse_color = self.dataset_colors.get(dataset_name, None)
                elif not self.is_categorical_dict.get(dataset_name, True):
                    # For continuous, use a neutral color
                    ellipse_color = "gray"
                else:
                    # For categorical, use dataset color
                    ellipse_color = self.dataset_colors.get(dataset_name, None)

                # Draw ellipse for this dataset
                add_confidence_ellipse(
                    ax,
                    scores[:, comp1],
                    scores[:, comp2],
                    confidence=self.confidence_ellipse,
                    edgecolor=ellipse_color,
                    linewidth=2,
                    linestyle="--",
                    alpha=0.5,
                )
