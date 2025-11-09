"""Predicted vs Actual plot for regression model evaluation."""

from typing import Optional, Any, Union
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from chemotools.plotting._utils import (
    setup_figure,
    get_colors_from_labels,
    detect_categorical,
    get_default_colormap,
    add_colorbar,
    apply_limits,
)


class PredictedVsActualPlot:
    """Scatter plot of predicted vs actual values to assess regression fit.

    This class creates scatter plots comparing predicted values against actual
    (true) values with an ideal prediction line (y=x). Useful for visually
    assessing model accuracy and detecting systematic errors or bias.

    Parameters
    ----------
    y_true : np.ndarray
        True (actual) y values with shape (n_samples,) for univariate or
        (n_samples, n_targets) for multivariate regression.
    y_pred : np.ndarray
        Predicted y values with same shape as y_true.
    target_index : int, optional
        For multivariate predictions, which target to plot (default: 0).
        Ignored if y_true/y_pred are 1D.
    color_by : np.ndarray, optional
        Values for coloring samples. Can be either:
        - Continuous (numeric): shows colorbar
        - Categorical (strings/classes): shows legend with discrete colors
    label : str, optional
        Legend label for this dataset (default: None).
    color : str, optional
        Color for all points when color_by is None (default: auto-assigned).
    colormap : str, optional
        Colormap name. Colorblind-friendly defaults:
        - "tab10" for categorical data
        - "viridis" for continuous data
    add_ideal_line : bool, optional
        Whether to add diagonal y=x line showing ideal predictions (default: True).

    Raises
    ------
    ValueError
        If y_true and y_pred have mismatched shapes.

    Examples
    --------
    **Basic predicted vs actual plot:**

    >>> plot = PredictedVsActualPlot(y_true, y_pred)
    >>> fig = plot.show(title="Model Performance")

    **With categorical coloring (e.g., by batch):**

    >>> batches = np.array(['A', 'B', 'A', 'B', ...])
    >>> plot = PredictedVsActualPlot(y_true, y_pred, color_by=batches)
    >>> fig = plot.show(title="Predictions by Batch")

    **Multiple models compared:**

    >>> fig, ax = plt.subplots()
    >>> PredictedVsActualPlot(y_true, y_pred_model1, label="Model 1", color="blue").render(ax)
    >>> PredictedVsActualPlot(y_true, y_pred_model2, label="Model 2", color="red").render(ax)
    >>> ax.legend()
    >>> plt.show()

    **Multivariate regression:**

    >>> fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    >>> for i in range(3):
    ...     PredictedVsActualPlot(y_true, y_pred, target_index=i).render(axes[i])
    ...     axes[i].set_title(f"Target {i+1}")
    >>> plt.tight_layout()
    >>> plt.show()
    """

    def __init__(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        *,
        target_index: int = 0,
        color_by: Optional[np.ndarray] = None,
        label: Optional[str] = None,
        color: Optional[str] = None,
        colormap: Optional[str] = None,
        add_ideal_line: bool = True,
    ):
        self.y_true = np.asarray(y_true)
        self.y_pred = np.asarray(y_pred)
        self.target_index = target_index
        self.color_by = color_by
        self.label = label
        self.color = color
        self.add_ideal_line = add_ideal_line

        # Validate inputs
        self._validate_inputs()

        # Extract the specific target if multivariate
        if self.y_true.ndim == 2:
            if target_index >= self.y_true.shape[1]:
                raise ValueError(
                    f"target_index {target_index} is out of bounds for "
                    f"y_true with {self.y_true.shape[1]} targets"
                )
            self.y_true_1d = self.y_true[:, target_index]
            self.y_pred_1d = self.y_pred[:, target_index]
        elif self.y_true.ndim == 1:
            self.y_true_1d = self.y_true
            self.y_pred_1d = self.y_pred
        else:
            raise ValueError("y_true and y_pred must be 1D or 2D arrays")

        # Detect if color_by is categorical
        self.is_categorical = (
            detect_categorical(color_by) if color_by is not None else False
        )

        # Get colormap
        self.colormap = get_default_colormap(self.is_categorical, colormap)

    def _validate_inputs(self) -> None:
        """Validate y_true and y_pred arrays."""
        if self.y_true.shape != self.y_pred.shape:
            raise ValueError(
                f"y_true and y_pred must have same shape. "
                f"Got y_true: {self.y_true.shape}, y_pred: {self.y_pred.shape}"
            )
        if self.y_true.size == 0:
            raise ValueError("y_true and y_pred arrays cannot be empty")

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
        """Create and display the predicted vs actual plot.

        Parameters
        ----------
        title : str, optional
            Plot title (default: "Predicted vs Actual").
        xlabel : str, optional
            X-axis label (default: "Actual").
        ylabel : str, optional
            Y-axis label (default: "Predicted").
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
        # Auto-generate labels if not provided
        if xlabel is None:
            xlabel = "Actual"
        if ylabel is None:
            ylabel = "Predicted"
        if title is None:
            if self.y_true.ndim == 2:
                title = f"Predicted vs Actual (Target {self.target_index + 1})"
            else:
                title = "Predicted vs Actual"

        # Use setup_figure utility for consistent styling
        fig, ax = setup_figure(
            figsize=figsize or (8, 8),
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
        )

        # Render the actual plot
        self.render(ax)

        # Apply axis limits
        apply_limits(ax, xlim=xlim, ylim=ylim)

        # Add grid
        ax.grid(alpha=0.3, linestyle="--")

        plt.tight_layout()
        return fig

    def render(self, ax: Axes) -> None:
        """Render the plot on existing axes.

        Parameters
        ----------
        ax : Axes
            Matplotlib axes to render the plot on.
        """
        # Prepare colors
        colors: Union[np.ndarray, str, None]
        if self.color_by is not None:
            if self.is_categorical:
                colors = get_colors_from_labels(self.color_by, colormap=self.colormap)
            else:
                # For continuous data, use the values directly for colormap
                colors = self.color_by
        elif self.color is not None:
            colors = self.color
        else:
            colors = None

        # Create scatter plot
        ax.scatter(
            self.y_true_1d,
            self.y_pred_1d,
            c=colors,
            alpha=0.7,
            s=50,
            label=self.label,
            edgecolors="none",
        )

        # Add ideal prediction line (y=x)
        if self.add_ideal_line:
            lims = [
                min(self.y_true_1d.min(), self.y_pred_1d.min()),
                max(self.y_true_1d.max(), self.y_pred_1d.max()),
            ]
            ax.plot(
                lims, lims, "k--", alpha=0.5, zorder=0, label="Ideal", linewidth=1.5
            )

        # Add colorbar for continuous color_by
        if self.color_by is not None and not self.is_categorical:
            add_colorbar(ax, self.color_by, self.colormap, label="Color By")

        # Add legend for categorical color_by
        if self.color_by is not None and self.is_categorical:
            from matplotlib import cm
            import matplotlib.patches as mpatches

            unique_labels = np.unique(self.color_by)
            if (
                len(unique_labels) <= 10
            ):  # Only show legend for reasonable number of categories
                cmap_obj = cm.get_cmap(self.colormap)
                patches = [
                    mpatches.Patch(
                        color=cmap_obj(i / len(unique_labels)), label=f"{lbl}"
                    )
                    for i, lbl in enumerate(unique_labels)
                ]
                ax.legend(handles=patches, loc="best", framealpha=0.9)
        elif self.label is not None:
            ax.legend(loc="best", framealpha=0.9)
