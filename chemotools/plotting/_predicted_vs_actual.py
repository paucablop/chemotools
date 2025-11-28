"""Predicted vs Actual plot for regression model evaluation."""

from typing import Literal, Optional, Any
import numpy as np
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from chemotools.plotting._base import BasePlot, ColoringMixin
from chemotools.plotting._utils import validate_data, scatter_with_colormap


class PredictedVsActualPlot(BasePlot, ColoringMixin):
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
    marker : str, optional
        Marker style for scatter points (default: "o"). Examples: "o", "s", "^", "v", "D".
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
        marker: str = "o",
        add_ideal_line: bool = True,
        color_mode: Optional[Literal["continuous", "categorical"]] = None,
    ):
        self.y_true = validate_data(y_true, name="y_true", ensure_2d=False)
        self.y_pred = validate_data(y_pred, name="y_pred", ensure_2d=False)
        self.target_index = target_index
        self.label = label
        self.color = color
        self.marker = marker
        self.add_ideal_line = add_ideal_line

        # Validate shapes match
        if self.y_true.shape != self.y_pred.shape:
            raise ValueError(
                f"y_true and y_pred must have same shape, got {self.y_true.shape} and {self.y_pred.shape}"
            )

        if color_by is not None:
            color_by = validate_data(
                color_by, name="color_by", ensure_2d=False, numeric=False
            )

        # Initialize coloring
        self._init_coloring(color_by, colormap, color_mode=color_mode)
        self.y_true = validate_data(y_true, name="y_true", ensure_2d=False)
        self.y_pred = validate_data(y_pred, name="y_pred", ensure_2d=False)
        self.target_index = target_index
        self.label = label
        self.color = color
        self.marker = marker
        self.add_ideal_line = add_ideal_line

        # Validate inputs
        self._validate_inputs()

        if color_by is not None:
            color_by = validate_data(
                color_by, name="color_by", ensure_2d=False, numeric=False
            )

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

        # Initialize coloring
        self._init_coloring(color_by, colormap, colorbar_label="Color By")

    def _validate_inputs(self) -> None:
        """Validate y_true and y_pred arrays."""
        if self.y_true.shape != self.y_pred.shape:
            raise ValueError(
                f"y_true and y_pred must have same shape. "
                f"Got y_true: {self.y_true.shape}, y_pred: {self.y_pred.shape}"
            )

    def _get_default_labels(self) -> dict[str, str]:
        if self.y_true.ndim == 2:
            default_title = f"Predicted vs Actual (Target {self.target_index + 1})"
        else:
            default_title = "Predicted vs Actual"

        return {
            "xlabel": "Actual",
            "ylabel": "Predicted",
            "title": default_title,
        }

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
        # Remove figsize from kwargs as it is not used in render
        kwargs.pop("figsize", None)

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

        # Add legend if categorical or if label is provided
        if self.is_categorical or self.label:
            # Only add legend if there are labeled artists
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend()

        return fig, ax

    def _render_plot(self, ax: Axes, **kwargs: Any) -> None:
        """Internal method to render the plot on given axes."""
        alpha = kwargs.pop("alpha", 0.7)
        s = kwargs.pop("s", 50)
        edgecolors = kwargs.pop("edgecolors", "none")

        # Determine colors
        scatter_with_colormap(
            ax,
            self.y_true_1d,
            self.y_pred_1d,
            color_by=self.color_by,
            is_categorical=self.is_categorical,
            colormap=self.colormap,
            color=self.color if self.color is not None else "steelblue",
            label=self.label,
            alpha=alpha,
            s=s,
            marker=self.marker,
            edgecolors=edgecolors,
            **kwargs,
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
