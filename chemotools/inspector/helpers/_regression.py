"""Regression-specific plot creation functions for inspectors.

This module contains plotting functions specific to regression models (PLS, etc.)
that are not applicable to unsupervised methods like PCA.

Each function handles both single and multi-dataset cases internally, using
plot objects from the chemotools.plotting module for consistent rendering.
"""

from __future__ import annotations
from typing import Dict, Tuple, TYPE_CHECKING, Optional, Union, Literal
import numpy as np
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from matplotlib.figure import Figure

from chemotools.plotting import (
    YResidualsPlot,
    QQPlot,
    ResidualDistributionPlot,
    PredictedVsActualPlot,
    DistancesPlot,
)
from chemotools.plotting._styles import DATASET_COLORS, DATASET_MARKERS
from chemotools.plotting._utils import annotate_points

from ..core.utils import prepare_annotations, prepare_color_values


def create_predicted_vs_actual_plot(
    datasets_data: Dict[str, Dict[str, np.ndarray]],
    color_by: Optional[Union[str, Dict[str, np.ndarray]]],
    figsize: Tuple[float, float],
    annotate_by: Optional[Union[str, Dict[str, np.ndarray]]] = None,
    color_mode: Optional[Literal["continuous", "categorical"]] = None,
) -> Figure:
    """Create predicted vs actual plot for one or multiple datasets.

    Handles both single and multi-dataset cases internally. For single dataset,
    can color by y-values. For multiple datasets, colors by dataset.

    Parameters
    ----------
    datasets_data : Dict[str, Dict[str, np.ndarray]]
        Dictionary mapping dataset names to dicts with 'y_true', 'y_pred', 'y' keys
    color_by : str or dict, optional
        Coloring specification
    figsize : Tuple[float, float]
        Figure size
    annotate_by : str or dict, optional
        Annotations for plot points.
    color_mode : str, optional
        Coloring mode ("continuous" or "categorical").

    Returns
    -------
    Figure
        Matplotlib figure with predicted vs actual plot
    """
    n_datasets = len(datasets_data)

    if n_datasets == 1:
        # Single dataset - use color_by option
        dataset_name, data = list(datasets_data.items())[0]
        y_true = data["y_true"]
        y_pred = data["y_pred"]
        X = data.get("X")

        color_values = prepare_color_values(
            color_by, dataset_name, y_true, y_true.shape[0]
        )

        fig, ax = plt.subplots(figsize=figsize)

        pred_actual_plot = PredictedVsActualPlot(
            y_true=y_true,
            y_pred=y_pred,
            color_by=color_values,
            color_mode=color_mode,
        )
        pred_actual_plot.render(ax=ax)

        # Add annotations if requested
        labels = prepare_annotations(annotate_by, dataset_name, X, y_true)
        if labels is not None:
            annotate_points(
                ax,
                y_true.ravel(),
                y_pred.ravel(),
                labels,
                fontsize=8,
                alpha=0.7,
                xytext=(3, 3),
                textcoords="offset points",
            )

        ax.set_title(
            f"Predicted vs Actual ({dataset_name})", fontsize=12, fontweight="bold"
        )
        ax.grid(alpha=0.3)
        plt.tight_layout()

        return fig

    # Multiple datasets - overlay on single plot, color by dataset
    fig, ax = plt.subplots(figsize=figsize)

    for i, (dataset_name, data) in enumerate(datasets_data.items()):
        y_true = data["y_true"]
        y_pred = data["y_pred"]
        X = data.get("X")

        color = DATASET_COLORS.get(dataset_name, "gray")
        marker = DATASET_MARKERS.get(dataset_name, "o")

        # Create predicted vs actual plot for this dataset
        # Add ideal line only for the first dataset to avoid duplicates
        pred_actual_plot = PredictedVsActualPlot(
            y_true=y_true,
            y_pred=y_pred,
            label=dataset_name.capitalize(),
            color=color,
            marker=marker,
            add_ideal_line=(i == 0),
        )
        pred_actual_plot.render(ax=ax)

        # Add annotations if requested
        labels = prepare_annotations(annotate_by, dataset_name, X, y_true)
        if labels is not None:
            annotate_points(
                ax,
                y_true.ravel(),
                y_pred.ravel(),
                labels,
                fontsize=8,
                alpha=0.7,
                xytext=(3, 3),
                textcoords="offset points",
            )

    ax.set_xlabel("Actual", fontsize=10)
    ax.set_ylabel("Predicted", fontsize=10)
    ax.set_title("Predicted vs Actual", fontsize=12, fontweight="bold")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    plt.tight_layout()

    return fig


def create_y_residual_plot(
    datasets_data: Dict[str, Dict[str, np.ndarray]],
    color_by: Optional[Union[str, Dict[str, np.ndarray]]],
    figsize: Tuple[float, float],
    annotate_by: Optional[Union[str, Dict[str, np.ndarray]]] = None,
    color_mode: Optional[Literal["continuous", "categorical"]] = None,
) -> Figure:
    """Create residual scatter plot for one or multiple datasets.

    Handles both single and multi-dataset cases internally. For single dataset,
    shows one plot with optional y-coloring. For multiple datasets, creates
    side-by-side subplots with confidence bands for each.

    Parameters
    ----------
    datasets_data : Dict[str, Dict[str, np.ndarray]]
        Dictionary mapping dataset names to dicts with 'y_true', 'y_pred', 'y' keys
    color_by : str or dict, optional
        Coloring specification
    figsize : Tuple[float, float]
        Figure size
    annotate_by : str or dict, optional
        Annotations for plot points.
    color_mode : str, optional
        Coloring mode ("continuous" or "categorical").

    Returns
    -------
    Figure
        Matplotlib figure with residual plot
    """
    n_datasets = len(datasets_data)

    if n_datasets == 1:
        # Single dataset - single plot with optional y-coloring
        dataset_name, data = list(datasets_data.items())[0]
        y_true = data["y_true"]
        y_pred = data["y_pred"]
        X = data.get("X")

        color_values = prepare_color_values(
            color_by, dataset_name, y_true, y_true.shape[0]
        )
        residuals = y_true - y_pred

        fig, ax = plt.subplots(figsize=figsize)

        residuals_plot = YResidualsPlot(
            residuals=residuals,
            x_values=y_pred,
            color_by=color_values,
            add_confidence_band=2.0,
            color_mode=color_mode,
        )
        residuals_plot.render(ax=ax)

        # Add annotations if requested
        labels = prepare_annotations(annotate_by, dataset_name, X, y_true)
        if labels is not None:
            annotate_points(
                ax,
                y_pred.ravel(),
                residuals.ravel(),
                labels,
                fontsize=8,
                alpha=0.7,
                xytext=(3, 3),
                textcoords="offset points",
            )

        ax.set_xlabel("Predicted Values", fontsize=10)
        ax.set_ylabel("Residuals", fontsize=10)
        ax.set_title(f"Residual Plot ({dataset_name})", fontsize=12, fontweight="bold")
        ax.grid(alpha=0.3)
        plt.tight_layout()

        return fig

    # Multiple datasets - side-by-side subplots
    fig, axes = plt.subplots(
        1, n_datasets, figsize=(figsize[0] * n_datasets, figsize[1])
    )

    # Ensure axes is always iterable (for single subplot it would be a single Axes)
    if n_datasets == 1:
        axes = [axes]

    for ax, (dataset_name, data) in zip(axes, datasets_data.items()):
        y_true = data["y_true"]
        y_pred = data["y_pred"]
        X = data.get("X")

        color_values = prepare_color_values(
            color_by, dataset_name, y_true, y_true.shape[0]
        )
        residuals = y_true - y_pred

        residuals_plot = YResidualsPlot(
            residuals=residuals,
            x_values=y_pred,
            color_by=color_values,
            add_confidence_band=2.0,
            color_mode=color_mode,
        )
        residuals_plot.render(ax=ax)

        # Add annotations if requested
        labels = prepare_annotations(annotate_by, dataset_name, X, y_true)
        if labels is not None:
            annotate_points(
                ax,
                y_pred.ravel(),
                residuals.ravel(),
                labels,
                fontsize=8,
                alpha=0.7,
                xytext=(3, 3),
                textcoords="offset points",
            )

        ax.set_xlabel("Predicted Values", fontsize=10)
        ax.set_ylabel("Residuals", fontsize=10)
        ax.set_title(f"{dataset_name.capitalize()}", fontsize=12, fontweight="bold")
        ax.grid(alpha=0.3)

    plt.tight_layout()
    return fig


def create_qq_plot(
    datasets_data: Dict[str, Dict[str, np.ndarray]],
    figsize: Tuple[float, float],
    confidence: float = 0.95,
) -> Figure:
    """Create Q-Q plots for one or multiple datasets.

    Handles both single and multi-dataset cases internally. For single dataset,
    shows one plot. For multiple datasets, creates side-by-side subplots.

    Parameters
    ----------
    datasets_data : Dict[str, Dict[str, np.ndarray]]
        Dictionary mapping dataset names to dicts with 'y_true', 'y_pred' keys
    figsize : Tuple[float, float]
        Figure size
    confidence : float, default=0.95
        Confidence level for the confidence band

    Returns
    -------
    Figure
        Matplotlib figure with Q-Q plot(s)
    """
    n_datasets = len(datasets_data)

    if n_datasets == 1:
        # Single dataset - single plot
        dataset_name, data = list(datasets_data.items())[0]
        y_true = data["y_true"]
        y_pred = data["y_pred"]
        residuals = y_true - y_pred

        fig, ax = plt.subplots(figsize=figsize)

        qq_plot = QQPlot(residuals=residuals, add_confidence_band=confidence)
        qq_plot.render(ax=ax)

        ax.set_title(f"Q-Q Plot ({dataset_name})", fontsize=12, fontweight="bold")
        ax.grid(alpha=0.3)
        plt.tight_layout()

        return fig

    # Multiple datasets - side-by-side subplots
    fig, axes = plt.subplots(
        1, n_datasets, figsize=(figsize[0] * n_datasets, figsize[1])
    )

    # Ensure axes is always iterable (for single subplot it would be a single Axes)
    if n_datasets == 1:
        axes = [axes]

    for ax, (dataset_name, data) in zip(axes, datasets_data.items()):
        y_true = data["y_true"]
        y_pred = data["y_pred"]
        residuals = y_true - y_pred

        qq_plot = QQPlot(residuals=residuals, add_confidence_band=confidence)
        qq_plot.render(ax=ax)

        ax.set_title(f"{dataset_name.capitalize()}", fontsize=12, fontweight="bold")
        ax.grid(alpha=0.3)

    fig.suptitle("Q-Q Plot", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()

    return fig


def create_residual_distribution_plot(
    datasets_data: Dict[str, Dict[str, np.ndarray]],
    figsize: Tuple[float, float],
) -> Figure:
    """Create residual distribution plot for one or multiple datasets.

    Handles both single and multi-dataset cases internally. For single dataset,
    shows one histogram. For multiple datasets, creates side-by-side subplots.

    Parameters
    ----------
    datasets_data : Dict[str, Dict[str, np.ndarray]]
        Dictionary mapping dataset names to dicts with 'y_true', 'y_pred' keys
    figsize : Tuple[float, float]
        Figure size

    Returns
    -------
    Figure
        Matplotlib figure with residual distribution plot(s)
    """
    n_datasets = len(datasets_data)

    if n_datasets == 1:
        # Single dataset - single histogram
        dataset_name, data = list(datasets_data.items())[0]
        y_true = data["y_true"]
        y_pred = data["y_pred"]
        residuals = y_true - y_pred

        fig, ax = plt.subplots(figsize=figsize)

        dist_plot = ResidualDistributionPlot(residuals=residuals, bins=30)
        dist_plot.render(ax=ax)

        ax.set_title(
            f"Residual Distribution ({dataset_name})", fontsize=12, fontweight="bold"
        )
        plt.tight_layout()

        return fig

    # Multiple datasets - side-by-side subplots
    fig, axes = plt.subplots(
        1, n_datasets, figsize=(figsize[0] * n_datasets, figsize[1])
    )

    # Ensure axes is always iterable (for single subplot it would be a single Axes)
    if n_datasets == 1:
        axes = [axes]

    for ax, (dataset_name, data) in zip(axes, datasets_data.items()):
        y_true = data["y_true"]
        y_pred = data["y_pred"]
        residuals = y_true - y_pred

        dist_plot = ResidualDistributionPlot(residuals=residuals, bins=30)
        dist_plot.render(ax=ax)

        ax.set_title(f"{dataset_name.capitalize()}", fontsize=12, fontweight="bold")
        ax.grid(alpha=0.3, axis="y")

    fig.suptitle("Residual Distribution", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()

    return fig


def create_regression_distances_plot(
    X: np.ndarray,
    y_true: np.ndarray,
    leverage_detector,
    student_detector,
    color_by: Optional[Union[str, Dict[str, np.ndarray]]] = None,
    figsize: Tuple[float, float] = (10, 6),
    annotate_by: Optional[Union[str, Dict[str, np.ndarray]]] = None,
    color_mode: Literal["continuous", "categorical"] = "continuous",
) -> Figure:
    """Create regression diagnostic distances plot for training data.

    Creates a plot of Leverage vs Studentized Residuals with confidence limits.
    This helps identify influential points and outliers in regression models.

    Parameters
    ----------
    X : np.ndarray
        Training data features.
    y_true : np.ndarray
        True target values (used for coloring and annotations).
    leverage_detector : Leverage
        Fitted leverage detector.
    student_detector : StudentizedResiduals
        Fitted studentized residuals detector.
    color_by : str or dict, optional
        Coloring specification.
    figsize : Tuple[float, float]
        Figure size.
    annotate_by : str or dict, optional
        Annotations for plot points.
    color_mode : Literal["continuous", "categorical"], default="continuous"
        Mode for coloring points.

    Returns
    -------
    Figure
        Matplotlib figure with regression distances plot.
    """
    # Get confidence limits from detectors
    leverage_limit = leverage_detector.critical_value_
    student_limit = student_detector.critical_value_

    leverages = leverage_detector.predict_residuals(X)
    studentized = student_detector.predict_residuals(X, y_true)

    # This plot is specifically for training data diagnostics
    dataset_name = "train"

    color_values = prepare_color_values(
        color_by, dataset_name, y_true, leverages.shape[0]
    )

    fig, ax = plt.subplots(figsize=figsize)

    # Create distances plot
    distances_plot = DistancesPlot(
        y=studentized,
        x=leverages,
        color_by=color_values,
        confidence_lines=(leverage_limit, student_limit),
        color_mode=color_mode,
    )
    distances_plot.render(ax=ax)

    # Add annotations if requested
    labels = prepare_annotations(annotate_by, dataset_name, X, y_true)
    if labels is not None:
        annotate_points(
            ax,
            leverages,
            studentized,
            labels,
            fontsize=8,
            alpha=0.7,
            xytext=(3, 3),
            textcoords="offset points",
        )

    if student_limit is not None:
        negative_limit = -abs(student_limit)
        ax.axhline(
            y=negative_limit,
            color="red",
            linestyle="--",
            linewidth=1,
            alpha=0.7,
        )

    ax.set_xlabel("Leverage", fontsize=10)
    ax.set_ylabel("Studentized Residuals", fontsize=10)
    ax.set_title(
        "Regression Distances: Leverage vs Studentized Residuals",
        fontsize=12,
        fontweight="bold",
    )
    ax.grid(alpha=0.3)
    plt.tight_layout()

    return fig
