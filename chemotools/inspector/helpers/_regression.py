"""Regression-specific plot creation functions for inspectors.

This module contains plotting functions specific to regression models (PLS, etc.)
that are not applicable to unsupervised methods like PCA.

Each function handles both single and multi-dataset cases internally, using
plot objects from the chemotools.plotting module for consistent rendering.
"""

from __future__ import annotations
from typing import Dict, Tuple, TYPE_CHECKING, Optional, Union
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

from .._utils import select_primary_target, prepare_annotations


def create_predicted_vs_actual_plot(
    datasets_data: Dict[str, Dict[str, np.ndarray]],
    color_by_y: bool,
    figsize: Tuple[float, float],
    annotate_by: Optional[Union[str, Dict[str, np.ndarray]]] = None,
) -> Figure:
    """Create predicted vs actual plot for one or multiple datasets.

    Handles both single and multi-dataset cases internally. For single dataset,
    can color by y-values. For multiple datasets, colors by dataset.

    Parameters
    ----------
    datasets_data : Dict[str, Dict[str, np.ndarray]]
        Dictionary mapping dataset names to dicts with 'y_true', 'y_pred', 'y' keys
    color_by_y : bool
        Whether to color by y values (only used for single dataset)
    figsize : Tuple[float, float]
        Figure size
    annotate_by : str or dict, optional
        Annotations for plot points.

    Returns
    -------
    Figure
        Matplotlib figure with predicted vs actual plot
    """
    n_datasets = len(datasets_data)

    if n_datasets == 1:
        # Single dataset - use color_by_y option
        dataset_name, data = list(datasets_data.items())[0]
        y_true = data["y_true"]
        y_pred = data["y_pred"]
        y = data.get("y")
        X = data.get("X")

        color_reference = (
            select_primary_target(y) if (color_by_y and y is not None) else None
        )

        fig, ax = plt.subplots(figsize=figsize)

        pred_actual_plot = PredictedVsActualPlot(
            y_true=y_true,
            y_pred=y_pred,
            color_by=color_reference,
        )
        pred_actual_plot.render(ax=ax)

        # Add annotations if requested
        labels = prepare_annotations(annotate_by, dataset_name, X, y)
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
        y = data.get("y")
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
        labels = prepare_annotations(annotate_by, dataset_name, X, y)
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
    color_by_y: bool,
    figsize: Tuple[float, float],
    annotate_by: Optional[Union[str, Dict[str, np.ndarray]]] = None,
) -> Figure:
    """Create residual scatter plot for one or multiple datasets.

    Handles both single and multi-dataset cases internally. For single dataset,
    shows one plot with optional y-coloring. For multiple datasets, creates
    side-by-side subplots with confidence bands for each.

    Parameters
    ----------
    datasets_data : Dict[str, Dict[str, np.ndarray]]
        Dictionary mapping dataset names to dicts with 'y_true', 'y_pred', 'y' keys
    color_by_y : bool
        Whether to color by y values (only used for single dataset)
    figsize : Tuple[float, float]
        Figure size
    annotate_by : str or dict, optional
        Annotations for plot points.

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
        y = data.get("y")
        X = data.get("X")

        color_reference = (
            select_primary_target(y) if (color_by_y and y is not None) else None
        )
        residuals = y_true - y_pred

        fig, ax = plt.subplots(figsize=figsize)

        residuals_plot = YResidualsPlot(
            residuals=residuals,
            x_values=y_pred,
            color_by=color_reference,
            add_confidence_band=2.0,
        )
        residuals_plot.render(ax=ax)

        # Add annotations if requested
        labels = prepare_annotations(annotate_by, dataset_name, X, y)
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
        y = data.get("y")
        X = data.get("X")

        color_reference = (
            select_primary_target(y) if (color_by_y and y is not None) else None
        )
        residuals = y_true - y_pred

        residuals_plot = YResidualsPlot(
            residuals=residuals,
            x_values=y_pred,
            color_by=color_reference,
            add_confidence_band=2.0,
        )
        residuals_plot.render(ax=ax)

        # Add annotations if requested
        labels = prepare_annotations(annotate_by, dataset_name, X, y)
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
    datasets_data: Dict[str, Dict[str, np.ndarray]],
    leverage_detector,
    student_detector,
    color_by_y: bool,
    figsize: Tuple[float, float],
    annotate_by: Optional[Union[str, Dict[str, np.ndarray]]] = None,
) -> Figure:
    """Create regression diagnostic distances plot for one or multiple datasets.

    Creates a plot of Leverage vs Studentized Residuals with confidence limits.
    This helps identify influential points and outliers in regression models.
    Handles both single and multi-dataset cases internally.

    Parameters
    ----------
    datasets_data : Dict[str, Dict[str, np.ndarray]]
        Dictionary mapping dataset names to dicts with 'X', 'y', 'y_true', 'y_pred' keys
    leverage_detector : Leverage
        Fitted leverage detector
    student_detector : StudentizedResiduals
        Fitted studentized residuals detector
    color_by_y : bool
        Whether to color by y (only for single dataset, ignored for multiple)
    figsize : Tuple[float, float]
        Figure size
    annotate_by : str or dict, optional
        Annotations for plot points.

    Returns
    -------
    Figure
        Matplotlib figure with regression distances plot
    """
    n_datasets = len(datasets_data)

    # Get confidence limits from detectors
    leverage_limit = leverage_detector.critical_value_
    student_limit = student_detector.critical_value_

    if n_datasets == 1:
        # Single dataset - single plot with optional y-coloring
        dataset_name, data = list(datasets_data.items())[0]
        X = data["X"]
        y = data.get("y")
        y_true = data["y_true"]

        leverages = leverage_detector.predict_residuals(X)
        studentized = student_detector.predict_residuals(X, y_true)

        fig, ax = plt.subplots(figsize=figsize)

        # Create distances plot
        distances_plot = DistancesPlot(
            y=studentized,
            x=leverages,
            color_by=select_primary_target(y) if color_by_y else None,
            confidence_lines=(leverage_limit, student_limit),
        )
        distances_plot.render(ax=ax)

        # Add annotations if requested
        labels = prepare_annotations(annotate_by, dataset_name, X, y)
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
            f"Regression Distances: Leverage vs Studentized Residuals ({dataset_name})",
            fontsize=12,
            fontweight="bold",
        )
        ax.grid(alpha=0.3)
        plt.tight_layout()

        return fig

    # Multiple datasets - overlay on single plot, color by dataset
    fig, ax = plt.subplots(figsize=figsize)

    # Plot each dataset
    for i, (dataset_name, data) in enumerate(datasets_data.items()):
        X = data["X"]
        y_true = data["y_true"]
        y = data.get("y")

        leverages = leverage_detector.predict_residuals(X)
        studentized = student_detector.predict_residuals(X, y_true)

        color = DATASET_COLORS.get(dataset_name, "gray")
        marker = DATASET_MARKERS.get(dataset_name, "o")

        # Create distances plot
        # Add confidence lines only for the first dataset (or training set)
        # Here we just add them once for simplicity
        should_add_lines = i == 0
        confidence_lines = (leverage_limit, student_limit) if should_add_lines else None

        distances_plot = DistancesPlot(
            y=studentized,
            x=leverages,
            label=dataset_name.capitalize(),
            color=color,
            marker=marker,
            confidence_lines=confidence_lines,
        )
        distances_plot.render(ax=ax)

        # Add annotations if requested
        labels = prepare_annotations(annotate_by, dataset_name, X, y)
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
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    plt.tight_layout()

    return fig
