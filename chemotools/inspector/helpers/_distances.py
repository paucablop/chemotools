"""Distance plot creation functions for inspectors."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Literal, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

from chemotools.outliers import HotellingT2, QResiduals
from chemotools.plotting import DistancesPlot
from chemotools.plotting._styles import DATASET_COLORS, DATASET_MARKERS
from chemotools.plotting._utils import annotate_points

from ..core.utils import prepare_annotations, prepare_color_values

# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _resolve_training_entry(
    datasets_data: Dict[str, Dict[str, Optional[np.ndarray]]],
    dataset_items: list,
    training_dataset: str,
) -> Tuple[Dict[str, Optional[np.ndarray]], str]:
    """Return ``(train_entry, training_dataset_lower)`` from *datasets_data*.

    Falls back to the first dataset when *training_dataset* is absent.
    """
    if training_dataset in datasets_data:
        return datasets_data[training_dataset], training_dataset.lower()

    first_name, first_entry = dataset_items[0]
    return first_entry, first_name.lower()


def _ensure_detectors(
    datasets_data: Dict[str, Dict[str, Optional[np.ndarray]]],
    dataset_items: list,
    model,
    confidence: float,
    training_dataset: str,
    *,
    hotelling_detector: Optional[HotellingT2] = None,
    q_residuals_detector: Optional[QResiduals] = None,
) -> Tuple[HotellingT2, QResiduals, str]:
    """Return ``(hotelling, q_residuals, training_dataset_lower)``.

    Detectors are fitted on the training data when not already provided.
    """
    training_dataset_lower = training_dataset.lower()

    if hotelling_detector is not None and q_residuals_detector is not None:
        return hotelling_detector, q_residuals_detector, training_dataset_lower

    train_entry, training_dataset_lower = _resolve_training_entry(
        datasets_data, dataset_items, training_dataset
    )
    train_X = train_entry.get("X")
    if train_X is None:
        raise ValueError(
            "X data is required for detector fitting when detectors are not supplied"
        )

    if hotelling_detector is None:
        hotelling_detector = HotellingT2(model, confidence=confidence)
        hotelling_detector.fit(train_X)
    if q_residuals_detector is None:
        q_residuals_detector = QResiduals(model, confidence=confidence)
        q_residuals_detector.fit(train_X)

    return hotelling_detector, q_residuals_detector, training_dataset_lower


def _ensure_q_detector(
    datasets_data: Dict[str, Dict[str, Optional[np.ndarray]]],
    dataset_items: list,
    model,
    confidence: float,
    training_dataset: str,
    q_residuals_detector: Optional[QResiduals] = None,
) -> Tuple[QResiduals, str]:
    """Return ``(q_residuals_detector, training_dataset_lower)``.

    The detector is fitted on the training data when not already provided.
    """
    training_dataset_lower = training_dataset.lower()

    if q_residuals_detector is not None:
        return q_residuals_detector, training_dataset_lower

    train_entry, training_dataset_lower = _resolve_training_entry(
        datasets_data, dataset_items, training_dataset
    )
    train_X = train_entry.get("X")
    if train_X is None:
        raise ValueError(
            "X data is required for detector fitting when detectors are not supplied"
        )

    q_residuals_detector = QResiduals(model, confidence=confidence)
    q_residuals_detector.fit(train_X)
    return q_residuals_detector, training_dataset_lower


def _compute_y_residuals(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Compute per-sample Y residuals (L2 norm for multi-target)."""
    y_true_arr = np.atleast_2d(np.asarray(y_true))
    y_pred_arr = np.atleast_2d(np.asarray(y_pred))

    # atleast_2d may add the extra dim in axis-0; ensure shape (n, targets)
    if y_true_arr.shape[0] == 1 and y_true_arr.shape[1] != 1:
        y_true_arr = y_true_arr.T
    if y_pred_arr.shape[0] == 1 and y_pred_arr.shape[1] != 1:
        y_pred_arr = y_pred_arr.T

    residuals_matrix = y_true_arr - y_pred_arr
    if residuals_matrix.shape[1] == 1:
        return residuals_matrix.ravel()
    return np.linalg.norm(residuals_matrix, axis=1)


def _decorate_distances_plot(
    ax: Axes,
    title_prefix: str,
    multi_dataset: bool,
    dataset_items: list,
) -> None:
    """Apply shared title / legend / grid decorations for distance plots."""
    if multi_dataset:
        ax.set_title(title_prefix, fontsize=12, fontweight="bold")
        ax.legend(loc="best")
    else:
        dataset_name = dataset_items[0][0].capitalize()
        ax.set_title(
            f"{title_prefix} ({dataset_name})",
            fontsize=12,
            fontweight="bold",
        )
    ax.grid(alpha=0.3)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def create_model_distances_plot(
    datasets_data: Dict[str, Dict[str, Optional[np.ndarray]]],
    model,
    confidence: float,
    color_by: Optional[Union[str, Dict[str, np.ndarray]]],
    figsize: Tuple[float, float],
    *,
    hotelling_detector: Optional[HotellingT2] = None,
    q_residuals_detector: Optional[QResiduals] = None,
    training_dataset: str = "train",
    annotate_by: Optional[Union[str, Dict[str, np.ndarray]]] = None,
    color_mode: Optional[Literal["continuous", "categorical"]] = None,
) -> Figure:
    """Create model diagnostic distances plot across one or more datasets.

    This function renders Hotelling's T² vs Q residuals for the provided
    datasets, drawing confidence limits and optional colouring by target
    values.

    Parameters
    ----------
    datasets_data : Dict[str, Dict[str, Optional[np.ndarray]]]
        Mapping from dataset name to a dictionary containing ``"X"``
        (required) and optional ``"y"`` arrays.
    model : fitted model
        Fitted decomposition model (PCA, PLS, etc.).
    confidence : float
        Confidence level for the detectors.
    color_by : str or dict, optional
        Coloring specification.
    figsize : Tuple[float, float]
        Figure size (width, height) in inches.
    hotelling_detector : Optional[HotellingT2], default=None
        Pre-fitted Hotelling's T² detector.
    q_residuals_detector : Optional[QResiduals], default=None
        Pre-fitted Q residuals detector.
    training_dataset : str, default="train"
        Name of the dataset used to train the detectors when they are not
        supplied.
    annotate_by : str or dict, optional
        Annotations for plot points.
    color_mode : Literal["continuous", "categorical"], optional
        Mode for coloring points.

    Returns
    -------
    Figure
        Matplotlib figure containing the composed distances plot.
    """

    if not datasets_data:
        raise ValueError("datasets_data must contain at least one dataset")

    fig, ax = plt.subplots(figsize=figsize)
    dataset_items = list(datasets_data.items())
    multi_dataset = len(dataset_items) > 1

    hotelling_detector, q_residuals_detector, training_dataset_lower = (
        _ensure_detectors(
            datasets_data,
            dataset_items,
            model,
            confidence,
            training_dataset,
            hotelling_detector=hotelling_detector,
            q_residuals_detector=q_residuals_detector,
        )
    )

    for ds_name, data in dataset_items:
        X = data.get("X")
        y = data.get("y")

        if X is None:
            raise ValueError(f"X data is required for dataset '{ds_name}'")

        t2 = hotelling_detector.predict_residuals(X)
        q = q_residuals_detector.predict_residuals(X)

        color_values = prepare_color_values(color_by, ds_name, y, X.shape[0])

        if multi_dataset:
            dataset_color = DATASET_COLORS.get(ds_name)
            marker = DATASET_MARKERS.get(ds_name, "o")
        else:
            dataset_color = None
            marker = "o"

        # Only draw confidence limits when plotting the training dataset
        should_draw_limits = (not multi_dataset) or (
            ds_name.lower() == training_dataset_lower
        )
        confidence_lines = (
            (
                hotelling_detector.critical_value_,
                q_residuals_detector.critical_value_,
            )
            if should_draw_limits
            else None
        )

        dist_plot = DistancesPlot(
            y=q,
            x=t2,
            color_by=color_values,
            label=ds_name.capitalize(),
            color=dataset_color if color_values is None else None,
            colormap=None,
            marker=marker,
            confidence_lines=confidence_lines,
            color_mode=color_mode,
        )
        dist_plot.render(ax)

        labels = prepare_annotations(annotate_by, ds_name, X, y)
        if labels is not None:
            annotate_points(
                ax,
                t2,
                q,
                labels,
                fontsize=8,
                alpha=0.7,
                xytext=(3, 3),
                textcoords="offset points",
            )

    ax.set_xlabel("Hotelling's T²", fontsize=10)
    ax.set_ylabel("Q Residuals", fontsize=10)
    _decorate_distances_plot(
        ax,
        "Model Distances: Hotelling's T² vs Q Residuals",
        multi_dataset,
        dataset_items,
    )
    plt.tight_layout()
    return fig


def create_q_vs_y_residuals_plot(
    datasets_data: Dict[str, Dict[str, Optional[np.ndarray]]],
    model,
    confidence: float,
    color_by: Optional[Union[str, Dict[str, np.ndarray]]],
    figsize: Tuple[float, float],
    *,
    q_residuals_detector: Optional[QResiduals] = None,
    training_dataset: str = "train",
    annotate_by: Optional[Union[str, Dict[str, np.ndarray]]] = None,
    color_mode: Literal["continuous", "categorical"] = "continuous",
) -> Figure:
    """Create Q residuals vs Y residuals diagnostic plot for regression models.

    Parameters
    ----------
    datasets_data : Dict[str, Dict[str, Optional[np.ndarray]]]
        Mapping from dataset name to a dictionary containing ``"X"``
        (required), ``"y_pred"`` (required), ``"y_true"`` (required),
        and optional ``"y"`` arrays.
    model : fitted model
        Fitted regression model with latent variables.
    confidence : float
        Confidence level for the Q residual detector.
    color_by : str or dict, optional
        Coloring specification.
    figsize : Tuple[float, float]
        Figure size (width, height) in inches.
    q_residuals_detector : Optional[QResiduals], default=None
        Pre-fitted Q residuals detector.
    training_dataset : str, default="train"
        Name of the dataset used to train the detectors.
    annotate_by : str or dict, optional
        Annotations for plot points.
    color_mode : Literal["continuous", "categorical"], default="continuous"
        Mode for coloring points.

    Returns
    -------
    Figure
        Matplotlib figure containing the Q vs Y residuals plot.
    """

    if not datasets_data:
        raise ValueError("datasets_data must contain at least one dataset")

    fig, ax = plt.subplots(figsize=figsize)
    dataset_items = list(datasets_data.items())
    multi_dataset = len(dataset_items) > 1

    q_residuals_detector, training_dataset_lower = _ensure_q_detector(
        datasets_data,
        dataset_items,
        model,
        confidence,
        training_dataset,
        q_residuals_detector=q_residuals_detector,
    )

    for ds_name, data in dataset_items:
        X = data.get("X")
        y = data.get("y")
        y_true = data.get("y_true")
        y_pred = data.get("y_pred")

        if X is None:
            raise ValueError(f"X data is required for dataset '{ds_name}'")
        if y_true is None:
            raise ValueError(f"y_true data is required for dataset '{ds_name}'")
        if y_pred is None:
            raise ValueError(f"y_pred data is required for dataset '{ds_name}'")

        q = q_residuals_detector.predict_residuals(X)
        y_residuals = _compute_y_residuals(y_true, y_pred)

        # When multiple datasets, always color by dataset, not by y values
        if multi_dataset:
            color_values = None
            dataset_color = DATASET_COLORS.get(ds_name)
            marker = DATASET_MARKERS.get(ds_name, "o")
        else:
            color_values = prepare_color_values(color_by, ds_name, y, q.shape[0])
            dataset_color = None
            marker = "o"

        # Only draw Q residuals confidence limit when plotting the training dataset
        should_draw_limits = (not multi_dataset) or (
            ds_name.lower() == training_dataset_lower
        )
        confidence_lines = (
            (
                None,  # No confidence limit for Y residuals (x-axis)
                q_residuals_detector.critical_value_,
            )
            if should_draw_limits
            else None
        )

        dist_plot = DistancesPlot(
            y=q,
            x=y_residuals,
            color_by=color_values,
            label=ds_name.capitalize(),
            color=dataset_color,
            colormap=None,
            marker=marker,
            confidence_lines=confidence_lines,
            color_mode=color_mode,
        )
        dist_plot.render(ax)

        labels = prepare_annotations(annotate_by, ds_name, X, y)
        if labels is not None:
            annotate_points(
                ax,
                y_residuals,
                q,
                labels,
                fontsize=8,
                alpha=0.7,
                xytext=(3, 3),
                textcoords="offset points",
            )

    ax.axvline(x=0, color="black", linestyle="-", linewidth=1, alpha=0.5, zorder=1)
    ax.set_xlabel("Y Residuals (Prediction Error)", fontsize=10)
    ax.set_ylabel("Q Residuals (SPE)", fontsize=10)
    _decorate_distances_plot(
        ax,
        "Regression Distances: Q Residuals vs Y Residuals",
        multi_dataset,
        dataset_items,
    )
    plt.tight_layout()
    return fig
