"""Diagnostic plot creation functions for inspectors.

This module contains functions for creating diagnostic plots like
Hotelling's T² vs Q residuals, which are used to identify outliers
in decomposition models (PCA, PLS, etc.).
"""

from __future__ import annotations
from typing import Dict, Optional, Tuple, TYPE_CHECKING
import numpy as np
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from matplotlib.figure import Figure

from chemotools.plotting import DistancesPlot
from chemotools.plotting._styles import DATASET_COLORS
from chemotools.outliers import HotellingT2, QResiduals

from ._utils import select_primary_target


def create_model_distances_plot(
    datasets_data: Dict[str, Dict[str, Optional[np.ndarray]]],
    model,
    confidence: float,
    color_by_y: bool,
    figsize: Tuple[float, float],
) -> Figure:
    """Create model diagnostic distances plot across one or more datasets.

    This function renders Hotelling's T² vs Q residuals for the provided
    datasets, drawing confidence limits and optional colouring by target
    values. It replaces the previous single- and multi-dataset helpers with
    a unified implementation.

    Parameters
    ----------
    datasets_data : Dict[str, Dict[str, Optional[np.ndarray]]]
        Mapping from dataset name to a dictionary containing ``"X"``
        (required) and optional ``"y"`` arrays. The function renders each
        dataset on the same axes, applying dataset-specific colours when
        ``color_by_y`` is False or target values are unavailable.
    model : fitted model
        Fitted decomposition model (PCA, PLS, etc.) that provides latent
        scores used by the distance detectors.
    confidence : float
        Confidence level for the Hotelling's T² and Q residual detectors.
    color_by_y : bool
        Whether to colour points using the provided ``y`` targets.
    figsize : Tuple[float, float]
        Figure size (width, height) in inches.

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

    for ds_name, data in dataset_items:
        X = data.get("X")
        y = data.get("y")

        if X is None:
            raise ValueError(f"X data is required for dataset '{ds_name}'")

        # Calculate Hotelling's T² residuals
        hotelling = HotellingT2(model, confidence=confidence)
        hotelling.fit(X)
        t2 = hotelling.predict_residuals(X)

        # Calculate Q residuals
        q_res_model = QResiduals(model, confidence=confidence)
        q_res_model.fit(X)
        q = q_res_model.predict_residuals(X)

        color_by = (
            select_primary_target(y) if (color_by_y and y is not None) else None
        )
        dataset_color = (
            DATASET_COLORS.get(ds_name, "#7f7f7f")
            if color_by is None and multi_dataset
            else None
        )

        # Only draw confidence limits when plotting the training dataset
        should_draw_limits = (not multi_dataset) or (ds_name.lower() == "train")
        confidence_lines = (
            (
                hotelling.critical_value_,
                q_res_model.critical_value_,
            )
            if should_draw_limits
            else None
        )

        dist_plot = DistancesPlot(
            y=q,
            x=t2,
            color_by=color_by,
            label=ds_name.capitalize(),
            color=dataset_color,
            colormap=None,
            confidence_lines=confidence_lines,
        )
        dist_plot.render(ax)

    ax.set_xlabel("Hotelling's T²", fontsize=10)
    ax.set_ylabel("Q Residuals", fontsize=10)

    title_prefix = "Model Distances: Hotelling's T² vs Q Residuals"
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
    plt.tight_layout()
    return fig
