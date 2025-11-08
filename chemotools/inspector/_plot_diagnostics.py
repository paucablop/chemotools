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


def create_distances_plot_single_dataset(
    X: np.ndarray,
    y: Optional[np.ndarray],
    model,
    confidence: float,
    dataset_name: str,
    color_by_y: bool,
    figsize: Tuple[float, float],
) -> Figure:
    """Create diagnostic distances plot for a single dataset.

    Creates a plot of Hotelling's T² vs Q residuals with confidence limits.
    This helps identify outliers and assess model fit.

    Parameters
    ----------
    X : np.ndarray
        Input data of shape (n_samples, n_features)
    y : Optional[np.ndarray]
        Target values for coloring
    model : fitted model
        Fitted decomposition model (PCA, PLS, etc.)
    confidence : float
        Confidence level for critical value lines (e.g., 0.95 for 95%)
    dataset_name : str
        Name of dataset (e.g., 'train', 'test', 'val')
    color_by_y : bool
        Whether to color points by y values
    figsize : Tuple[float, float]
        Figure size (width, height) in inches

    Returns
    -------
    Figure
        Matplotlib figure with distances plot

    Examples
    --------
    >>> from sklearn.decomposition import PCA
    >>> X = np.random.rand(100, 50)
    >>> model = PCA(n_components=5).fit(X)
    >>> fig = create_distances_plot_single_dataset(
    ...     X, None, model, 0.95, 'train', False, (8, 6)
    ... )
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Calculate Hotelling's T² residuals
    hotelling = HotellingT2(model, confidence=confidence)
    hotelling.fit(X)
    t2 = hotelling.predict_residuals(X)

    # Calculate Q residuals
    q_res_model = QResiduals(model, confidence=confidence)
    q_res_model.fit(X)
    q = q_res_model.predict_residuals(X)

    # Combine into 2D array
    distances = np.column_stack([t2, q])

    # Determine color_by parameter
    color_by = y if (color_by_y and y is not None) else None

    # Create and render DistancesPlot
    dist_plot = DistancesPlot(
        distances=distances,
        distances_selection=(0, 1),
        color_by=color_by,
        label=dataset_name.capitalize(),
        colormap="viridis" if color_by is not None else None,
        confidence_lines=(
            hotelling.critical_value_,
            q_res_model.critical_value_,
        ),
    )
    dist_plot.render(ax)

    # Apply decorations
    ax.set_xlabel("Hotelling's T²", fontsize=10)
    ax.set_ylabel("Q Residuals", fontsize=10)
    ax.set_title(
        f"Diagnostic Distances Plot ({dataset_name.capitalize()})",
        fontsize=12,
        fontweight="bold",
    )
    ax.grid(alpha=0.3)

    plt.tight_layout()
    return fig


def create_distances_plot_multi_dataset(
    datasets_data: Dict[str, Dict[str, Optional[np.ndarray]]],
    model,
    confidence: float,
    color_by_y: bool,
    figsize: Tuple[float, float],
) -> Figure:
    """Create diagnostic distances plot with multiple datasets.

    Parameters
    ----------
    datasets_data : Dict[str, Dict[str, Optional[np.ndarray]]]
        Dictionary mapping dataset names to {'X': ..., 'y': ...}
        'y' can be None
    model
        Fitted decomposition model (PCA, PLS, etc.)
    confidence : float
        Confidence level for critical value lines (e.g., 0.95)
    color_by_y : bool
        Whether to color points by y values
    figsize : Tuple[float, float]
        Figure size (width, height) in inches

    Returns
    -------
    Figure
        Matplotlib figure with distances plot
    """
    fig, ax = plt.subplots(figsize=figsize)

    for ds_name, data in datasets_data.items():
        X = data["X"]
        y = data["y"]

        # X should always be present for diagnostic plots
        assert X is not None, f"X data is required for dataset {ds_name}"

        # Calculate Hotelling's T² residuals
        hotelling = HotellingT2(model, confidence=confidence)
        hotelling.fit(X)
        t2 = hotelling.predict_residuals(X)

        # Calculate Q residuals
        q_res_model = QResiduals(model, confidence=confidence)
        q_res_model.fit(X)
        q = q_res_model.predict_residuals(X)

        # Combine into 2D array
        distances = np.column_stack([t2, q])

        color = DATASET_COLORS.get(ds_name, "#7f7f7f")

        # Determine color_by parameter
        color_by = y if (color_by_y and y is not None) else None

        # Create and render DistancesPlot
        dist_plot = DistancesPlot(
            distances=distances,
            distances_selection=(0, 1),
            color_by=color_by,
            label=ds_name.capitalize(),
            color=color if color_by is None else None,
            colormap="viridis" if color_by is not None else None,
            confidence_lines=(
                hotelling.critical_value_,
                q_res_model.critical_value_,
            ),
        )
        dist_plot.render(ax)

    # Apply decorations
    ax.set_xlabel("Hotelling's T²", fontsize=10)
    ax.set_ylabel("Q Residuals", fontsize=10)
    ax.set_title("Diagnostic Distances Plot", fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")

    plt.tight_layout()
    return fig
