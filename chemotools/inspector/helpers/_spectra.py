"""Spectra comparison plot creation functions.

This module contains functions for creating spectra comparison plots,
showing raw vs preprocessed spectra. Useful for spectroscopy-based
decomposition models (IR, Raman, NMR, etc.).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Literal, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from matplotlib.figure import Figure

from chemotools.plotting import SpectraPlot
from chemotools.plotting._styles import DATASET_COLORS

from ..core.utils import prepare_color_values


def create_spectra_plots_single_dataset(
    X_raw: np.ndarray,
    X_preprocessed: Optional[np.ndarray],
    y: Optional[np.ndarray],
    x_axis: np.ndarray,
    preprocessed_x_axis: Optional[np.ndarray],
    dataset_name: str,
    color_by: Optional[Union[str, Dict[str, np.ndarray]]],
    xlabel: str,
    xlim: Optional[Tuple[float, float]],
    figsize: Tuple[float, float],
    color_mode: Optional[Literal["continuous", "categorical"]] = None,
) -> Dict[str, Figure]:
    """Create raw and (optionally) preprocessed spectra plots for a single dataset.

    Always creates a raw spectra figure.  If *X_preprocessed* is provided,
    also creates a preprocessed spectra figure.

    Parameters
    ----------
    X_raw : np.ndarray
        Raw spectra data of shape (n_samples, n_features)
    X_preprocessed : np.ndarray or None
        Preprocessed spectra data of shape (n_samples, n_features_preprocessed).
        If ``None``, no preprocessed figure is generated.
    y : Optional[np.ndarray]
        Target values for coloring spectra
    x_axis : np.ndarray
        X-axis values (e.g. wavenumbers/wavelengths) for raw spectra
    preprocessed_x_axis : np.ndarray or None
        X-axis values for preprocessed spectra (may differ if feature selection
        was applied).  Required when *X_preprocessed* is not ``None``.
    dataset_name : str
        Name of dataset (e.g., 'train', 'test', 'val')
    color_by : str or dict, optional
        Coloring specification
    xlabel : str
        Label for x-axis (e.g., "Wavenumber (cm⁻¹)")
    xlim : Optional[Tuple[float, float]]
        X-axis limits for zooming into spectral regions
    figsize : Tuple[float, float]
        Figure size (width, height) in inches
    color_mode : Literal["continuous", "categorical"], default="continuous"
        Mode for coloring points.

    Returns
    -------
    Dict[str, Figure]
        Always contains ``'raw_spectra'``.  Contains
        ``'preprocessed_spectra'`` only when *X_preprocessed* is not ``None``.

    Examples
    --------
    >>> X_raw = np.random.rand(50, 1000)
    >>> X_preprocessed = np.random.rand(50, 800)
    >>> wavenumbers = np.linspace(4000, 400, 1000)
    >>> preprocessed_wn = np.linspace(4000, 400, 800)
    >>> figs = create_spectra_plots_single_dataset(
    ...     X_raw, X_preprocessed, None, wavenumbers, preprocessed_wn,
    ...     'train', None, 'Wavenumber (cm⁻¹)', None, (12, 5)
    ... )
    >>> figs['raw_spectra'].savefig('raw.png')
    """
    figures = {}

    color_values = prepare_color_values(color_by, dataset_name, y, X_raw.shape[0])

    # Suppress default labels when not using color_by to avoid cluttered legend
    # Pass empty strings as labels to prevent "Spectrum 0", "Spectrum 1", etc.
    suppress_labels = color_values is None
    empty_labels = [""] * X_raw.shape[0] if suppress_labels else None

    # Figure 1: Raw spectra
    plot_raw = SpectraPlot(
        x=x_axis,
        y=X_raw,
        color_by=color_values,
        colormap="shap",
        labels=empty_labels,
        color_mode=color_mode,
    )
    fig1 = plot_raw.show(
        figsize=figsize,
        title=f"Raw Spectra ({dataset_name.capitalize()})",
        xlabel=xlabel,
        ylabel="Intensity",
        xlim=xlim,
    )
    figures["raw_spectra"] = fig1

    # Figure 2: Preprocessed spectra (only when data is provided)
    if X_preprocessed is not None:
        if preprocessed_x_axis is None:
            raise ValueError(
                "preprocessed_x_axis must be provided when X_preprocessed is not None."
            )
        empty_labels_preproc = (
            [""] * X_preprocessed.shape[0] if suppress_labels else None
        )
        plot_preprocessed = SpectraPlot(
            x=preprocessed_x_axis,
            y=X_preprocessed,
            color_by=color_values,
            colormap="shap",
            labels=empty_labels_preproc,
            color_mode=color_mode,
        )
        fig2 = plot_preprocessed.show(
            figsize=figsize,
            title=f"Preprocessed Spectra ({dataset_name.capitalize()})",
            xlabel=xlabel,
            ylabel="Intensity",
            xlim=xlim,
        )
        figures["preprocessed_spectra"] = fig2

    return figures


def create_spectra_plots_multi_dataset(
    raw_data: Dict[str, np.ndarray],
    preprocessed_data: Optional[Dict[str, np.ndarray]],
    x_axis: np.ndarray,
    preprocessed_x_axis: Optional[np.ndarray],
    xlabel: str,
    xlim: Optional[Tuple[float, float]],
    figsize: Tuple[float, float],
    color_mode: Optional[Literal["continuous", "categorical"]] = None,
) -> Dict[str, Figure]:
    """Create raw and (optionally) preprocessed spectra plots with multiple datasets.

    Always creates a raw spectra figure with all datasets overlaid.  If
    *preprocessed_data* is provided, also creates a preprocessed spectra
    figure.

    Parameters
    ----------
    raw_data : Dict[str, np.ndarray]
        Dictionary mapping dataset names to raw spectra arrays
    preprocessed_data : Dict[str, np.ndarray] or None
        Dictionary mapping dataset names to preprocessed spectra arrays.
        If ``None``, no preprocessed figure is generated.
    x_axis : np.ndarray
        X-axis values for raw spectra
    preprocessed_x_axis : np.ndarray or None
        X-axis values for preprocessed spectra.  Required when
        *preprocessed_data* is not ``None``.
    xlabel : str
        Label for x-axis
    xlim : Optional[Tuple[float, float]]
        X-axis limits
    figsize : Tuple[float, float]
        Figure size
    color_mode : Literal["continuous", "categorical"], default="continuous"
        Mode for coloring points.

    Returns
    -------
    Dict[str, Figure]
        Always contains ``'raw_spectra'``.  Contains
        ``'preprocessed_spectra'`` only when *preprocessed_data* is not
        ``None``.
    """
    figures = {}

    # Create raw spectra plot with all datasets
    fig_raw, ax_raw = plt.subplots(figsize=figsize)

    for name, X in raw_data.items():
        # Get color for this dataset (default to black if not found)
        color = DATASET_COLORS.get(name, "black")

        # Create plot for this dataset
        # We use the dataset name as label for all spectra in this group
        # Matplotlib handles duplicate labels in legend automatically
        # But to be safe and efficient, we only label the first spectrum
        labels = [name.capitalize()] + [None] * (X.shape[0] - 1)
        plot = SpectraPlot(x=x_axis, y=X, labels=labels, color_mode=color_mode)

        plot.render(ax=ax_raw, color=color, alpha=0.6, linewidth=1)

    ax_raw.set_title("Raw Spectra Comparison", fontsize=14, fontweight="bold")
    ax_raw.set_xlabel(xlabel, fontsize=12)
    ax_raw.set_ylabel("Intensity", fontsize=12)
    ax_raw.grid(alpha=0.3)
    if xlim:
        ax_raw.set_xlim(xlim)
    ax_raw.legend()

    figures["raw_spectra"] = fig_raw

    # Create preprocessed spectra plot with all datasets (only when data is provided)
    if preprocessed_data is not None:
        if preprocessed_x_axis is None:
            raise ValueError(
                "preprocessed_x_axis must be provided when preprocessed_data is not "
                "None."
            )
        fig_prep, ax_prep = plt.subplots(figsize=figsize)

        for name, X in preprocessed_data.items():
            color = DATASET_COLORS.get(name, "black")

            labels = [name.capitalize()] + [None] * (X.shape[0] - 1)
            plot = SpectraPlot(
                x=preprocessed_x_axis, y=X, labels=labels, color_mode=color_mode
            )

            plot.render(ax=ax_prep, color=color, alpha=0.6, linewidth=1)

        ax_prep.set_title(
            "Preprocessed Spectra Comparison", fontsize=14, fontweight="bold"
        )
        ax_prep.set_xlabel(xlabel, fontsize=12)
        ax_prep.set_ylabel("Intensity", fontsize=12)
        ax_prep.grid(alpha=0.3)
        if xlim:
            ax_prep.set_xlim(xlim)
        ax_prep.legend()

        figures["preprocessed_spectra"] = fig_prep

    return figures
