"""Spectra comparison plot creation functions.

This module contains functions for creating spectra comparison plots,
showing raw vs preprocessed spectra. Useful for spectroscopy-based
decomposition models (IR, Raman, NMR, etc.).
"""

from __future__ import annotations
from typing import Dict, Optional, Tuple, TYPE_CHECKING
import numpy as np
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from matplotlib.figure import Figure

from chemotools.plotting import SpectrumPlot
from chemotools.plotting._styles import DATASET_COLORS

from ._utils import select_primary_target


def create_spectra_plots_single_dataset(
    X_raw: np.ndarray,
    X_preprocessed: np.ndarray,
    y: Optional[np.ndarray],
    wavenumbers: np.ndarray,
    preprocessed_wavenumbers: np.ndarray,
    dataset_name: str,
    color_by_y: bool,
    xlabel: str,
    xlim: Optional[Tuple[float, float]],
    figsize: Tuple[float, float],
) -> Dict[str, Figure]:
    """Create raw and preprocessed spectra plots for a single dataset.

    Creates two separate figures comparing raw and preprocessed spectra.
    Useful for visualizing the effect of preprocessing steps.

    Parameters
    ----------
    X_raw : np.ndarray
        Raw spectra data of shape (n_samples, n_features)
    X_preprocessed : np.ndarray
        Preprocessed spectra data of shape (n_samples, n_features_preprocessed)
    y : Optional[np.ndarray]
        Target values for coloring spectra
    wavenumbers : np.ndarray
        Wavenumbers/wavelengths for raw spectra
    preprocessed_wavenumbers : np.ndarray
        Wavenumbers/wavelengths for preprocessed spectra
        (may differ if feature selection was applied)
    dataset_name : str
        Name of dataset (e.g., 'train', 'test', 'val')
    color_by_y : bool
        Whether to color spectra by y values
    xlabel : str
        Label for x-axis (e.g., "Wavenumber (cm⁻¹)")
    xlim : Optional[Tuple[float, float]]
        X-axis limits for zooming into spectral regions
    figsize : Tuple[float, float]
        Figure size (width, height) in inches

    Returns
    -------
    Dict[str, Figure]
        Dictionary with 'raw_spectra' and 'preprocessed_spectra' keys

    Examples
    --------
    >>> X_raw = np.random.rand(50, 1000)
    >>> X_preprocessed = np.random.rand(50, 800)
    >>> wavenumbers = np.linspace(4000, 400, 1000)
    >>> preprocessed_wn = np.linspace(4000, 400, 800)
    >>> figs = create_spectra_plots_single_dataset(
    ...     X_raw, X_preprocessed, None, wavenumbers, preprocessed_wn,
    ...     'train', False, 'Wavenumber (cm⁻¹)', None, (12, 5)
    ... )
    >>> figs['raw_spectra'].savefig('raw.png')
    """
    figures = {}

    color_values = None
    if color_by_y and y is not None:
        color_values = select_primary_target(y)

    # Figure 1: Raw spectra
    plot_raw = SpectrumPlot(
        x=wavenumbers,
        y=X_raw,
        color_by=color_values,
        colormap="viridis",
    )
    fig1 = plot_raw.show(
        figsize=figsize,
        title=f"Raw Spectra ({dataset_name.capitalize()})",
        xlabel=xlabel,
        ylabel="Intensity",
        xlim=xlim,
    )
    figures["raw_spectra"] = fig1

    # Figure 2: Preprocessed spectra
    plot_preprocessed = SpectrumPlot(
        x=preprocessed_wavenumbers,
        y=X_preprocessed,
        color_by=color_values,
        colormap="viridis",
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
    preprocessed_data: Dict[str, np.ndarray],
    wavenumbers: np.ndarray,
    preprocessed_wavenumbers: np.ndarray,
    xlabel: str,
    xlim: Optional[Tuple[float, float]],
    figsize: Tuple[float, float],
) -> Dict[str, Figure]:
    """Create raw and preprocessed spectra plots with multiple datasets.

    Creates two figures with all datasets plotted together, useful for
    comparing spectra across train/test/validation sets.

    Parameters
    ----------
    raw_data : Dict[str, np.ndarray]
        Dictionary mapping dataset names to raw spectra arrays
    preprocessed_data : Dict[str, np.ndarray]
        Dictionary mapping dataset names to preprocessed spectra arrays
    wavenumbers : np.ndarray
        Wavenumbers/wavelengths for raw spectra
    preprocessed_wavenumbers : np.ndarray
        Wavenumbers/wavelengths for preprocessed spectra
    xlabel : str
        Label for x-axis (e.g., "Wavenumber (cm⁻¹)")
    xlim : Optional[Tuple[float, float]]
        X-axis limits for zooming
    figsize : Tuple[float, float]
        Figure size (width, height) in inches

    Returns
    -------
    Dict[str, Figure]
        Dictionary with 'raw_spectra' and 'preprocessed_spectra' keys

    Examples
    --------
    >>> train = np.random.rand(50, 1000)
    >>> test = np.random.rand(30, 1000)
    >>> raw_data = {'train': train, 'test': test}
    >>> preprocessed_data = {'train': train * 2, 'test': test * 2}
    >>> wavenumbers = np.linspace(4000, 400, 1000)
    >>> figs = create_spectra_plots_multi_dataset(
    ...     raw_data, preprocessed_data, wavenumbers, wavenumbers,
    ...     'Wavenumber (cm⁻¹)', None, (12, 5)
    ... )
    """
    figures = {}

    # Create raw spectra plot with all datasets
    fig1, ax1 = plt.subplots(figsize=figsize)

    for ds_name, X in raw_data.items():
        color = DATASET_COLORS.get(
            ds_name,
        )
        for i in range(X.shape[0]):
            ax1.plot(
                wavenumbers,
                X[i, :],
                color=color,
                alpha=0.6,
                linewidth=1,
                label=ds_name.capitalize() if i == 0 else None,
            )

    ax1.set_xlabel(xlabel, fontsize=10)
    ax1.set_ylabel("Intensity", fontsize=10)
    ax1.set_title("Raw Spectra Comparison", fontsize=12, fontweight="bold")
    ax1.grid(alpha=0.3)
    if xlim:
        ax1.set_xlim(xlim)
    ax1.legend(loc="best")
    plt.tight_layout()
    figures["raw_spectra"] = fig1

    # Create preprocessed spectra plot with all datasets
    fig2, ax2 = plt.subplots(figsize=figsize)

    for ds_name, X in preprocessed_data.items():
        color = DATASET_COLORS.get(ds_name)
        for i in range(X.shape[0]):
            ax2.plot(
                preprocessed_wavenumbers,
                X[i, :],
                color=color,
                alpha=0.6,
                linewidth=1,
                label=ds_name.capitalize() if i == 0 else None,
            )

    ax2.set_xlabel(xlabel, fontsize=10)
    ax2.set_ylabel("Intensity", fontsize=10)
    ax2.set_title("Preprocessed Spectra Comparison", fontsize=12, fontweight="bold")
    ax2.grid(alpha=0.3)
    if xlim:
        ax2.set_xlim(xlim)
    ax2.legend(loc="best")
    plt.tight_layout()
    figures["preprocessed_spectra"] = fig2

    return figures
