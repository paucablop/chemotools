"""Spectra comparison plot creation functions.

This module contains functions for creating spectra comparison plots,
showing raw vs preprocessed spectra. Useful for spectroscopy-based
decomposition models (IR, Raman, NMR, etc.).
"""

from __future__ import annotations
from typing import Dict, Optional, Tuple, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from matplotlib.figure import Figure

from chemotools.plotting import SpectrumPlot

from .._utils import select_primary_target


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

    # Suppress default labels when not using color_by to avoid cluttered legend
    # Pass empty strings as labels to prevent "Spectrum 0", "Spectrum 1", etc.
    suppress_labels = color_values is None
    empty_labels = [""] * X_raw.shape[0] if suppress_labels else None

    # Figure 1: Raw spectra
    plot_raw = SpectrumPlot(
        x=wavenumbers,
        y=X_raw,
        color_by=color_values,
        colormap="viridis",
        labels=empty_labels,
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
    empty_labels_preproc = [""] * X_preprocessed.shape[0] if suppress_labels else None
    plot_preprocessed = SpectrumPlot(
        x=preprocessed_wavenumbers,
        y=X_preprocessed,
        color_by=color_values,
        colormap="viridis",
        labels=empty_labels_preproc,
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

    def _prepare_data(
        data_dict: Dict[str, np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        X_list = []
        labels = []
        for name, X in data_dict.items():
            X_list.append(X)
            labels.extend([name] * X.shape[0])
        return np.vstack(X_list), np.array(labels)

    # Create raw spectra plot with all datasets
    X_raw, labels_raw = _prepare_data(raw_data)
    plot_raw = SpectrumPlot(
        x=wavenumbers,
        y=X_raw,
        color_by=labels_raw,
        colormap="tab10",
        categorical=True,
    )
    figures["raw_spectra"] = plot_raw.show(
        figsize=figsize,
        title="Raw Spectra Comparison",
        xlabel=xlabel,
        ylabel="Intensity",
        xlim=xlim,
        alpha=0.6,
        linewidth=1,
    )

    # Create preprocessed spectra plot with all datasets
    X_prep, labels_prep = _prepare_data(preprocessed_data)
    plot_prep = SpectrumPlot(
        x=preprocessed_wavenumbers,
        y=X_prep,
        color_by=labels_prep,
        colormap="tab10",
        categorical=True,
    )
    figures["preprocessed_spectra"] = plot_prep.show(
        figsize=figsize,
        title="Preprocessed Spectra Comparison",
        xlabel=xlabel,
        ylabel="Intensity",
        xlim=xlim,
        alpha=0.6,
        linewidth=1,
    )

    return figures
