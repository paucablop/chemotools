"""Helper functions for creating preprocessing step plots.

This module contains plotting utilities used by
:class:`~chemotools.inspector.PreprocessingInspector` to generate
per-step spectral visualisations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from matplotlib.figure import Figure

from chemotools.plotting import SpectraPlot


def create_preprocessing_step_plot(
    X: np.ndarray,
    x_axis: np.ndarray,
    title: str,
    xlabel: str,
    color_values: Optional[np.ndarray] = None,
    xlim: Optional[Tuple[float, float]] = None,
    figsize: Tuple[float, float] = (12, 5),
    color_mode: Optional[Literal["continuous", "categorical"]] = None,
) -> "Figure":
    """Create a single spectra plot for one preprocessing step.

    Parameters
    ----------
    X : np.ndarray
        Transformed spectra of shape ``(n_samples, n_features)``.
    x_axis : np.ndarray
        X-axis values (e.g. wavenumbers, wavelengths, or feature indices).
    title : str
        Plot title.
    xlabel : str
        Label for the x-axis.
    color_values : np.ndarray, optional
        Per-sample colour values.  If ``None`` no colouring is applied.
    xlim : tuple of float, optional
        X-axis limits for zooming.
    figsize : tuple of float, default=(12, 5)
        Figure size ``(width, height)`` in inches.
    color_mode : {``'continuous'``, ``'categorical'``}, optional
        Override automatic colour-mode detection.

    Returns
    -------
    Figure
        The matplotlib ``Figure`` object.
    """
    # Suppress per-spectrum labels when no colour reference is given
    suppress_labels = color_values is None
    empty_labels = [""] * X.shape[0] if suppress_labels else None

    plot = SpectraPlot(
        x=x_axis,
        y=X,
        color_by=color_values,
        labels=empty_labels,
        color_mode=color_mode,
    )

    fig = plot.show(
        figsize=figsize,
        title=title,
        xlabel=xlabel,
        ylabel="Intensity",
        xlim=xlim,
    )

    return fig
