"""Plotting utilities and display protocols for chemotools visualizations."""

from chemotools.plotting._display import Display, is_displayable
from chemotools.plotting._utilities import (
    setup_figure,
    get_colors_from_labels,
    add_confidence_ellipse,
    annotate_points,
    calculate_ylim_for_xlim,
)
from chemotools.plotting._spectrum import SpectrumPlot
from chemotools.plotting._scores import ScoresPlot
from chemotools.plotting._loadings import LoadingsPlot
from chemotools.plotting._distances import DistancesPlot
from chemotools.plotting._styles import DATASET_COLORS, DATASET_MARKERS
from chemotools.plotting._explained_variance import ExplainedVariancePlot
from chemotools.plotting._y_residuals import YResidualsPlot
from chemotools.plotting._qq_plot import QQPlot
from chemotools.plotting._residual_distribution import ResidualDistributionPlot

__all__ = [
    # Protocols
    "Display",
    "is_displayable",
    # Plot classes
    "SpectrumPlot",
    "ScoresPlot",
    "LoadingsPlot",
    "DistancesPlot",
    "ExplainedVariancePlot",
    "YResidualsPlot",
    "QQPlot",
    "ResidualDistributionPlot",
    # Utilities
    "setup_figure",
    "get_colors_from_labels",
    "add_confidence_ellipse",
    "annotate_points",
    "calculate_ylim_for_xlim",
    # Constants
    "DATASET_COLORS",
    "DATASET_MARKERS",
]
