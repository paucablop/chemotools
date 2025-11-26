"""Plotting utilities and display protocols for chemotools visualizations."""

import warnings

from chemotools.plotting._base import BasePlot, Display, is_displayable
from chemotools.plotting._utils import (
    setup_figure,
    get_colors_from_labels,
    add_confidence_ellipse,
    annotate_points,
    calculate_ylim_for_xlim,
)
from chemotools.plotting._spectra import SpectraPlot
from chemotools.plotting._feature_selection import FeatureSelectionPlot
from chemotools.plotting._scores import ScoresPlot
from chemotools.plotting._loadings import LoadingsPlot
from chemotools.plotting._distances import DistancesPlot
from chemotools.plotting._styles import DATASET_COLORS, DATASET_MARKERS
from chemotools.plotting._explained_variance import ExplainedVariancePlot
from chemotools.plotting._y_residuals import YResidualsPlot
from chemotools.plotting._qq_plot import QQPlot
from chemotools.plotting._residual_distribution import ResidualDistributionPlot
from chemotools.plotting._predicted_vs_actual import PredictedVsActualPlot

__all__ = [
    # Protocols
    "Display",
    "is_displayable",
    "BasePlot",
    # Plot classes
    "SpectraPlot",
    "FeatureSelectionPlot",
    "ScoresPlot",
    "LoadingsPlot",
    "DistancesPlot",
    "ExplainedVariancePlot",
    "YResidualsPlot",
    "QQPlot",
    "ResidualDistributionPlot",
    "PredictedVsActualPlot",
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

# Show experimental warning on module import
warnings.warn(
    "The plotting module is experimental and under active development. "
    "The API may change in future versions. We welcome your feedback! "
    "Please report issues or suggestions at: "
    "https://github.com/paucablop/chemotools/issues/208",
    FutureWarning,
    stacklevel=2,
)
