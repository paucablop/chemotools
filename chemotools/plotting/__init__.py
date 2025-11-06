"""Plotting utilities and display protocols for chemotools visualizations."""

from chemotools.plotting._display import Display, is_displayable
from chemotools.plotting._utilities import (
    setup_figure,
    get_colors_from_labels,
    add_confidence_ellipse,
    annotate_points,
)
from chemotools.plotting._spectrum import SpectrumPlot
from chemotools.plotting._scores import ScoresPlot
from chemotools.plotting._styles import DATASET_COLORS
from chemotools.plotting._inspector_plots import (
    LoadingsPlot,
    ExplainedVariancePlot,
)

__all__ = [
    # Protocols
    "Display",
    "is_displayable",
    # Plot classes
    "SpectrumPlot",
    # Utilities
    "setup_figure",
    "get_colors_from_labels",
    "add_confidence_ellipse",
    "annotate_points",
    # Constants
    "DATASET_COLORS",
    "ScoresPlot",
    "LoadingsPlot",
    "ExplainedVariancePlot",
]
