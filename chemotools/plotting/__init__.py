"""Plotting utilities and display protocols for chemotools visualizations."""

import warnings

from chemotools._optional import import_optional_dependency

import_optional_dependency(
    "matplotlib",
    caller_name="chemotools.plotting",
    extra_name="viz",
)

from chemotools.plotting._base import BasePlot, Display, is_displayable  # noqa: E402
from chemotools.plotting._distances import DistancesPlot  # noqa: E402
from chemotools.plotting._explained_variance import ExplainedVariancePlot  # noqa: E402
from chemotools.plotting._feature_selection import FeatureSelectionPlot  # noqa: E402
from chemotools.plotting._loadings import LoadingsPlot  # noqa: E402
from chemotools.plotting._predicted_vs_actual import PredictedVsActualPlot  # noqa: E402
from chemotools.plotting._qq_plot import QQPlot  # noqa: E402
from chemotools.plotting._residual_distribution import (  # noqa: E402
    ResidualDistributionPlot,
)
from chemotools.plotting._scores import ScoresPlot  # noqa: E402
from chemotools.plotting._spectra import SpectraPlot  # noqa: E402
from chemotools.plotting._styles import DATASET_COLORS, DATASET_MARKERS  # noqa: E402
from chemotools.plotting._utils import (  # noqa: E402
    add_confidence_ellipse,
    annotate_points,
    calculate_ylim_for_xlim,
    get_colors_from_labels,
    setup_figure,
)
from chemotools.plotting._y_residuals import YResidualsPlot  # noqa: E402

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
    "https://github.com/paucablop/chemotools/issues",
    FutureWarning,
    stacklevel=2,
)
