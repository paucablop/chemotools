"""Compatibility layer – import from chemotools.projection or chemotools.regression."""

import warnings

from chemotools.projection import (
    ExternalParameterOrthogonalization,
    OrthogonalSignalCorrection,
)
from chemotools.regression import PLSRegression

__all__ = [
    "ExternalParameterOrthogonalization",
    "OrthogonalSignalCorrection",
    "PLSRegression",
]

warnings.warn(
    "chemotools.cross_decomposition has been split into "
    "chemotools.projection (EPO, OSC) and chemotools.regression (PLSRegression). "
    "Please update your imports. chemotools.cross_decomposition will be removed "
    "in a future release.",
    FutureWarning,
    stacklevel=2,
)
