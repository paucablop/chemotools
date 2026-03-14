"""Enhanced models for chemometrics with automatic diagnostics."""

import warnings

from chemotools.regression import PLSRegression

__all__ = ["PLSRegression"]

# Show compatibility notices on module import
warnings.warn(
    "chemotools.models.PLSRegression has moved to "
    "chemotools.regression.PLSRegression. Import from "
    "chemotools.regression instead; chemotools.models is kept as a "
    "compatibility layer.",
    FutureWarning,
    stacklevel=2,
)
