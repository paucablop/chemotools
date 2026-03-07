"""Enhanced models for chemometrics with automatic diagnostics."""

import warnings

from chemotools.cross_decomposition import PLSRegression

__all__ = ["PLSRegression"]

# Show compatibility notices on module import
warnings.warn(
    "chemotools.models.PLSRegression has moved to "
    "chemotools.cross_decomposition.PLSRegression. Import from "
    "chemotools.cross_decomposition instead; chemotools.models is kept as a "
    "compatibility layer.",
    FutureWarning,
    stacklevel=2,
)

warnings.warn(
    "chemotools.cross_decomposition.PLSRegression extends sklearn's "
    "PLSRegression with explained_x_variance_ratio_ and "
    "explained_y_variance_ratio_ attributes. This feature is being "
    "contributed to scikit-learn (see PR #32722). Once it is available in "
    "scikit-learn, the chemotools wrapper may be deprecated. Track progress "
    "at: https://github.com/scikit-learn/scikit-learn/pull/32722",
    FutureWarning,
    stacklevel=2,
)
