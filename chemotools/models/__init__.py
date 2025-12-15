"""Enhanced models for chemometrics with automatic diagnostics."""

import warnings

from chemotools.models._cross_decomposition import PLSRegression

__all__ = ["PLSRegression"]

# Show deprecation notice on module import
warnings.warn(
    "chemotools.models.PLSRegression extends sklearn's PLSRegression with "
    "explained_x_variance_ratio_ and explained_y_variance_ratio_ attributes. "
    "This feature is being contributed to scikit-learn (see PR #32722). "
    "Once available in sklearn, this module may be deprecated. "
    "Track progress at: https://github.com/scikit-learn/scikit-learn/pull/32722",
    FutureWarning,
    stacklevel=2,
)
