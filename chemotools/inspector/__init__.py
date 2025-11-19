"""Inspector module for model diagnostics and visualization."""

import warnings

from ._pca_inspector import PCAInspector
from ._pls_regression_inspector import PLSRegressionInspector

__all__ = ["PCAInspector", "PLSRegressionInspector"]

# Show experimental warning on module import
warnings.warn(
    "The inspector module is experimental and under active development. "
    "The API may change in future versions. We welcome your feedback! "
    "Please report issues or suggestions at: "
    "https://github.com/paucablop/chemotools/issues/208",
    FutureWarning,
    stacklevel=2,
)
