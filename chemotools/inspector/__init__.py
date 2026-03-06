"""Inspector module for model diagnostics and visualization."""

import warnings

from chemotools._optional import import_optional_dependency

import_optional_dependency(
    "matplotlib",
    caller_name="chemotools.inspector",
    extra_name="viz",
)

from ._pca_inspector import PCAInspector  # noqa: E402
from ._pls_regression_inspector import PLSRegressionInspector  # noqa: E402

__all__ = ["PCAInspector", "PLSRegressionInspector"]

# Show experimental warning on module import
warnings.warn(
    "The inspector module is experimental and under active development. "
    "The API may change in future versions. We welcome your feedback! "
    "Please report issues or suggestions at: "
    "https://github.com/paucablop/chemotools/issues",
    FutureWarning,
    stacklevel=2,
)
