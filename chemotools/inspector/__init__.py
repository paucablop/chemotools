"""Inspector module for model diagnostics and visualization."""

from ._pca_inspector import PCAInspector
from ._pls_regression_inspector import PLSRegressionInspector

__all__ = ["PCAInspector", "PLSRegressionInspector"]
