"""Canonical type aliases for chemotools model types.

``EstimatorType`` represents the *extracted* estimator (always PCA or PLS).
``ModelInput`` represents the user-facing parameter that may also be a Pipeline.
"""

from typing import Union

from sklearn.cross_decomposition._pls import _PLS
from sklearn.decomposition._base import _BasePCA
from sklearn.pipeline import Pipeline

EstimatorType = Union[_BasePCA, _PLS]
ModelInput = Union[_BasePCA, _PLS, Pipeline]
