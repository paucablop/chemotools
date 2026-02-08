"""Canonical type aliases for chemotools model types.

``EstimatorType`` represents the *extracted* estimator (always PCA or PLS).
``ModelInput`` represents the user-facing parameter that may also be a Pipeline.

We use sklearn's abstract bases rather than concrete classes because:
- The codebase accepts *any* PCA/PLS variant, not a fixed enumeration.
- Runtime isinstance dispatch in _validation.py requires nominal types.
- These bases are stable internal contracts in sklearn (used by
  _parameter_constraints throughout sklearn itself).

These aliases define the **input vocabulary** — what constructors accept.
Concrete subclasses that constrain the model further (e.g. PLS-only) should
narrow via class-level annotations and property overrides, *not* by introducing
new type aliases.  If sklearn ever restructures these bases, the fix is
localised to this file.
"""

from typing import Union

from sklearn.cross_decomposition._pls import _PLS
from sklearn.decomposition._base import _BasePCA
from sklearn.pipeline import Pipeline

EstimatorType = Union[_BasePCA, _PLS]
ModelInput = Union[_BasePCA, _PLS, Pipeline]
