"""Canonical type aliases for chemotools model types.

``EstimatorType`` represents the *extracted* estimator (always PCA or PLS).
``PLSEstimatorType`` is the PLS-only narrowing of it.
``ModelInput`` represents the user-facing parameter that may also be a Pipeline.

The PCA/NIPALS-PLS side uses sklearn's abstract bases rather than concrete
classes because:
- The codebase accepts *any* PCA/PLS variant, not a fixed enumeration.
- Runtime isinstance dispatch in _validation.py requires nominal types.
- These bases are stable internal contracts in sklearn (used by
  _parameter_constraints throughout sklearn itself).

chemotools' own :class:`~chemotools.regression.PLSRegression` is backed by
ikpls rather than sklearn's NIPALS, so it is *not* a ``_PLS`` subclass; it is
included explicitly. ``PLS_TYPES`` is the runtime tuple for isinstance
dispatch on "any PLS model" — every member exposes the fitted-attribute
surface the rest of the codebase relies on (``x_scores_``, ``x_loadings_``,
``coef_``, ``n_components``, ``transform``/``inverse_transform``).

These aliases define the **input vocabulary** — what constructors accept.
Concrete subclasses that constrain the model further should narrow via
class-level annotations and property overrides. If sklearn ever restructures
its bases, the fix is localised to this file.
"""

from typing import Tuple, Type, Union

from sklearn.cross_decomposition._pls import _PLS
from sklearn.decomposition._base import _BasePCA
from sklearn.pipeline import Pipeline

from chemotools.regression import PLSRegression

# Runtime tuple for isinstance dispatch on "any PLS model".
PLS_TYPES: Tuple[Type, ...] = (_PLS, PLSRegression)

PLSEstimatorType = Union[_PLS, PLSRegression]
EstimatorType = Union[_BasePCA, _PLS, PLSRegression]
ModelInput = Union[_BasePCA, _PLS, PLSRegression, Pipeline]
