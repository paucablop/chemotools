"""Private helpers for deprecated public API aliases."""

from __future__ import annotations

import warnings
from typing import Any

from sklearn.utils._param_validation import Hidden, StrOptions

DEPRECATED_PARAMETER = "deprecated"


def deprecated_parameter_constraint() -> Hidden:
    """Return the validation constraint used for deprecated parameters."""
    return Hidden(StrOptions({DEPRECATED_PARAMETER}))


def is_deprecated_parameter(value: Any) -> bool:
    """Return whether a parameter still has the deprecated sentinel value."""
    return isinstance(value, str) and value == DEPRECATED_PARAMETER


def resolve_renamed_parameter(
    *,
    new_name: str,
    new_value: Any,
    new_default: Any,
    old_name: str,
    old_value: Any,
    stacklevel: int = 3,
) -> Any:
    """Resolve a deprecated parameter alias to its canonical replacement."""
    if is_deprecated_parameter(old_value):
        return new_value

    if not _matches_default(new_value, new_default):
        raise ValueError(
            f"Only one of `{new_name}` or deprecated `{old_name}` can be provided."
        )

    warnings.warn(
        f"`{old_name}` is deprecated and will be removed in a future release. "
        f"Use `{new_name}` instead.",
        FutureWarning,
        stacklevel=stacklevel,
    )
    return old_value


def _matches_default(value: Any, default: Any) -> bool:
    if value is default:
        return True

    if default is None:
        return value is None

    try:
        result = value == default
    except Exception:
        return False

    return isinstance(result, bool) and result
