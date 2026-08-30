"""
Shared model validation for metadata-based transformers and function routing.
"""

import inspect
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
from sklearn.utils.validation import check_array


def check_metadata_function(
    func: Callable[..., Any],
    X,
    *,
    metadata: Mapping[str, Any] | None = None,
    preserve_features: bool = True,
) -> np.ndarray:
    """Validate a metadata function by executing it on representative data and return
    its validated output.

    Parameters
    ----------
    func : callable
        Function invoked as ``func(X, **metadata)``.

    X : array-like of shape (n_samples, n_features)
        Representative input data.

    metadata : mapping of str to object, default=None
        Representative metadata passed to the function.

    preserve_features : bool, default=True
        Whether the output must preserve the number of features.

    Returns
    -------
    result : np.ndarray
        Validated result produced by the function.
    """

    # Check that the function is callable
    if not callable(func):
        raise TypeError(f"`func` must be callable. Got {func!r}.")

    # Check the X array is valid
    X_checked = check_array(X, ensure_2d=True, dtype="numeric")

    # Pass metadata **kwarg as a dict, or empty dict if None.
    # This is needed for signature checking.
    metadata = {} if metadata is None else dict(metadata)

    # Check the function's signature
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"Could not inspect the signature of {func!r}.") from exc

    # Ensure that metadata keys match the function's signature and that required
    # parameters are present.
    try:
        signature.bind(X_checked, **metadata)
    except TypeError as exc:
        name = getattr(func, "__name__", repr(func))
        raise ValueError(
            f"`{name}` cannot be called as `func(X, **metadata)`: {exc}"
        ) from exc

    # Call the function and validate the output
    output_checked = func(X_checked, **metadata)

    try:
        output_checked = check_array(
            output_checked,
            ensure_2d=True,
            dtype="numeric",
            input_name="func output",
        )
    except ValueError as exc:
        name = getattr(func, "__name__", repr(func))
        raise ValueError(f"`{name}` must return a numeric 2-D array: {exc}") from exc

    if output_checked.shape[0] != X_checked.shape[0]:
        raise ValueError(
            "`func` changed the number of samples: "
            f"expected {X_checked.shape[0]}, got {output_checked.shape[0]}."
        )

    if preserve_features and output_checked.shape[1] != X_checked.shape[1]:
        raise ValueError(
            "`func` changed the number of features: "
            f"expected {X_checked.shape[1]}, got {output_checked.shape[1]}."
        )

    return output_checked


def _check_metadata_signature(
    fn: Callable, metadata: Sequence[str], estimator_name: str = "Estimator"
) -> None:
    """
    Validates that a function's signature matches the requested routing metadata.
    """
    sig = inspect.signature(fn)
    params = sig.parameters

    has_var_keyword = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
    )

    # 1. Reject metadata keys that correspond to positional-only parameters:
    #    the transformer forwards metadata as keyword arguments, so positional-only
    #    parameters can never be reached that way.
    positional_only_keys = [
        key
        for key in metadata
        if key in params and params[key].kind == inspect.Parameter.POSITIONAL_ONLY
    ]
    if positional_only_keys:
        raise ValueError(
            f"[{estimator_name}] The following keys in `metadata` correspond to "
            f"positional-only parameters of "
            f"'{getattr(fn, '__name__', repr(fn))}' and cannot be forwarded as "
            f"keyword arguments: {positional_only_keys}"
        )

    # 2. Check for orphaned metadata (requested, but function can't accept it).
    #    Skipped when **kwargs is present — the function absorbs any extra key.
    if not has_var_keyword:
        missing_in_func = [key for key in metadata if key not in params]
        if missing_in_func:
            raise ValueError(
                f"[{estimator_name}] The function "
                f"'{getattr(fn, '__name__', repr(fn))}' does not accept the "
                f"following arguments requested in `metadata`: {missing_in_func}"
            )

    # 3. Check for starved function (function requires it, but not in metadata).
    #    Always run — even when **kwargs is present, required positional parameters
    #    must be declared in `metadata` so they are forwarded correctly.
    param_names = list(params.keys())
    if len(param_names) > 1:
        expected_kwargs = param_names[1:]
        missing_in_metadata = []

        for param_name in expected_kwargs:
            p = params[param_name]
            if (
                p.default == inspect.Parameter.empty
                and p.kind
                not in (
                    inspect.Parameter.VAR_POSITIONAL,
                    inspect.Parameter.VAR_KEYWORD,
                    inspect.Parameter.POSITIONAL_ONLY,
                )
                and param_name not in metadata
            ):
                missing_in_metadata.append(param_name)

        if missing_in_metadata:
            raise ValueError(
                f"[{estimator_name}] The function "
                f"'{getattr(fn, '__name__', repr(fn))}' requires the following "
                f"arguments without defaults, which are missing from "
                f"`metadata`: {missing_in_metadata}"
            )
