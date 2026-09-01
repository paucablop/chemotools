"""Validation utilities for metadata-aware transformation functions."""

import inspect
from collections import Counter
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
    """Validate a metadata function on representative data.

    The function is called once as ``func(X_checked, **metadata)``. Its output
    is validated as a finite, numeric, 2-D array that preserves the number of
    input samples and, by default, the number of input features.

    Parameters
    ----------
    func : callable
        Function invoked as ``func(X_checked, **metadata)``.

    X : array-like of shape (n_samples, n_features)
        Representative input data.

    metadata : mapping of str to object, default=None
        Representative keyword arguments passed to ``func``. Use ``None`` when
        the function requires no metadata. The names ``"X"`` and ``"y"`` are
        reserved by the estimator API and cannot be used as metadata keys.

    preserve_features : bool, default=True
        If ``True``, require the output to have the same number of features as
        ``X``. The number of samples is always required to match.

    Returns
    -------
    result : np.ndarray
        Validated output produced by ``func``.

    Raises
    ------
    TypeError
        If ``func`` is not callable or its signature cannot be inspected.
    ValueError
        If ``X`` is invalid, ``func`` cannot be called with the supplied
        arguments, its output is not a finite numeric 2-D array, or its output
        does not preserve the required dimensions.

    Notes
    -----
    This check executes user-provided code. Exceptions raised by ``func`` are
    propagated unchanged, and functions with side effects will perform those
    side effects once. The validated input and output may be converted to
    NumPy arrays by :func:`sklearn.utils.validation.check_array`.

    Examples
    --------
    >>> import numpy as np
    >>> from chemotools.adaptation.functions import subtract_reference
    >>> from chemotools.adaptation.validation import check_metadata_function
    >>> X = np.array([[1.0, 2.0], [3.0, 4.0]])
    >>> reference = np.array([[0.5, 0.5]])
    >>> check_metadata_function(
    ...     subtract_reference, X, metadata={"reference": reference}
    ... )
    array([[0.5, 1.5],
           [2.5, 3.5]])

    See Also
    --------
    chemotools.adaptation.MetadataFunctionTransformer : Wrap a function for
        scikit-learn metadata routing.
    """

    # Check that the function is callable
    if not callable(func):
        raise TypeError(f"`func` must be callable. Got {func!r}.")

    # Check the X array is valid
    X_checked = check_array(X, ensure_2d=True, dtype="numeric")

    # Pass metadata **kwarg as a dict, or empty dict if None.
    # This is needed for signature checking.
    metadata = {} if metadata is None else dict(metadata)
    _validate_metadata_names(list(metadata), "check_metadata_function")

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


def _validate_metadata_names(metadata: Sequence[str], estimator_name: str) -> list[str]:
    """Return validated, unique metadata names."""
    metadata_names = list(metadata)
    invalid_names = [name for name in metadata_names if not isinstance(name, str)]
    if invalid_names:
        raise TypeError(
            f"[{estimator_name}] All entries in `metadata` must be strings. "
            f"Got invalid entries: {invalid_names}"
        )

    duplicate_names = sorted(
        name for name, count in Counter(metadata_names).items() if count > 1
    )
    if duplicate_names:
        raise ValueError(
            f"[{estimator_name}] Entries in `metadata` must be unique. "
            f"Got duplicates: {duplicate_names}"
        )

    for reserved_name in ("X", "y"):
        if reserved_name in metadata_names:
            raise ValueError(
                f"[{estimator_name}] `{reserved_name}` cannot be requested in "
                "`metadata` because it is reserved by the estimator API."
            )

    return metadata_names


def _check_metadata_signature(
    fn: Callable, metadata: Sequence[str], estimator_name: str = "Estimator"
) -> None:
    """Validate compatibility with ``fn(X, **metadata)`` without executing ``fn``."""
    metadata_names = _validate_metadata_names(metadata, estimator_name)

    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"[{estimator_name}] Could not inspect the signature of {fn!r}."
        ) from exc

    params = sig.parameters

    has_var_keyword = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
    )

    # 1. Reject metadata keys that correspond to positional-only parameters:
    #    the transformer forwards metadata as keyword arguments, so positional-only
    #    parameters can never be reached that way.
    positional_only_keys = [
        key
        for key in metadata_names
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
        missing_in_func = [key for key in metadata_names if key not in params]
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
                and param_name not in metadata_names
            ):
                missing_in_metadata.append(param_name)

        if missing_in_metadata:
            raise ValueError(
                f"[{estimator_name}] The function "
                f"'{getattr(fn, '__name__', repr(fn))}' requires the following "
                f"arguments without defaults, which are missing from "
                f"`metadata`: {missing_in_metadata}"
            )

    placeholder = object()
    metadata_placeholders = dict.fromkeys(metadata_names, placeholder)
    try:
        sig.bind(placeholder, **metadata_placeholders)
    except TypeError as exc:
        raise ValueError(
            f"[{estimator_name}] The function "
            f"'{getattr(fn, '__name__', repr(fn))}' cannot be called as "
            f"`func(X, **metadata)`: {exc}"
        ) from exc
