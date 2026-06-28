"""
Predefined, picklable functions for use with :class:`MetadataFunctionTransformer`.

Each function follows the calling convention expected by
``MetadataFunctionTransformer``: ``X`` is the first positional argument and
any additional inputs are named keyword arguments declared in the transformer's
``metadata`` parameter.

Because these are module-level named functions they can be pickled, which is
required whenever a pipeline is serialised (``joblib.dump``) or cloned
(e.g. inside ``GridSearchCV``).

Examples
--------
>>> import numpy as np
>>> from chemotools.adaptation import MetadataFunctionTransformer
>>> from chemotools.adaptation.functions import subtract_reference
>>>
>>> rng = np.random.default_rng(0)
>>> X = rng.normal(size=(10, 50))
>>> reference = rng.normal(size=(1, 50))
>>>
>>> mft = MetadataFunctionTransformer(
...     func=subtract_reference, metadata=("reference",)
... )
>>> X_corrected = mft.fit_transform(X, reference=reference)
"""

import numpy as np
from sklearn.utils.validation import check_array


def _check_metadata(arr, name: str) -> np.ndarray:
    """Validate *arr* as a numeric input for a metadata argument.

    Scalars are returned as-is.  Array-like inputs are passed through
    :func:`sklearn.utils.validation.check_array` with ``ensure_2d=True``,
    which raises a ``ValueError`` for 1-D inputs and guides the user to
    reshape with ``.reshape(-1, 1)`` (per-sample) or ``.reshape(1, -1)``
    (shared).
    """
    arr = np.asarray(arr)
    if arr.ndim == 0:
        return arr  # scalar — valid for any function
    try:
        return check_array(arr, ensure_2d=True, dtype="numeric", input_name=name)
    except ValueError as exc:
        raise ValueError(f"Invalid metadata argument `{name}`: {exc}") from exc


def subtract_reference(X: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Subtract a reference spectrum from every row of ``X``.

    A common operation in spectroscopy for blank or solvent subtraction,
    background removal, or single-beam to double-beam correction.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Input spectra.

    reference : float or np.ndarray
        Reference to subtract.  Accepted shapes:

        * scalar ``float`` — subtracted from every element.
        * ``(1, n_features)`` — shared reference broadcast across all samples.
        * ``(n_samples, 1)`` — per-sample scalar subtracted from every feature.
        * ``(n_samples, n_features)`` — per-sample full spectrum.

        1-D inputs are rejected; use ``.reshape(1, -1)`` for a shared
        spectrum or ``.reshape(-1, 1)`` for a per-sample scalar.

    Returns
    -------
    X_corrected : np.ndarray of shape (n_samples, n_features)
        ``X - reference``.

    Examples
    --------
    >>> import numpy as np
    >>> from chemotools.adaptation.functions import subtract_reference
    >>> X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    >>> ref = np.array([[0.5, 0.5, 0.5]])
    >>> subtract_reference(X, ref)
    array([[0.5, 1.5, 2.5],
           [3.5, 4.5, 5.5]])
    """
    return X - _check_metadata(reference, "reference")


def divide_by_reference(X: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Divide every row of ``X`` by a reference spectrum.

    Useful for single-beam transmission corrections or ratiometric
    normalisation where each sample is divided by a simultaneously
    measured reference channel.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Input spectra.

    reference : float or np.ndarray
        Reference to divide by.  Accepted shapes:

        * scalar ``float`` — divides every element.
        * ``(1, n_features)`` — shared reference broadcast across all samples.
        * ``(n_samples, 1)`` — per-sample scalar divisor.
        * ``(n_samples, n_features)`` — per-sample full spectrum.

        1-D inputs are rejected; use ``.reshape(1, -1)`` for a shared
        spectrum or ``.reshape(-1, 1)`` for a per-sample scalar.  Zero
        values will produce ``inf`` or ``nan`` in the output.

    Returns
    -------
    X_corrected : np.ndarray of shape (n_samples, n_features)
        ``X / reference``.

    Examples
    --------
    >>> import numpy as np
    >>> from chemotools.adaptation.functions import divide_by_reference
    >>> X = np.array([[2.0, 4.0, 6.0], [8.0, 10.0, 12.0]])
    >>> ref = np.array([[2.0, 2.0, 2.0]])
    >>> divide_by_reference(X, ref)
    array([[1., 2., 3.],
           [4., 5., 6.]])
    """
    return X / _check_metadata(reference, "reference")


def scale_by_factor(X: np.ndarray, factor: float | np.ndarray) -> np.ndarray:
    """Multiply every row of ``X`` by a scalar or array factor.

    Useful for per-batch or per-sample intensity rescaling, e.g. correcting
    for integration time, path-length, or dilution differences.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Input spectra.

    factor : float or np.ndarray
        Scaling factor.  Accepted shapes:

        * scalar ``float`` — multiplies every element.
        * ``(n_samples, 1)`` — per-sample scalar factor.
        * ``(1, n_features)`` — per-feature factor shared across samples.

        1-D inputs are rejected; use ``.reshape(-1, 1)`` for per-sample
        or ``.reshape(1, -1)`` for per-feature.

    Returns
    -------
    X_scaled : np.ndarray of shape (n_samples, n_features)
        ``X * factor``.

    Examples
    --------
    >>> import numpy as np
    >>> from chemotools.adaptation.functions import scale_by_factor
    >>> X = np.array([[1.0, 2.0], [3.0, 4.0]])
    >>> scale_by_factor(X, np.array([[2.0], [0.5]]))
    array([[2. , 4. ],
           [1.5, 2. ]])
    """
    return X * _check_metadata(factor, "factor")


def add_offset(X: np.ndarray, offset: np.ndarray) -> np.ndarray:
    """Add a baseline offset to every row of ``X``.

    Useful for correcting instrument drift, dark-current baselines, or
    additive scatter effects that have been measured externally.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Input spectra.

    offset : float or np.ndarray
        Offset to add.  Accepted shapes:

        * scalar ``float`` — added to every element.
        * ``(n_samples, 1)`` — per-sample scalar offset.
        * ``(1, n_features)`` — per-feature offset shared across samples.
        * ``(n_samples, n_features)`` — per-sample full spectrum.

        1-D inputs are rejected; use ``.reshape(-1, 1)`` for per-sample
        or ``.reshape(1, -1)`` for per-feature.

    Returns
    -------
    X_shifted : np.ndarray of shape (n_samples, n_features)
        ``X + offset``.

    Examples
    --------
    >>> import numpy as np
    >>> from chemotools.adaptation.functions import add_offset
    >>> X = np.array([[1.0, 2.0, 3.0]])
    >>> add_offset(X, np.array([[0.1, 0.2, 0.3]]))
    array([[1.1, 2.2, 3.3]])
    """
    return X + _check_metadata(offset, "offset")
