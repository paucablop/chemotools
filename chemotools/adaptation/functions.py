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


def subtract_reference(X: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Subtract a reference spectrum from every row of ``X``.

    A common operation in spectroscopy for blank or solvent subtraction,
    background removal, or single-beam to double-beam correction.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Input spectra.

    reference : np.ndarray of shape (1, n_features) or (n_samples, n_features)
        Reference spectrum (or per-sample references) to subtract.
        Must broadcast against ``X``.

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
    return X - reference


def divide_by_reference(X: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Divide every row of ``X`` by a reference spectrum.

    Useful for single-beam transmission corrections or ratiometric
    normalisation where each sample is divided by a simultaneously
    measured reference channel.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Input spectra.

    reference : np.ndarray of shape (1, n_features) or (n_samples, n_features)
        Reference spectrum (or per-sample references) to divide by.
        Must broadcast against ``X``.  Zero values in ``reference`` will
        produce ``inf`` or ``nan`` in the output.

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
    return X / reference


def scale_by_factor(X: np.ndarray, factor: float | np.ndarray) -> np.ndarray:
    """Multiply every row of ``X`` by a scalar or array factor.

    Useful for per-batch or per-sample intensity rescaling, e.g. correcting
    for integration time, path-length, or dilution differences.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Input spectra.

    factor : float or np.ndarray of shape (n_samples, 1) or (1, n_features)
        Scaling factor.  Must broadcast against ``X``.

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
    return X * factor


def add_offset(X: np.ndarray, offset: np.ndarray) -> np.ndarray:
    """Add a baseline offset to every row of ``X``.

    Useful for correcting instrument drift, dark-current baselines, or
    additive scatter effects that have been measured externally.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Input spectra.

    offset : np.ndarray of shape (1, n_features) or (n_samples, n_features)
        Offset to add.  Must broadcast against ``X``.

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
    return X + offset
