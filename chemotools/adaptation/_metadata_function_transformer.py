"""
The :mod:`chemotools.adaptation._metadata_function_transformer`
module implements the MetadataFunctionTransformer.
"""

# Author: Pau Cabaneros
# Licence: MIT

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.metadata_routing import MetadataRequest
from sklearn.utils.validation import check_is_fitted, validate_data

from chemotools._doc_mixin import DocLinkMixin
from chemotools.adaptation.validation import _check_metadata_signature


class MetadataFunctionTransformer(DocLinkMixin, TransformerMixin, BaseEstimator):
    """
    Apply a callable that consumes feature data and routed metadata.

    This is useful when a preprocessing function requires per-sample or
    per-batch auxiliary information (e.g. reference spectra, wavelength
    arrays) that must be threaded through a scikit-learn ``Pipeline`` via
    the metadata-routing API.

    Parameters
    ----------
    func : Callable[..., np.ndarray]
        Function applied during ``transform``. It must accept ``X`` as its
        first positional argument and each requested metadata value as a
        keyword argument. The function is responsible for returning a numeric,
        2-D ``np.ndarray`` with the same number of samples and features as ``X``.
        Use :func:`chemotools.adaptation.validation.check_metadata_function`
        to verify this contract on representative data.

    metadata : Sequence[str], default=()
        Names of the keyword arguments requested through scikit-learn metadata
        routing and forwarded to ``func``. Keys passed to ``transform`` but not
        listed here are ignored. Every required keyword argument of ``func``
        must be listed, and the corresponding value must be supplied when
        calling ``transform``. The name ``"y"`` is reserved by the estimator
        API and cannot be requested as metadata.

    validate : bool, default=True
        If ``True``, validate ``X`` as a numeric, 2-D array during ``fit`` and
        ``transform``, and require the number of features to remain consistent.
        This does not validate the output of ``func``. If ``False``, pass ``X``
        to ``func`` unchanged. In both cases, ``fit`` must be called before
        ``transform``.

    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during ``fit``.  Only set when
        ``validate=True``.

    Examples
    --------
    >>> import numpy as np
    >>> from chemotools.adaptation import MetadataFunctionTransformer
    >>>
    >>> def subtract_reference(X, reference):
    ...     return X - reference
    >>>
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(10, 50))
    >>> reference = rng.normal(size=(1, 50))
    >>>
    >>> mft = MetadataFunctionTransformer(
    ...     func=subtract_reference, metadata=("reference",)
    ... )
    >>> X_transf = mft.fit_transform(X, reference=reference)

    Notes
    -----
    To route metadata through a scikit-learn ``Pipeline``, enable metadata
    routing globally with ``sklearn.set_config(enable_metadata_routing=True)``.
    This is not required when calling this estimator directly.

    The callable signature is validated during ``fit``, but the callable is
    executed only during ``transform``. Consequently, errors that depend on
    metadata values or callable execution are raised during ``transform``.
    Use :func:`chemotools.adaptation.validation.check_metadata_function` to
    validate a callable eagerly on representative inputs.

    See Also
    --------
    chemotools.adaptation.validation.check_metadata_function : Validate a custom
        metadata function on representative inputs.
    chemotools.adaptation.functions : Predefined metadata-aware functions.
    """

    _parameter_constraints: dict = {
        "func": [callable],
        "metadata": ["array-like"],
        "validate": ["boolean"],
    }

    def __init__(
        self,
        func: Callable[..., np.ndarray],
        metadata: Sequence[str] = (),
        validate: bool = True,
    ):
        self.func = func
        self.metadata = metadata
        self.validate = validate

    def fit(self, X, y=None, **metadata: Any):
        """
        Fit the transformer by recording the number of input features.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : array-like, default=None
            Ignored. Present for API compatibility.

        **metadata : Any
            Metadata accepted for routing compatibility. It is not forwarded
            to ``func`` during fitting.

        Returns
        -------
        self : MetadataFunctionTransformer
        """
        # Validate the input parameters
        self._validate_params()

        # Validate the X data
        if self.validate:
            validate_data(self, X, ensure_2d=True, reset=True, dtype="numeric")

        # Validate that the provided metadata keys match the function signature
        _check_metadata_signature(
            self.func, self.metadata, estimator_name=type(self).__name__
        )
        self._is_fitted = True
        return self

    def transform(self, X: Any, **metadata: Any) -> np.ndarray:
        """
        Apply ``func`` to ``X``, forwarding any requested metadata keys.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to transform.

        **metadata : Any
            Metadata values passed to ``func`` as keyword arguments. Only keys
            listed in ``self.metadata`` are forwarded.

        Returns
        -------
        X_transformed : np.ndarray of shape (n_samples, n_features)
            Result returned by ``func(X, **kwargs)``. Output type and shape are
            the responsibility of ``func`` and are not validated here.
        """
        # Ensures .fit() was called
        check_is_fitted(self, "_is_fitted")

        # Validates X and ensures it has the same number of features as seen in fit
        if self.validate:
            X = validate_data(self, X, ensure_2d=True, reset=False, dtype="numeric")

        # Extract metadata keys if they exist
        kwargs = {key: metadata[key] for key in self.metadata if key in metadata}
        return self.func(X, **kwargs)

    def fit_transform(self, X, y=None, **metadata: Any) -> np.ndarray:
        """
        Fit and transform in a single step.

        Overrides the default ``TransformerMixin.fit_transform`` to ensure
        that ``**metadata`` is forwarded to both ``fit`` and ``transform``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : array-like, default=None
            Ignored. Present for API compatibility.

        **metadata : Any
            Keyword arguments forwarded to both ``fit`` and ``transform``.

        Returns
        -------
        X_transformed : np.ndarray of shape (n_samples, n_features)
            Result returned by ``func``.
        """
        # Explicitly pass metadata to BOTH fit and transform
        return self.fit(X, y, **metadata).transform(X, **metadata)

    def get_metadata_routing(self):
        """
        Return the metadata routing configuration for this transformer.

        Registers each name in ``self.metadata`` as a requested parameter
        for both ``fit`` and ``transform``, enabling sklearn's metadata
        routing to propagate them through a ``Pipeline``.

        Returns
        -------
        request : MetadataRequest
            The populated routing object.
        """
        request = MetadataRequest(owner=self.__class__.__name__)
        for key in self.metadata:
            request.fit.add_request(param=key, alias=True)
            request.transform.add_request(param=key, alias=True)
        return request
