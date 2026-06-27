from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.metadata_routing import MetadataRequest
from sklearn.utils.validation import check_is_fitted, validate_data

from chemotools._doc_mixin import DocLinkMixin
from chemotools._validation import check_metadata_signature

# Author: Pau Cabaneros
# Licence: MIT


class MetadataFunctionTransformer(DocLinkMixin, TransformerMixin, BaseEstimator):
    """
    A transformer that wraps an arbitrary callable and routes named metadata
    to it during ``transform``.

    This is useful when a preprocessing function requires per-sample or
    per-batch auxiliary information (e.g. reference spectra, wavelength
    arrays) that must be threaded through a scikit-learn ``Pipeline`` via
    the metadata-routing API.

    Parameters
    ----------
    func : Callable[..., np.ndarray]
        The function to apply during ``transform``.  It must accept ``X``
        as its first positional argument and return a 2-D ``np.ndarray``
        of the same shape.

    metadata : Sequence[str], default=()
        Names of the keyword arguments that ``func`` expects in addition to
        ``X``.  Only keys listed here are extracted from the ``**metadata``
        dict and forwarded to ``func``; any extra keys are silently ignored.

    validate : bool, default=True
        If ``True``, ``transform`` calls ``validate_data`` to enforce the scikit-learn 
        contract.  Set to ``False`` when ``func`` handles its own input validation or 
        when used outside a fitted pipeline.

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

        y : None
            Ignored; present for API compatibility.

        **metadata : Any
            Additional metadata forwarded by the pipeline; not used during
            fitting.

        Returns
        -------
        self : MetadataFunctionTransformer
        """
        # Validate the input parameters
        self._validate_params()

        # Validate the X data
        validate_data(self, X, ensure_2d=True, reset=True, dtype="numeric")

        # Validate that the provided metadata keys match the function signature
        check_metadata_signature(
            self.func, self.metadata, estimator_name=type(self).__name__
        )
        return self

    def transform(self, X: Any, **metadata: Any) -> np.ndarray:
        """
        Apply ``func`` to ``X``, forwarding any requested metadata keys.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to transform.

        **metadata : Any
            Keyword arguments passed to ``func``.  Only keys listed in
            ``self.metadata`` are forwarded.

        Returns
        -------
        X_transformed : np.ndarray of shape (n_samples, n_features)
            The result of calling ``func(X, **kwargs)``.
        """

        # Ensures .fit() was called by checking for trailing underscore attributes
        check_is_fitted(self)

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

        y : None
            Ignored; present for API compatibility.

        **metadata : Any
            Keyword arguments forwarded to both ``fit`` and ``transform``.

        Returns
        -------
        X_transformed : np.ndarray of shape (n_samples, n_features)
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
