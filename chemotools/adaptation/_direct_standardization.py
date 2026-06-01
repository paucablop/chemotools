"""
The :mod: `chemotools.adaptation._direct_standardization`
module implements the Direct Standardization (DS) transformer
"""

# Authors: Ruggero Guerrini
# License: MIT

import warnings

import numpy as np
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin
from sklearn.utils.validation import (
    check_is_fitted,
    validate_data,
)

from chemotools._doc_mixin import DocLinkMixin


class DirectStandardization(
    DocLinkMixin, OneToOneFeatureMixin, TransformerMixin, BaseEstimator
):
    """
    Direct Standardization (DS) is a transformer used for domain adaptation (calibration
    ) applications. The transformer uses least squares to find a linear map from
    the target instrument space to the source instrument space, following the
    implementation by [1]_.

    Attributes
    ----------
    V_ds_ : np.ndarray of shape (n_features, n_transfer_samples)
        Right singular vectors of the target transfer data X, used as the
        column basis of the low-rank transformation. Empty (shape
        ``(n_features, 0)``) when no ``X_source`` was provided.

    B_ds_ : np.ndarray of shape (n_transfer_samples, n_features)
        Projection of the source transfer data onto the singular basis.  The
        full transformation is ``X @ V_ds_ @ B_ds_``, which is algebraically
        equivalent to ``X @ T_`` but requires only
        ``O(n_features × n_transfer_samples)`` storage instead of
        ``O(n_features²)``.

    x_source_provided_ : bool
        Boolean value to flag if X_source was provided during fitting

    Raises
    ------
    ValueError
        If X and X_source do not have the same shape.

    See Also
    --------
    PiecewiseDirectStandardization : Localized version using windowed PLS regression.

    References
    ----------
    .. [1] Wang, Yongdong., Veltkamp, D. J., & Kowalski, B. R. (1991),
        Multivariate instrument standardization,
        Analytical Chemistry, 63(23), Pages 2750–2756,
        https://doi.org/10.1021/ac00023a016.

    Examples
    --------
    **Basic usage**
    >>> import numpy as np
    >>> from chemotools.adaptation import DirectStandardization
    >>>
    >>> rng = np.random.default_rng(17)
    >>> X_source = rng.normal(size=(100, 20))
    >>> X_target = X_source * 2 - rng.normal(size=(100, 20)) * 0.02
    >>>
    >>> ds = DirectStandardization().fit(X_target, X_source=X_source)
    >>> X_transf = ds.transform(X_target)

    """

    # Fitted attributes (set during fit, typed for type checkers)
    n_features_in_: int
    V_ds_: np.ndarray
    B_ds_: np.ndarray
    x_source_provided_: bool

    _parameter_constraints: dict = {}

    def fit(
        self, X: np.ndarray, y=None, *, X_source: np.ndarray | None = None
    ) -> "DirectStandardization":
        """
        Fit the Direct Standardization model.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Data from the target instrument.

        y : None
            Ignored to align with API.

        X_source : np.ndarray of shape (n_samples, n_features), optional
            Data from the source instrument. If None, the transformer defaults to
            an identity transformation.

        Returns
        -------
        self : DirectStandardization
        """
        # Validate the input parameters
        self._validate_params()
        # Check that X is a 2D array and has only finite values
        X = validate_data(self, X, ensure_2d=True, reset=True, dtype=np.float64)

        # If X_source is None, default to identity transformation
        if X_source is None:
            warnings.warn(
                "X_source is None, the transformer will act as an identity "
                "transformation."
            )
            self.V_ds_ = np.empty((X.shape[1], 0))
            self.B_ds_ = np.empty((0, X.shape[1]))
            self.x_source_provided_ = False

            return self

        # Check that X_source is a 2D array and has only finite values
        X_source = validate_data(
            self, X_source, ensure_2d=True, reset=False, dtype=np.float64
        )

        # Check consistency between X and X_source
        if X_source.shape != X.shape:
            raise ValueError(
                f"X and X_source must have the same shape, "
                f"got X={X.shape} and X_source={X_source.shape}."
            )

        # Low-rank factorisation: T_* = V_ds_ @ B_ds_
        # rank(T_*) <= n_transfer_samples << n_features, so storing the two
        # thin factors is O(n_features * r) vs O(n_features^2) for the full
        # matrix, and transform becomes two small matmuls instead of one huge one.
        U, s, Vt = np.linalg.svd(X, full_matrices=False)
        self.V_ds_ = Vt.T  # (n_features, r)
        self.B_ds_ = (1.0 / s[:, None]) * (U.T @ X_source)  # (r, n_features)
        self.x_source_provided_ = True

        return self

    def transform(self, X) -> np.ndarray:
        """
        Transform the data from the target space to the source space using the map
        ``self.T_``.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            The input data to transform

        Returns
        -------
        X_transf : np.ndarray of shape (n_samples, n_features)
            The data transformed
        """
        # Check that the estimator is fitted
        check_is_fitted(self, "V_ds_")

        # Validate the input data
        X = validate_data(
            self,
            X,
            ensure_2d=True,
            reset=False,
            dtype=np.float64,
        )

        # Identity fallback: no source data was provided at fit time
        if not self.x_source_provided_:
            return X

        # Low-rank transform: equivalent to X @ T_* but O(n * p * r) instead
        # of O(n * p^2), where r = n_transfer_samples << p = n_features.
        return (X @ self.V_ds_) @ self.B_ds_
