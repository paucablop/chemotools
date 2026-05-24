"""
The :mod: `chemotools.adaptation._spectral_space_transform`
module implements the Spectral Space Transform (SST) transformer
"""

# Authors: Ruggero Guerrini
# License: MIT

import warnings
from numbers import Integral

import numpy as np
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin
from sklearn.utils._param_validation import Interval
from sklearn.utils.validation import (
    check_is_fitted,
    validate_data,
)

from chemotools._doc_mixin import DocLinkMixin


class SpectralSpaceTransform(
    DocLinkMixin, OneToOneFeatureMixin, TransformerMixin, BaseEstimator
):
    """
    Spectral Space Transform (SST) is a transformer used for domain adaptation
    (calibration) applications.

    SST is a linear transformation method used to transfer spectral data from
    a target domain to a source (reference) domain, allowing calibration
    models to remain valid across different instruments or acquisition
    conditions, following the implementation by [1]_.

    The method constructs a shared latent space using singular value
    decomposition (SVD) of concatenated source and target data, and derives
    projection matrices to align the target data to the source space.

    Parameters
    ----------
    n_components : int, default=2
        Number of latent components retained from the singular value
        decomposition (SVD). Controls the dimensionality of the shared subspace.

    with_mean : bool, default=True
        If True, center each domain's data by subtracting its mean before
        computing the SVD. The source mean is added back after transforming.

    with_std : bool, default=False
        If True, scale each domain's data by its standard deviation after
        centering. Features with zero variance are left unchanged (std set to 1).
        Following the same convention as ``StandardScaler``.


    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during fit.

    X_target_mean_ : ndarray of shape (n_features,)
        Per-feature mean of the target data. Zero vector when ``with_mean=False``.

    X_source_mean_ : ndarray of shape (n_features,)
        Per-feature mean of the source data. Zero vector when ``with_mean=False``.

    X_target_std_ : ndarray of shape (n_features,)
        Per-feature standard deviation of the target data.
        Ones vector when ``with_std=False``.

    X_source_std_ : ndarray of shape (n_features,)
        Per-feature standard deviation of the source data.
        Ones vector when ``with_std=False``.

    P1_ : ndarray of shape (n_components, n_features), or None
        Projection matrix associated with the source domain. It maps the
        shared latent space back to the source spectral space.

    P2_ : ndarray of shape (n_components, n_features), or None
        Projection matrix associated with the target domain. It maps target
        data into the shared latent space.

    A_ : ndarray of shape (n_features, n_components), or None
        Pseudoinverse of ``P2_``. Left factor of the low-rank correction;
        stored to avoid recomputing the pseudoinverse on every transform call.

    A_eff_ : ndarray of shape (n_features, n_components), or None
        Effective left factor ``A_ / X_target_std_[:, None]`` with target
        scaling absorbed. Equals ``A_`` when ``with_std=False``.

    dP_eff_ : ndarray of shape (n_components, n_features), or None
        Effective right factor ``(P1_ - P2_) * X_source_std_[None, :]`` with
        source scaling absorbed. Equals ``P1_ - P2_`` when ``with_std=False``.

    scale_ : ndarray of shape (n_features,), or None
        Per-feature combined scale ``X_source_std_ / X_target_std_``.
        Ones vector when ``with_std=False``.

    bias_ : ndarray of shape (n_features,), or None
        Precomputed per-feature bias that absorbs all mean and std shifts,
        allowing :meth:`transform` to avoid intermediate array allocations.

    T_ : ndarray of shape (n_features, n_features), or None
        Full transformation matrix ``I + A_ @ (P1_ - P2_)`` (property,
        computed on demand). Equivalent to ``X_scaled @ T_`` in the SST
        formula but never materialised during :meth:`transform`.

    x_source_provided_ : bool
        Boolean flag indicating if X_source was provided during fitting.

    Raises
    ------
    ValueError
        If ``X`` and ``X_source`` do not have the same shape.
    ValueError
        If ``n_components`` exceeds ``min(n_samples, 2 * n_features)`` of the
        concatenated matrix.

    See Also
    --------
    PiecewiseDirectStandardization : Local standardization using moving windows.
    DirectStandardization : Global linear transformation without local windows.

    References
    ----------
    .. [1] Du, W., Chen, Z.-P., Zhong, L.-J.,Wang, S.-X., Yu, R.-Q., Nordon, A.,
        Littlejohn, D., & Holden, M. (2011).
        Maintaining the predictive abilities of multivariate calibration models by
        spectral space transformation,
        Analytica Chimica Acta, 690(1), Pages 64–70,
        https://doi.org/10.1016/j.aca.2011.02.014.

    Examples
    --------
    **Basic usage**
    >>> import numpy as np
    >>> from chemotools.adaptation import SpectralSpaceTransform
    >>>
    >>> rng = np.random.default_rng(17)
    >>> X_source = rng.normal(size=(100, 20))
    >>> X_target = X_source * 2 - rng.normal(size=(100, 20)) * 0.02
    >>>
    >>> sst = SpectralSpaceTransform(n_components=2).fit(X_target, X_source=X_source)
    >>> X_transf = sst.transform(X_target)

    """

    _parameter_constraints = {
        "n_components": [Interval(Integral, 1, None, closed="left")],
        "with_mean": ["boolean"],
        "with_std": ["boolean"],
    }

    n_features_in_: int
    P1_: np.ndarray | None
    P2_: np.ndarray | None
    A_: np.ndarray | None

    def __init__(
        self,
        n_components: int = 2,
        with_mean: bool = True,
        with_std: bool = False,
    ):
        self.n_components = n_components
        self.with_mean = with_mean
        self.with_std = with_std

    def fit(
        self, X: np.ndarray, y=None, *, X_source: np.ndarray | None = None
    ) -> "SpectralSpaceTransform":
        """
        Fit the SpectralSpaceTransform model.

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
        self : SpectralSpaceTransform
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
            self.x_source_provided_ = False
            self.X_target_mean_ = np.zeros(X.shape[1])
            self.X_source_mean_ = np.zeros(X.shape[1])
            self.X_target_std_ = np.ones(X.shape[1])
            self.X_source_std_ = np.ones(X.shape[1])
            self.P1_ = None
            self.P2_ = None
            self.A_ = None
            self.A_eff_ = None
            self.dP_eff_ = None
            self.scale_ = None
            self.bias_ = None
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

        # Compute centering/scaling statistics for source and target domains
        self.X_target_mean_ = (
            np.mean(X, axis=0) if self.with_mean else np.zeros(X.shape[1])
        )
        self.X_source_mean_ = (
            np.mean(X_source, axis=0) if self.with_mean else np.zeros(X_source.shape[1])
        )
        if self.with_std:
            self.X_target_std_ = np.std(X, axis=0, ddof=0)
            self.X_source_std_ = np.std(X_source, axis=0, ddof=0)
            # Handle zero-variance features to avoid division by zero
            target_zero = self.X_target_std_ == 0
            source_zero = self.X_source_std_ == 0
            if target_zero.any():
                warnings.warn(
                    f"X (target) has {target_zero.sum()} constant feature(s). "
                    "Their standard deviation is set to 1 to avoid division by zero."
                )
            if source_zero.any():
                warnings.warn(
                    f"X_source has {source_zero.sum()} constant feature(s). "
                    "Their standard deviation is set to 1 to avoid division by zero."
                )
            self.X_target_std_[target_zero] = 1.0
            self.X_source_std_[source_zero] = 1.0
        else:
            self.X_target_std_ = np.ones(X.shape[1])
            self.X_source_std_ = np.ones(X.shape[1])

        # Center and scale source and target data
        X_scaled = (X - self.X_target_mean_) / self.X_target_std_
        X_source_scaled = (X_source - self.X_source_mean_) / self.X_source_std_

        # Create X_comb from centered/scaled source and target data (Section 2 in [1])
        X_comb = np.hstack([X_source_scaled, X_scaled])

        # Validate that n_components does not exceed the rank of X_comb ([X_source, X]).
        # The maximum rank is at most min(n_samples, 2*n_features).

        max_components = min(X_comb.shape[0], X_comb.shape[1])
        if self.n_components > max_components:
            raise ValueError(
                f"n_components={self.n_components} is too large. "
                f"The rank of the concatenated matrix is at most "
                f"min(n_samples, 2*n_features) = {max_components}. "
                f"Set n_components to a value <= {max_components}."
            )

        # Compute the SVD of the joint matrix X_comb = [X_source | X]
        # Equation 1 in [1].
        _, _, Vt = np.linalg.svd(X_comb, full_matrices=False)

        # Transpose Vt to get V of shape (2*n_features, n_components)
        V = Vt.T

        # n_col_ref corresponds to the number of features (for both Source and Target)
        n_col_ref = X_source.shape[1]

        # Define projection matrices for the source and target domains.
        # Equation 1 in [1].
        self.P1_ = V[0:n_col_ref, 0 : self.n_components].T
        self.P2_ = V[n_col_ref:, 0 : self.n_components].T

        # Precompute A_ = pinv(P2_), the left factor of the low-rank correction.
        # Storing A_ avoids recomputing the pseudoinverse on every transform call.
        # T_ = I + A_ @ (P1_ - P2_) is available as a property (from Equation 6 in [1]).
        self.A_ = np.linalg.pinv(self.P2_)

        # Fold centering and scaling into effective factor matrices so that
        # transform() needs only two matmuls and one vector addition.
        dP = self.P1_ - self.P2_
        self.A_eff_ = self.A_ / self.X_target_std_[:, None]
        self.dP_eff_ = dP * self.X_source_std_[None, :]
        self.scale_ = self.X_source_std_ / self.X_target_std_
        self.bias_ = (
            self.X_source_mean_
            - self.X_target_mean_ * self.scale_
            - (self.X_target_mean_ @ self.A_eff_) @ self.dP_eff_
        )

        self.x_source_provided_ = True

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Use the trained model to transform the target data

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to transform

        Returns
        -------
        X_transformed : np.ndarray of shape (n_samples, n_features)
            Data transformed
        """
        # Verify that the model was trained
        check_is_fitted(self, ["x_source_provided_"])

        # Check the data
        X = validate_data(
            self,
            X,
            ensure_2d=True,
            reset=False,
            dtype=np.float64,
        )

        # If fitted as identity, return X unchanged
        if not self.x_source_provided_:
            return X

        # Apply the precomputed factored transform (Equation 6 in [1]):
        #   (X - mean_t)/std_t @ T_ * std_s + mean_s
        # bias_ absorbs all mean/std shifts; A_eff_ and dP_eff_ absorb the
        # per-feature scaling, so no intermediate arrays are needed.
        correction = (X @ self.A_eff_) @ self.dP_eff_ + self.bias_
        if self.with_std:
            return X * self.scale_ + correction
        return X + correction

    @property
    def T_(self) -> np.ndarray | None:
        if not self.x_source_provided_:
            return None
        assert self.A_ is not None
        assert self.P1_ is not None
        assert self.P2_ is not None
        n = self.n_features_in_
        return np.eye(n) + self.A_ @ (self.P1_ - self.P2_)
