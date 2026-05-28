"""
The :mod:`chemotools.adaptation._coral`
module implements the CORrelation ALignment (CORAL) transformer
"""

# Authors: Ruggero Guerrini
# License: MIT

import warnings
from numbers import Real

import numpy as np
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin
from sklearn.utils._param_validation import Interval
from sklearn.utils.validation import check_is_fitted, validate_data

from chemotools._doc_mixin import DocLinkMixin


class CORAL(DocLinkMixin, OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    """
    CORrelation ALignment (CORAL) is an unsupervised domain adaptation transformer
    that aligns the second-order statistics (covariance) of source and target feature
    distributions, without requiring any target labels.

    CORAL is a linear transformation method used to transfer spectral data from
    a target domain to a source (reference) domain, allowing calibration
    models to remain valid across different instruments or acquisition
    conditions, following the implementation by [1]_.

    Parameters
    ----------
    reg : float, default=1.0
        Regularisation parameter added to the diagonal of both covariance matrices
        before computing their matrix square roots.  A positive value guarantees
        that the matrices are invertible and improves numerical stability.
        Setting ``reg=0`` uses the analytical (SVD-based) solution derived in [1]_,
        but may be numerically unstable when the covariance matrices are singular
        or ill-conditioned.

    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during :meth:`fit`.

    A_ : np.ndarray of shape (n_features, n_features)
        The full CORAL linear transformation matrix.

    C_X_ : np.ndarray of shape (n_features, n_features)
        Regularised covariance matrix of the target domain ``X``.

    C_X_source_ : np.ndarray of shape (n_features, n_features)
        Regularised covariance matrix of the source domain ``X_source``.

    C_X_inv_sqrt_ : np.ndarray of shape (n_features, n_features)
        Inverse square root of ``C_X_``.

    C_X_source_sqrt_ : np.ndarray of shape (n_features, n_features)
        Square root of ``C_X_source_``.

    X_mean_ : np.ndarray of shape (n_features,)
        Mean of the target domain data ``X``.

    X_centered_ : np.ndarray of shape (n_samples, n_features)
        Target domain data ``X`` centred by subtracting ``X_mean_``.

    X_source_mean_ : np.ndarray of shape (n_features,)
        Mean of the source domain data ``X_source``.

    X_source_centered_ : np.ndarray of shape (n_samples_source, n_features)
        Source domain data ``X_source`` centred by subtracting
        ``X_source_mean_``.

    x_source_provided_ : bool
        Boolean flag indicating if X_source was provided during fitting.

    Raises
    ------
    ValueError
        If ``X`` and ``X_source`` do not have the same number of features.

    See Also
    --------
    DirectStandardization : Supervised calibration transfer via least squares.
    SpectralSpaceTransform : Subspace-based calibration transfer via SVD.

    References
    ----------
    .. [1] Sun, B., Feng, J., & Saenko, K. (2016).
        Return of Frustratingly Easy Domain Adaptation,
        Proceedings of the AAAI Conference on Artificial Intelligence, 30(1).
        arXiv:1511.05547.

    Examples
    --------
    **Basic usage**

    >>> import numpy as np
    >>> from chemotools.adaptation import CORAL
    >>>
    >>> rng = np.random.default_rng(42)
    >>> X_source = rng.normal(size=(80, 50))
    >>> X_source = X_source @ np.diag(rng.uniform(0.5, 2.0, 50)) + rng.normal(
    ...     scale=0.1, size=(80, 50)
    ... )
    >>>
    >>> coral = CORAL(reg=1.0).fit(X_source, X_source=X_source)
    >>> X_adapted = coral.transform(X_source)

    **Inside a Pipeline**

    >>> from sklearn.pipeline import Pipeline
    >>> from sklearn.preprocessing import StandardScaler
    >>>
    >>> pipe = Pipeline([
    ...     ("scaler", StandardScaler()),
    ...     ("coral", CORAL(reg=1.0)),
    ... ])
    >>> pipe.fit(X_source, coral__X_source=X_source)
    Pipeline(steps=[('scaler', StandardScaler()), ('coral', CORAL())])
    >>> X_adapted = pipe.transform(X_source)
    """

    _parameter_constraints: dict = {
        "reg": [Interval(Real, 0, None, closed="left")],
    }
    # Fitted attributes
    n_features_in_: int
    X_centered_: np.ndarray | None
    X_source_mean_: np.ndarray | None
    X_source_centered_: np.ndarray | None
    C_X_: np.ndarray | None
    C_X_source_: np.ndarray | None
    C_X_inv_sqrt_: np.ndarray | None
    C_X_source_sqrt_: np.ndarray | None
    x_source_provided_: bool

    def __init__(self, reg: float = 1.0) -> None:
        self.reg = reg

    def fit(
        self,
        X: np.ndarray,
        y=None,
        *,
        X_source: np.ndarray | None = None,
    ) -> "CORAL":
        """
        Fit the CORAL model.

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
        self : CORAL
            Fitted estimator.
        """
        self._validate_params()

        X = validate_data(self, X, ensure_2d=True, reset=True, dtype=np.float64)

        if X_source is None:
            warnings.warn(
                "X_source is None, the transformer will act as an identity "
                "transformation."
            )
            self.X_mean_ = None
            self.X_centered_ = None
            self.X_source_mean_ = None
            self.X_source_centered_ = None
            self.C_X_ = None
            self.C_X_source_ = None
            self.C_X_inv_sqrt_ = None
            self.C_X_source_sqrt_ = None
            self.x_source_provided_ = False
            return self

        # Validate X_source as a plain array (do not update n_features_in_)
        X_source = np.asarray(X_source, dtype=np.float64)
        if X_source.ndim != 2:
            raise ValueError(
                f"X_source must be a 2D array, got {X_source.ndim}D array."
            )
        if X_source.shape[1] != X.shape[1]:
            raise ValueError(
                f"X and X_source must have the same number of features, "
                f"got X={X.shape[1]} and X_source={X_source.shape[1]}."
            )

        # Center the data
        self.X_mean_ = X.mean(axis=0)
        self.X_centered_ = X - self.X_mean_
        self.X_source_mean_ = X_source.mean(axis=0)
        self.X_source_centered_ = X_source - self.X_source_mean_

        # Covariance matrix
        self.C_X_ = np.cov(self.X_centered_, rowvar=False, ddof=1) + self.reg * np.eye(
            self.n_features_in_
        )
        self.C_X_source_ = np.cov(
            self.X_source_centered_, rowvar=False, ddof=1
        ) + self.reg * np.eye(self.n_features_in_)
        # Cs inverse square root
        eps = 1e-12
        eigenvalues_source, eigenvectors_source = np.linalg.eigh(self.C_X_)
        eigenvalues_source = np.clip(eigenvalues_source, eps, None)
        self.C_X_inv_sqrt_ = (
            eigenvectors_source
            @ np.diag(1.0 / np.sqrt(eigenvalues_source))
            @ eigenvectors_source.T
        )
        # Ct square root
        eigenvalues_target, eigenvectors_target = np.linalg.eigh(self.C_X_source_)
        eigenvalues_target = np.clip(eigenvalues_target, eps, None)
        self.C_X_source_sqrt_ = (
            eigenvectors_target
            @ np.diag(np.sqrt(eigenvalues_target))
            @ eigenvectors_target.T
        )

        # Compute the CORAL transformation:
        self.A_ = self.C_X_inv_sqrt_ @ self.C_X_source_sqrt_

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
        check_is_fitted(self)

        X = validate_data(self, X, ensure_2d=True, reset=False, dtype=np.float64)
        if not self.x_source_provided_:
            return X

        assert self.X_mean_ is not None
        assert self.X_centered_ is not None
        assert self.X_source_mean_ is not None
        assert self.X_source_centered_ is not None
        assert self.C_X_ is not None
        assert self.C_X_source_ is not None
        assert self.C_X_inv_sqrt_ is not None
        assert self.C_X_source_sqrt_ is not None
        assert self.x_source_provided_ is not False

        X_centered = X - self.X_mean_
        X_adapted = X_centered @ self.A_
        return X_adapted + self.X_source_mean_
