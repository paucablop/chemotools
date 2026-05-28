"""
The :mod:`chemotools.adaptation._piecewise_direct_standardization`
module implements the Piecewise Direct Standardization (PDS) transformer
"""

# Author: Ruggero Guerrini & Pau Cabaneros
# Licence: MIT

import warnings
from numbers import Integral

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.validation import check_is_fitted, validate_data

from chemotools._doc_mixin import DocLinkMixin


class PiecewiseDirectStandardization(
    DocLinkMixin, OneToOneFeatureMixin, TransformerMixin, BaseEstimator
):
    """
    Piecewise Direct Standardization (PDS) is a transformer used for domain adaptation
    (calibration) applications. The transformer uses least squares to find a linear map
    from the target instrument space to the source instrument space, following the
    implementation by [1]_ and [2]_.

    Parameters
    ----------
    window_length : int, default=25
        Half-width (w) of the local spectral window used in PDS

    n_components : int, default=2
        Number of components to keep for PLS model

    scale : bool, default = True
        Whether to scale X and Y in the PLS model

    storage : str {"dense", "sparse"}, default="dense"
        Storage format for the regression coefficients.
        - "dense" stores the full matrix with zeros outside the local windows, while
        - "sparse" stores only the non-zero coefficients for memory efficiency.

        "sparse" is recommended for large feature sets and small window_length, while
        "dense" may be faster for small feature sets or large window_length.

    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during fit (set automatically by sklearn).

    T_ : np.ndarray or scipy.sparse.csr_matrix of shape (n_features, n_features), or
        None.
        Banded transformation matrix mapping target instrument space to source
        instrument space. Dense ndarray when ``storage="dense"``, CSR sparse
        matrix when ``storage="sparse"``. None if fitted with X_source=None.

    bias_ : np.ndarray of shape (n_features,), or None
        Precomputed per-feature bias that absorbs local PLS centering, allowing
        :meth:`transform` to avoid per-sample intermediate allocations. None if
        fitted with X_source=None.

    x_source_provided_ : bool
        Boolean flag indicating if X_source was provided during fitting.


    Raises
    ------
    ValueError
        If X and X_source do not have the same shape.
    ValueError
        If ``n_components`` exceeds ``n_samples``.
    ValueError
        If ``n_components`` exceeds the minimum window size at the boundaries
        (``window_length + 1``).

    See Also
    --------
    DirectStandardization : Global linear transformation without local windows.
    SpectralSpaceTransform : Linear alignment via SVD of the source-target matrix.
    CORrelationALignment : Unsupervised alignment via covariance matrix whitening.
    SubspaceAlignment : Unsupervised alignment via PCA subspace projection.

    References
    ----------
    .. [1] Wang, Yongdong., Veltkamp, D. J., & Kowalski, B. R. (1991),
        Multivariate instrument standardization,
        Analytical Chemistry, 63(23), Pages 2750–2756,
        https://doi.org/10.1021/ac00023a016.

    .. [2] Bouveresse, E.; Massart, D. L. (1996),
        Improvement of the piecewise direct standardisation procedure for the transfer
        of NIR spectra for multivariate calibration,
        Chemometrics and Intelligent Laboratory Systems, 32(2), Pages 201–213,
        https://doi.org/10.1016/0169-7439(95)00074-7.

    Examples
    --------
    >>> import numpy as np
    >>> from chemotools.adaptation import PiecewiseDirectStandardization
    >>> rng = np.random.default_rng(42)
    >>> X = rng.normal(size=(50, 100))
    >>> X_source = X * 1.2 + rng.normal(0, 0.1, size=(50, 100))
    >>> pds = PiecewiseDirectStandardization(window_length=5, n_components=2)
    >>> pds.fit(X, X_source=X_source)
    PiecewiseDirectStandardization(n_components=2, window_length=5)
    >>> X_transformed = pds.transform(X)
    >>> X_transformed.shape
    (50, 100)

    """

    _parameter_constraints: dict = {
        "window_length": [Interval(Integral, 1, None, closed="left")],
        "n_components": [Interval(Integral, 1, None, closed="left")],
        "scale": ["boolean"],
        "storage": [StrOptions({"dense", "sparse"})],
    }

    n_features_in_: int
    T_: np.ndarray | csr_matrix | None
    bias_: np.ndarray | None
    x_source_provided_: bool

    def __init__(
        self,
        window_length: int = 25,
        n_components: int = 2,
        scale: bool = True,
        storage: str = "dense",
    ):
        self.window_length = window_length
        self.n_components = n_components
        self.scale = scale
        self.storage = storage

    def fit(
        self, X: np.ndarray, y=None, *, X_source: np.ndarray | None = None
    ) -> "PiecewiseDirectStandardization":
        """
        Fit the PiecewiseDirectStandardization to the input data.

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
        self : PiecewiseDirectStandardization
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
            self.T_ = None
            self.bias_ = None
            return self

        # Validate n_components against n_samples
        if self.n_components > X.shape[0]:
            raise ValueError(
                f"n_components={self.n_components} must be <= n_samples={X.shape[0]}"
            )

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

        n_features = X.shape[1]

        # Validate n_components against the minimum window size at the boundaries
        min_win_size = self.window_length + 1
        if self.n_components > min_win_size:
            raise ValueError(
                f"n_components={self.n_components} cannot be strictly greater than "
                f"the minimum window size at the boundaries ({min_win_size}). "
                f"Please decrease n_components or increase window_length."
            )

        # Pre-allocate a local build matrix (lil_matrix for sparse, ndarray for dense)
        # and only assign to self.T_ after the final conversion, so the annotated
        # type (ndarray | csr_matrix | None) is never violated mid-construction.
        _T: np.ndarray | lil_matrix = (
            lil_matrix((n_features, n_features), dtype=np.float64)
            if self.storage == "sparse"
            else np.zeros((n_features, n_features), dtype=np.float64)
        )

        self.bias_ = np.zeros(n_features, dtype=np.float64)

        for i in range(n_features):
            l_lim = max(0, i - self.window_length)
            r_lim = min(n_features, i + self.window_length + 1)

            # Fit local PLS model
            model = PLSRegression(
                n_components=self.n_components,
                scale=self.scale,
            ).fit(X[:, l_lim:r_lim], X_source[:, i])

            coef = model.coef_.ravel()
            mean = X[:, l_lim:r_lim].mean(axis=0)
            intercept = model.intercept_[0]

            # Populate the diagonal band in the build matrix
            _T[l_lim:r_lim, i] = coef

            # Precalculate the shifted bias
            self.bias_[i] = intercept - np.dot(mean, coef)

        # Convert to the appropriate format for efficient arithmetic during transform
        if isinstance(_T, lil_matrix):
            self.T_ = _T.tocsr()
        else:
            self.T_ = _T

        self.x_source_provided_ = True

        return self

    def transform(self, X) -> np.ndarray:
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
        check_is_fitted(self, ["x_source_provided_"])
        X = validate_data(self, X, ensure_2d=True, reset=False, dtype=np.float64)

        if not self.x_source_provided_:
            return X

        assert self.T_ is not None
        assert self.bias_ is not None

        return np.asarray(X @ self.T_ + self.bias_)
