"""
The :mod:`chemotools.adaptation._piecewise_direct_standardization`
module implements the Piecewise Direct Standardization (PDS) transformer
"""

# Author: Ruggero Guerrini & Pau Cabaneros
# Licence: MIT

import warnings
from numbers import Integral

import numpy as np
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

    scale : bool, default=True
        Whether to scale X and Y in the PLS model

    storage : str {"dense", "band"}, default="dense"
        Storage format for the regression coefficients.
        - ``"dense"`` stores the full ``(n_features, n_features)`` matrix.
          Fastest when ``n_features`` is small enough that the matrix fits in CPU
          cache (roughly ``n_features`` ≤ 1 500 at float64).
        - ``"band"`` stores coefficients in a compact
          ``(n_features, 2 * window_length + 1)`` array and exploits the band
          structure in :meth:`transform` via a strided sliding-window view and
          ``einsum``.  Fastest when ``n_features`` is large (roughly
          ``n_features`` > 1 500), and uses ~2–5 % of the memory of ``"dense"``.

    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during fit (set automatically by sklearn).

    T_ : np.ndarray of shape (n_features, n_features), or None.
        Banded transformation matrix.  Dense ndarray when ``storage="dense"``.
        ``None`` when ``storage="band"`` or when fitted without ``X_source``.

    coef_band_ : np.ndarray of shape (n_features, 2 * window_length + 1), or None.
        Packed band of per-feature PLS coefficients.  Set only when
        ``storage="band"``; ``None`` otherwise.

    interior_start_ : int
        Index of the first feature whose local window is fully interior
        (``window_length``).  Set only when ``storage="band"``.

    interior_end_ : int
        One past the last interior feature (``n_features - window_length``).
        Set only when ``storage="band"``.

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
        "storage": [StrOptions({"dense", "band"})],
    }

    n_features_in_: int
    T_: np.ndarray | None
    coef_band_: np.ndarray | None
    interior_start_: int
    interior_end_: int
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
            self.coef_band_ = None
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

        full_win = 2 * self.window_length + 1

        # Pre-allocate storage.  For "band" we skip the (n×n) matrix entirely.
        _coef_band: np.ndarray | None = None
        _T: np.ndarray | None = None
        if self.storage == "band":
            _coef_band = np.zeros((n_features, full_win), dtype=np.float64)
        else:  # dense
            _T = np.zeros((n_features, n_features), dtype=np.float64)

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

            # Store coefficients in the appropriate structure
            if self.storage == "band":
                assert _coef_band is not None
                _coef_band[i, : r_lim - l_lim] = coef
            else:  # dense
                assert _T is not None
                _T[l_lim:r_lim, i] = coef

            # Precalculate the shifted bias
            self.bias_[i] = intercept - np.dot(mean, coef)

        # Finalise fitted attributes
        if self.storage == "band":
            self.coef_band_ = _coef_band
            # Clamp so that interior_start_ <= interior_end_ and both are in
            # [0, n_features].  When window_length >= n_features there are no
            # interior features and every feature is handled by the boundary
            # loop in transform().
            self.interior_start_ = min(self.window_length, n_features)
            self.interior_end_ = max(
                n_features - self.window_length, self.interior_start_
            )
            self.T_ = None
        else:  # dense
            self.T_ = _T
            self.coef_band_ = None

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

        assert self.bias_ is not None

        if self.storage == "band":
            assert self.coef_band_ is not None
            full_win = 2 * self.window_length + 1
            is_ = self.interior_start_
            ie_ = self.interior_end_
            p = X.shape[1]
            out = np.empty_like(X)
            if is_ < ie_:
                # Zero-copy strided view: shape (n_samples, n_interior, full_win)
                X_wins = np.lib.stride_tricks.sliding_window_view(X, full_win, axis=1)
                # Interior features all share the same window width — one einsum call
                out[:, is_:ie_] = np.einsum(
                    "nfw,fw->nf",
                    X_wins[:, : ie_ - is_],
                    self.coef_band_[is_:ie_],
                    optimize=True,
                )
            # Boundary features (2 * window_length of them total) — small loop.
            # When window_length >= n_features every feature is a boundary feature.
            for i in list(range(is_)) + list(range(ie_, p)):
                lo = max(0, i - self.window_length)
                r = min(p, i + self.window_length + 1)
                out[:, i] = X[:, lo:r] @ self.coef_band_[i, : r - lo]
            return np.asarray(out + self.bias_)

        assert self.T_ is not None
        return np.asarray(X @ self.T_ + self.bias_)
