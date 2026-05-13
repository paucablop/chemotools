"""
The :mod: `chemotools.adaptation._piecewise_direct_standardization`
module implements the Piecewise Direct Standardization (PDS) transformer
"""

# Author: Ruggero Guerrini
# Licence: MIT

import warnings
from numbers import Integral

import numpy as np
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.utils._param_validation import Interval
from sklearn.utils.validation import check_is_fitted, validate_data


class PiecewiseDirectStandardization(
    OneToOneFeatureMixin, TransformerMixin, BaseEstimator
):
    """
    Piecewise Direct Standardization (PDS) is a transformer used for domain adaptation
    (calibration) applications. The transformer uses least squares to find a linear map
    from the target instrument space to the source instrument space, following the
    implementation by [1] and [2].

    Parameters
    ----------
    window_length : int
        Half-width (w) of the local spectral window used in PDS

    n_components : int
        Number of components to keep for PLS model

    scale : bool, default = True
        Whether to scale X and Y in the PLS model

    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during fit (set automatically by sklearn).

    x_mean_ : np.ndarray of shape (n_features, 2 * window_length + 1) or None
        Mean of the local X window for each feature. None if fitted with X_source=None
        (identity transformation).

    coef_ : np.ndarray of shape (n_features, 2 * window_length + 1) or None
        Regression coefficients for each local PLS model. None if fitted with
        X_source=None (identity transformation).

    intercept_ : np.ndarray of shape (n_features,) or None
        Intercept term for each local PLS model. None if fitted with X_source=None
        (identity transformation).

    x_source_provided_ : bool
        Boolean flag indicating if X_source was provided during fitting.


    Raises
    ------
    ValueError
        If X and X_source do not have the same shape.

    See Also
    --------
    DirectStandardization : Global linear transformation without local windows.

    Reference
    ---------
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
    }

    # Fitted attributes (set during fit, typed for type checkers)
    n_features_in_: int
    x_mean_: np.ndarray | None
    coef_: np.ndarray | None
    intercept_: np.ndarray | None
    x_source_provided_: bool

    def __init__(
        self,
        window_length: int = 25,
        n_components: int = 2,
        scale: bool = True,
    ):
        self.window_length = window_length
        self.n_components = n_components
        self.scale = scale

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

        # Check that X is a 2D array and has only finite values
        X = validate_data(self, X, ensure_2d=True, reset=True, dtype=np.float64)

        # Validate n_components
        if self.n_components > X.shape[0]:
            raise ValueError(
                f"n_components={self.n_components} must be <= n_samples={X.shape[0]}"
            )

        # If X_source is None, default to identity transformation
        if X_source is None:
            warnings.warn(
                "X_source is None, the transformer will act as an identity "
                "transformation."
            )
            self.x_source_provided_ = False
            self.x_mean_ = None
            self.coef_ = None
            self.intercept_ = None
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

        self.x_source_provided_ = True
        p = X.shape[1]

        max_win = 2 * self.window_length + 1
        self.x_mean_ = np.zeros((p, max_win), dtype=np.float64)
        self.coef_ = np.zeros((p, max_win), dtype=np.float64)
        self.intercept_ = np.empty(p, dtype=np.float64)

        for i in range(p):
            l_lim = max(0, i - self.window_length)
            r_lim = min(p, i + self.window_length + 1)
            win_size = r_lim - l_lim

            model = PLSRegression(
                n_components=self.n_components,
                scale=self.scale,
            ).fit(X[:, l_lim:r_lim], X_source[:, i])

            self.x_mean_[i, :win_size] = X[:, l_lim:r_lim].mean(axis=0)
            self.coef_[i, :win_size] = model.coef_.ravel()
            self.intercept_[i] = model.intercept_[0]
        return self

    def transform(self, X) -> np.ndarray:
        """
        Use the trained model to transform the source data

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
        check_is_fitted(self)

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

        # Type assertions for type checker - these are guaranteed non-None when
        # x_source_provided_ is True
        assert self.x_mean_ is not None
        assert self.coef_ is not None
        assert self.intercept_ is not None

        X_transformed = np.zeros(X.shape)
        for i in range(self.n_features_in_):
            l_lim = max(0, i - self.window_length)
            r_lim = min(self.n_features_in_, i + self.window_length + 1)
            win_size = r_lim - l_lim

            X_win = X[:, l_lim:r_lim] - self.x_mean_[i, :win_size]
            X_transformed[:, i] = X_win @ self.coef_[i, :win_size] + self.intercept_[i]
        return X_transformed
