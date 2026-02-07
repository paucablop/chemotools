import numpy as np
from typing import Literal, Optional
from numbers import Integral

from sklearn.base import BaseEstimator, TransformerMixin, OneToOneFeatureMixin
from sklearn.utils import check_array, check_consistent_length
from sklearn.utils.validation import check_is_fitted, validate_data
from sklearn.utils._param_validation import Interval, StrOptions


class ExtendedMultiplicativeScatterCorrection(
    OneToOneFeatureMixin, TransformerMixin, BaseEstimator
):
    """Extended Multiplicative Scatter Correction (EMSC).

    EMSC extends MSC by adding a polynomial baseline to the regression model.
    This accounts for non-linear scatter effects and baseline shifts.

    Parameters
    ----------
    method : {"mean", "median"}, default="mean"
        The statistic used to calculate the reference spectrum if `reference` is None.

    order : int, default=2
        The order of the polynomial baseline. 0 is a constant offset,
        1 is linear, 2 is quadratic, etc.

    reference : array-like of shape (n_features,), default=None
        A custom reference spectrum. If provided, `method` is ignored.

    weights : array-like of shape (n_features,), default=None
        Wavelength weights for Weighted EMSC (WEMSC).

    Attributes
    ----------
    reference_ : ndarray of shape (n_features,)
        The reference spectrum used.

    weights_ : ndarray of shape (n_features,)
        The weights vector used (defaults to ones).

    A_ : ndarray of shape (n_features, order + 2)
        The design matrix containing polynomial terms and the reference spectrum.

    pinv_A_ : ndarray of shape (order + 2, n_features)
        The precomputed weighted pseudo-inverse of the design matrix.
    """

    _parameter_constraints: dict = {
        "method": [StrOptions({"mean", "median"})],
        "order": [Interval(Integral, 0, None, closed="left")],
        "reference": ["array-like", None],
        "weights": ["array-like", None],
    }

    def __init__(
        self,
        method: Literal["mean", "median"] = "mean",
        order: int = 2,
        reference: Optional[np.ndarray] = None,
        weights: Optional[np.ndarray] = None,
    ):
        self.method = method
        self.order = order
        self.reference = reference
        self.weights = weights

    def fit(self, X, y=None):
        self._validate_params()
        X = validate_data(self, X, reset=True, dtype=np.float64)
        n_features = X.shape[1]

        # 1. Resolve Reference
        if self.reference is not None:
            self.reference_ = check_array(self.reference, ensure_2d=False)
            check_consistent_length(self.reference_, X.T)
        elif self.method == "mean":
            self.reference_ = np.mean(X, axis=0)
        else:
            self.reference_ = np.median(X, axis=0)

        # 2. Resolve Weights
        if self.weights is not None:
            self.weights_ = check_array(self.weights, ensure_2d=False)
            check_consistent_length(self.weights_, X.T)
        else:
            self.weights_ = np.ones(n_features)

        # 3. Build Design Matrix A: [1, x, x^2, ..., reference]
        # Using a vandermonde matrix for the polynomial terms
        x_indices = np.linspace(0, 1, n_features)
        poly_terms = np.vander(x_indices, N=self.order + 1, increasing=True)
        self.A_ = np.column_stack([poly_terms, self.reference_])

        # 4. Precompute Weighted Pseudo-inverse for WLS
        # (A.T @ W @ A)^-1 @ A.T @ W
        W = np.diag(self.weights_)
        WA = W @ self.A_
        self.pinv_A_ = np.linalg.pinv(WA.T @ WA) @ WA.T

        return self

    def transform(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, reset=False, dtype=np.float64)

        # Vectorized Solve: Regress all spectra at once
        # coeffs shape: (order + 2, n_samples)
        WX = (X * self.weights_).T
        coeffs = self.pinv_A_ @ WX

        # Extract parameters
        # m is the last coefficient (scaling for the reference)
        # poly_coeffs are everything before that
        m = coeffs[-1, :].reshape(-1, 1)
        poly_coeffs = coeffs[:-1, :]

        # Calculate the baseline: A_poly @ poly_coeffs
        # A_poly is A_ without the last (reference) column
        baseline = (self.A_[:, :-1] @ poly_coeffs).T

        # Corrected spectrum: (Original - Baseline) / Scaling
        return (X - baseline) / m
