from typing import Literal, Optional

import numpy as np
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin
from sklearn.utils import check_array, check_consistent_length
from sklearn.utils._param_validation import StrOptions
from sklearn.utils.validation import check_is_fitted, validate_data

from chemotools._doc_mixin import DocLinkMixin


class MultiplicativeScatterCorrection(
    DocLinkMixin, OneToOneFeatureMixin, TransformerMixin, BaseEstimator
):
    """Multiplicative Scatter Correction (MSC).

    MSC is a transformation method used to compensate for additive and/or
    multiplicative scatter effects in spectral data (like NIR). It linearizes
    each spectrum against a reference spectrum (usually the mean or median)
    using Ordinary Least Squares (OLS) or Weighted Least Squares (WLS).

    Read more in the :ref:`User Guide <msc>`.

    Parameters
    ----------
    method : {"mean", "median"}, default="mean"
        The statistic used to calculate the reference spectrum if `reference`
        is None.
        - "mean": Use the average spectrum of the training set.
        - "median": Use the median spectrum of the training set.

    reference : array-like of shape (n_features,), default=None
        A custom reference spectrum to use for the correction. If provided,
        `method` is ignored.

    weights : array-like of shape (n_features,), default=None
        Weighting vector applied during the linear regression for each spectrum.
        Useful for de-emphasizing noisy wavelengths.

    Attributes
    ----------
    reference_ : ndarray of shape (n_features,)
        The reference spectrum used for the correction, either passed via
        `reference` or calculated during :meth:`fit`.

    weights_ : ndarray of shape (n_features,)
        The weights used in the correction. Defaults to a vector of ones.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

    pinv_A_ : ndarray of shape (2, n_features)
        The precomputed weighted pseudo-inverse of the design matrix used
        to solve for $m$ (slope) and $c$ (intercept) efficiently.

    Notes
    -----
    The correction follows the linear model:

    .. math::
        x_{raw} = m \cdot x_{ref} + c + e

    where $x_{raw}$ is the observed spectrum, $x_{ref}$ is the reference
    spectrum, $m$ is the multiplicative scaling, and $c$ is the additive
    offset. The corrected spectrum is calculated as:

    .. math::
        x_{corr} = \\frac{x_{raw} - c}{m}

    References
    ----------
    .. [1] Åsmund Rinnan, Frans van den Berg, Søren Balling Engelsen,
       "Review of the most common pre-processing techniques for near-infrared
       spectra," TrAC Trends in Analytical Chemistry 28 (10) 1201-1222 (2009).

    Examples
    --------
    >>> import numpy as np
    >>> from chemotools.scatter import MultiplicativeScatterCorrection
    >>> X = np.random.rand(10, 100)
    >>> msc = MultiplicativeScatterCorrection(method='mean')
    >>> msc.fit(X)
    MultiplicativeScatterCorrection()
    >>> X_corr = msc.transform(X)
    """

    # Defining constraints properly fixes the check_estimator issues
    _parameter_constraints: dict = {
        "method": [StrOptions({"mean", "median"})],
        "reference": ["array-like", None],
        "weights": ["array-like", None],
    }

    def __init__(
        self,
        method: Literal["mean", "median"] = "mean",
        reference: Optional[np.ndarray] = None,
        weights: Optional[np.ndarray] = None,
    ):
        self.method = method
        self.reference = reference
        self.weights = weights

    def fit(self, X, y=None):
        # 1. Validate parameters via the built-in sklearn machinery
        self._validate_params()

        # 2. Validate input data
        X = validate_data(self, X, reset=True, dtype=np.float64)

        # 3. Determine the reference spectrum
        if self.reference is not None:
            self.reference_ = check_array(
                self.reference, ensure_2d=False, dtype=np.float64
            )
            check_consistent_length(self.reference_, X.T)
        elif self.method == "mean":
            self.reference_ = np.mean(X, axis=0)
        else:  # median
            self.reference_ = np.median(X, axis=0)

        # 4. Handle weights
        if self.weights is not None:
            self.weights_ = check_array(self.weights, ensure_2d=False, dtype=np.float64)
            check_consistent_length(self.weights_, X.T)
        else:
            self.weights_ = np.ones_like(self.reference_)

        # Pre-calculate the design matrix A and the
        # (A^T A)^-1 A^T part for the pseudoinverse
        # This makes transform() much faster.
        # We apply weights to the design matrix here.
        self.A_ = np.vstack([self.reference_, np.ones_like(self.reference_)]).T
        W = np.diag(self.weights_)
        # Precompute the hat matrix for WLS: (A^T W A)^-1 A^T W
        WA = W @ self.A_
        self.pinv_A_ = np.linalg.inv(WA.T @ WA) @ WA.T

        return self

    def transform(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, reset=False, dtype=np.float64)

        # Vectorized MSC: Solve (m, c) for all rows at once
        # coefficients shape will be (2, n_samples)
        # We multiply by weighted X: W @ X.T
        WX = (X * self.weights_).T
        coeffs = self.pinv_A_ @ WX

        m = coeffs[0, :].reshape(-1, 1)  # slope
        c = coeffs[1, :].reshape(-1, 1)  # intercept

        # Correct the spectra: (X - intercept) / slope
        return (X - c) / m
