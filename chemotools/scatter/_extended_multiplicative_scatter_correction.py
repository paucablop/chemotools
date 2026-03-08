"""
The :mod:`chemotools.scatter._extended_multiplicative_scatter_correction` module
implements an Extended Multiplicative Scatter Correction transformer.
"""

# Authors: Pau Cabaneros
# License: MIT

from numbers import Integral
from typing import Literal, Optional

import numpy as np
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin
from sklearn.utils import check_array, check_consistent_length
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.validation import check_is_fitted, validate_data


class ExtendedMultiplicativeScatterCorrection(
    OneToOneFeatureMixin, TransformerMixin, BaseEstimator
):
    """Extended Multiplicative Scatter Correction (EMSC).

    EMSC is a preprocessing technique used to remove non-linear scatter effects
    and baseline shifts from spectral data. It fits a model consisting of a
    polynomial baseline, a reference spectrum, and optional interference
    spectra to each sample.

    Parameters
    ----------
    method : {"mean", "median"}, default="mean"
        The statistic used to calculate the reference spectrum if `reference`
        is None.

    order : int, default=2
        The order of the polynomial baseline. 0 is a constant offset,
        1 is linear, 2 is quadratic, etc.

    reference : array-like of shape (n_features,), default=None
        A custom reference spectrum. If provided, `method` is ignored.

    interferences : array-like of shape (n_interferences, n_features), default=None
        Known spectra of chemical interferents (e.g., water, CO2) to be
        mathematically removed from the signal.

    weights : array-like of shape (n_features,), default=None
        Wavelength weights for Weighted EMSC. Useful for de-emphasizing
        noisy regions of the spectrum.

    Attributes
    ----------
    reference_ : ndarray of shape (n_features,)
        The reference spectrum used for the correction.

    weights_ : ndarray of shape (n_features,)
        The actual weights applied during fitting.

    A_ : ndarray of shape (n_features, n_components)
        The design matrix used for regression.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

    Notes
    -----
    The model for each spectrum $x$ is:

    .. math::
        x = \sum_{i=0}^{order} c_i \lambda^i
        + m \cdot x_{ref} + \sum g_j \cdot z_j + \epsilon


    The corrected spectrum is calculated by removing the polynomial baseline
    and the interferences, then normalizing by the scaling factor $m$:

    .. math::
        x_{corr} = \\frac{x - (\sum c_i \lambda^i + \sum g_j \cdot z_j)}{m}

    References
    ----------
    .. [1] Nils Kristian Afseth, Achim Kohler. "Extended multiplicative signal
       correction in vibrational spectroscopy, a tutorial,"
       Chemometrics and Intelligent Laboratory Systems, 2012.
    """

    _parameter_constraints: dict = {
        "method": [StrOptions({"mean", "median"})],
        "order": [Interval(Integral, 0, None, closed="left")],
        "reference": ["array-like", None],
        "interferences": ["array-like", None],
        "weights": ["array-like", None],
    }

    def __init__(
        self,
        method: Literal["mean", "median"] = "mean",
        order: int = 2,
        reference: Optional[np.ndarray] = None,
        interferences: Optional[np.ndarray] = None,
        weights: Optional[np.ndarray] = None,
    ):
        self.method = method
        self.order = order
        self.reference = reference
        self.interferences = interferences
        self.weights = weights

    def fit(self, X, y=None):
        """Fit the EMSC model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training data.
        y : None
            Ignored.

        Returns
        -------
        self : object
            Fitted transformer.
        """
        self._validate_params()
        X = validate_data(self, X, reset=True, dtype=np.float64)
        n_features = X.shape[1]

        # 1. Resolve Reference Spectrum
        if self.reference is not None:
            self.reference_ = check_array(self.reference, ensure_2d=False)
            check_consistent_length(self.reference_, X.T)
        elif self.method == "mean":
            self.reference_ = np.mean(X, axis=0)
        else:
            self.reference_ = np.median(X, axis=0)

        # 2. Resolve Weights
        self.weights_ = (
            check_array(self.weights, ensure_2d=False)
            if self.weights is not None
            else np.ones(n_features)
        )
        check_consistent_length(self.weights_, X.T)

        # 3. Build Design Matrix A
        # Polynomial part
        x_indices = np.linspace(-1, 1, n_features)  # Normalized indices for stability
        poly_terms = np.vander(x_indices, N=self.order + 1, increasing=True)

        # Signal part (Reference)
        self.A_ = np.column_stack([poly_terms, self.reference_])

        # Orthogonal Subspace (Interferences)
        if self.interferences is not None:
            interf = check_array(self.interferences, ensure_2d=True)
            if interf.shape[1] != n_features:
                raise ValueError("Interference spectra must match X feature count.")
            self.A_ = np.column_stack([self.A_, interf.T])

        return self

    def transform(self, X):
        """Apply EMSC correction to X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to transform.

        Returns
        -------
        X_corr : ndarray of shape (n_samples, n_features)
            Corrected spectra.
        """
        check_is_fitted(self)
        X = validate_data(self, X, reset=False, dtype=np.float64)

        # Apply weights to A and X for Weighted Least Squares
        W = self.weights_[:, np.newaxis]
        WA = self.A_ * W
        WX = (X * self.weights_).T

        # Solve regression: WA @ coeffs = WX
        # lstsq is more robust than inv() or pinv() for singular matrices
        coeffs, _, _, _ = np.linalg.lstsq(WA, WX, rcond=None)

        # Partition coefficients
        n_poly = self.order + 1
        m = coeffs[n_poly, :].reshape(-1, 1)  # Scaling factor for reference

        # Calculate the "Noise" (Polynomials + Interferences)
        # We zero out the 'm' coefficient to reconstruct only the noise
        noise_coeffs = coeffs.copy()
        noise_coeffs[n_poly, :] = 0
        noise_contribution = (self.A_ @ noise_coeffs).T

        # Final correction: (Original - Baseline - Interferences) / Scaling
        return (X - noise_contribution) / m
