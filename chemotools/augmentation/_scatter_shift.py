"""
The :mod:`chemotools.augmentation._scatter_shift` module implements the ScatterShift
transformer to augment spectral data with random multiplicative scatter and a
random polynomial baseline, following the forward Extended Multiplicative Scatter
Correction (EMSC) model.
"""

# Authors: Prabesh Joshi
# License: MIT

from numbers import Integral
from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin
from sklearn.utils import check_random_state
from sklearn.utils._param_validation import Interval, Real
from sklearn.utils.validation import check_is_fitted, validate_data

from chemotools._doc_mixin import DocLinkMixin


class ScatterShift(DocLinkMixin, TransformerMixin, OneToOneFeatureMixin, BaseEstimator):
    """Augment spectra with random multiplicative scatter and a polynomial baseline.

    This transformer applies the *forward* Extended Multiplicative Scatter
    Correction (EMSC) model to each spectrum, injecting the same scatter
    structure that :class:`~chemotools.scatter.ExtendedMultiplicativeScatterCorrection`
    is designed to remove. It is intended for data augmentation: generating
    synthetic spectra that mimic the multiplicative scaling and wavelength-
    dependent baseline drift introduced by physical light scatter (e.g.,
    differences in particle size, path length, or packing density).

    Unlike chaining :class:`~chemotools.augmentation.SpectrumScale` (a scalar
    multiplicative factor) with :class:`~chemotools.augmentation.BaselineShift`
    (a constant offset), this transformer can introduce a *wavelength-dependent*
    polynomial baseline, which is the characteristic signature of scatter that
    EMSC models.

    At default parameters (``multiplicative_scale=0.0`` and
    ``additive_scale=0.0``) the transformer is the identity.

    Parameters
    ----------
    order : int, default=2
        The order of the random polynomial baseline. 0 is a constant offset,
        1 is linear, 2 is quadratic, etc. Matches the ``order`` parameter of
        :class:`~chemotools.scatter.ExtendedMultiplicativeScatterCorrection`.

    multiplicative_scale : float, default=0.0
        Range of the uniform distribution used to draw the multiplicative
        scatter factor ``m``, sampled per spectrum from
        ``U(1 - multiplicative_scale, 1 + multiplicative_scale)``.

    additive_scale : float, default=0.0
        Range of the uniform distribution used to draw each polynomial baseline
        coefficient, sampled per spectrum from
        ``U(-additive_scale, additive_scale)``.

    random_state : int, default=None
        The random state to use for the random number generator.

    Attributes
    ----------
    polynomial_basis_ : ndarray of shape (n_features, order + 1)
        The polynomial design matrix evaluated on normalized wavelength indices,
        built identically to the EMSC baseline basis.

    n_features_in_ : int
        Number of features seen during :meth:`fit`.

    Notes
    -----
    The forward model applied to each spectrum :math:`x` is:

    .. math::
        x_{aug} = m \\cdot x + \\sum_{i=0}^{order} c_i \\lambda^i

    where :math:`m` is the multiplicative scatter factor, :math:`c_i` are the
    random polynomial baseline coefficients, and :math:`\\lambda` are the
    wavelength indices normalized to the interval :math:`[-1, 1]`. This is the
    inverse of the EMSC correction model.

    References
    ----------
    .. [1] Nils Kristian Afseth, Achim Kohler. "Extended multiplicative signal
       correction in vibrational spectroscopy, a tutorial,"
       Chemometrics and Intelligent Laboratory Systems, 2012.

    Examples
    --------
    >>> from chemotools.augmentation import ScatterShift
    >>> from chemotools.datasets import load_fermentation_train
    >>> # Load sample data
    >>> X, _ = load_fermentation_train()
    >>> # Instantiate the transformer
    >>> transformer = ScatterShift(
    ...     order=2, multiplicative_scale=0.05, additive_scale=0.01, random_state=42
    ... )
    >>> transformer.fit(X)
    ScatterShift()
    >>> # Generate scatter-augmented data
    >>> X_augmented = transformer.transform(X)
    """

    _parameter_constraints: dict = {
        "order": [Interval(Integral, 0, None, closed="left")],
        "multiplicative_scale": [Interval(Real, 0, None, closed="both")],
        "additive_scale": [Interval(Real, 0, None, closed="both")],
        "random_state": [None, int, np.random.RandomState],
    }

    def __init__(
        self,
        order: int = 2,
        multiplicative_scale: float = 0.0,
        additive_scale: float = 0.0,
        random_state: Optional[int] = None,
    ):
        self.order = order
        self.multiplicative_scale = multiplicative_scale
        self.additive_scale = additive_scale
        self.random_state = random_state

    def fit(self, X: np.ndarray, y=None) -> "ScatterShift":
        """
        Fit the transformer to the input data.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            The input data to fit the transformer to.

        y : None
            Ignored.

        Returns
        -------
        self : ScatterShift
            The fitted transformer.
        """
        # Validate the input parameters
        self._validate_params()

        # Check that X is a 2D array and has only finite values
        X = validate_data(
            self, X, y="no_validation", ensure_2d=True, reset=True, dtype=np.float64
        )

        # Build the polynomial baseline basis on normalized wavelength indices.
        # This mirrors the EMSC design matrix so the augmentation is the forward
        # counterpart of ExtendedMultiplicativeScatterCorrection.
        x_indices = np.linspace(-1, 1, self.n_features_in_)
        self.polynomial_basis_ = np.vander(x_indices, N=self.order + 1, increasing=True)

        # Instantiate the random number generator
        self._rng = check_random_state(self.random_state)

        return self

    def transform(self, X: np.ndarray, y=None) -> np.ndarray:
        """
        Transform the input data by applying random multiplicative scatter and a
        random polynomial baseline.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            The input data to transform.

        y : None
            Ignored.

        Returns
        -------
        X_transformed : np.ndarray of shape (n_samples, n_features)
            The transformed data.
        """
        # Check that the estimator is fitted
        check_is_fitted(self, "n_features_in_")

        # Check that X is a 2D array and has only finite values
        X_ = validate_data(
            self,
            X,
            y="no_validation",
            ensure_2d=True,
            copy=True,
            reset=False,
            dtype=np.float64,
        )

        n_samples = X_.shape[0]

        # Draw one multiplicative factor per spectrum: m ~ U(1 - s, 1 + s)
        multiplicative_factors = self._rng.uniform(
            low=1 - self.multiplicative_scale,
            high=1 + self.multiplicative_scale,
            size=(n_samples, 1),
        )

        # Draw polynomial baseline coefficients per spectrum: c_i ~ U(-s, s)
        baseline_coefficients = self._rng.uniform(
            low=-self.additive_scale,
            high=self.additive_scale,
            size=(n_samples, self.order + 1),
        )

        # Forward EMSC model: x_aug = m * x + polynomial baseline
        baselines = baseline_coefficients @ self.polynomial_basis_.T
        return multiplicative_factors * X_ + baselines
