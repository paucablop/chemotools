"""
The :mod:`chemotools.baseline._ar_pls` module implements the Asymmetrically Reweighted
Penalized Least Squares (ArPLS) baseline correction algorithm
"""

# Authors: Niklas Zell <nik.zoe@web.de>, Pau Cabaneros
# License: MIT

import logging

import numpy as np

from ._base import _BaseWhittaker


logger = logging.getLogger(__name__)


class ArPls(_BaseWhittaker):
    """
    Asymmetrically Reweighted Penalized Least Squares (ArPLS) baseline correction.

    This algorithm estimates and removes smooth baselines from spectroscopic data
    by iteratively reweighting residuals in a penalized least squares framework.
    A second-order difference operator is used as the penalty term, which promotes
    a smooth baseline estimate.

    The Whittaker smoothing step can be solved using either:
    - a **banded solver** (fast and memory-efficient, recommended for most spectra), or
    - a **sparse LU solver** (more stable for ill-conditioned problems).

    For efficiency, the algorithm supports warm-starting: when processing multiple
    spectra with similar baseline structure, weights from a previous fit can be
    reused, typically reducing the number of iterations needed.

    Parameters
    ----------
    lam : float, default=1e4
        Regularization parameter controlling smoothness of the baseline.
        Larger values yield smoother baselines.

    ratio : float, default=0.01
        Convergence threshold for weight updates.

    nr_iterations : int, default=100
        Maximum number of reweighting iterations.

    use_banded : bool, default=True
        If True, use the banded solver for Whittaker smoothing.
        Otherwise, use a sparse LU decomposition.

    max_iter_after_warmstart : int, default=20
        Maximum iterations allowed when warm-starting from previous weights.

    Methods
    -------
    fit(X, y=None)
        Fit the estimator to the input spectra.

    transform(X, y=None)
        Remove baselines from the input spectra.

    _calculate_baseline(x, w, max_iter)
        Internal method: compute the baseline for a single spectrum.

    References
    ----------
    [1] Sung-June Baek, Aaron Park, Young-Jin Ahn, Jaebum Choo.
        "Baseline correction using asymmetrically reweighted penalized
        least squares smoothing." Analyst 140 (1), 250–257 (2015).
    """

    def __init__(
        self,
        lam=1e4,
        ratio=1e-2,
        nr_iterations=100,
        use_banded=True,
        max_iter_after_warmstart=20,
    ):
        super().__init__(
            lam=lam,
            nr_iterations=nr_iterations,
            use_banded=use_banded,
            max_iter_after_warmstart=max_iter_after_warmstart,
        )
        self.ratio = ratio

    def fit(self, X: np.ndarray, y=None) -> "ArPls":
        """
        Fit ArPLS model to spectra.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            The input spectra to fit the model to.

        y : None
            Ignored.

        Returns
        -------
        self : ArPlS
            Fitted estimator.
        """
        return super().fit(X, y)

    def transform(self, X: np.ndarray, y=None, copy=True) -> np.ndarray:
        """Apply ArPLS baseline correction.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            The input spectra to transform.

        y : None
            Ignored.

        copy : bool, default=True
            If True, a copy of X is made before transforming.

        Returns
        -------
        X_transformed : np.ndarray of shape (n_samples, n_features)
            The baseline-corrected spectra.
        """
        return super().transform(X, y)

    def _calculate_baseline(
        self, x: np.ndarray, w: np.ndarray, max_iter: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Run ArPls iterations on a single spectrum.

        Parameters
        ----------
        x : ndarray
            Input spectrum.
        w : ndarray
            Initial weights.
        max_iter : int
            Maximum number of iterations.

        Returns
        -------
        z : ndarray
            Estimated baseline.
        w : ndarray
            Final weights.
        """
        for _ in range(max_iter):
            # Solve Whittaker
            z = self._solve_whittaker(x, w)

            # Calculate residuals
            d = x - z

            # Update weights
            dn = d[d < 0]

            # Early stopping if no negative residuals
            if dn.size == 0:
                break

            # Early stopping if std is zero
            m, s = dn.mean(), dn.std()
            if s == 0:
                break

            exponent = np.clip(2 * (d - (2 * s - m)) / s, -709, 709)
            wt = 1.0 / (1.0 + np.exp(exponent))

            # Early stopping if weights do not change
            if np.linalg.norm(w - wt) / np.linalg.norm(w) < self.ratio:
                w = wt
                break
            w = wt

        return z, w
