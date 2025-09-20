"""
The :mod:`chemotools.baseline._as_ls` module implements the Asymmetric
Least Squares (AsLs) baseline correction algorithm
"""

# Authors: Niklas Zell <nik.zoe@web.de>, Pau Cabaneros
# License: MIT

from typing import Callable, Literal

import numpy as np
from sklearn.utils._param_validation import Interval, Real, StrOptions

from ._base import _BaselineWhittakerMixin
from chemotools.smooth._base import _BaseWhittaker


class AsLs(_BaselineWhittakerMixin, _BaseWhittaker):
    """
    Asymmetric Least Squares (AsLs) baseline correction.

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

    penalty : float, default=0.01
        The asymmetry parameter. It is recommended to set between 0.001 and 0.1 [1]

    nr_iterations : int, default=100
        Maximum number of reweighting iterations.

    solver_type : Literal["banded", "sparse"], default="banded"
        If "banded", use the banded solver for Whittaker smoothing.
        If "sparse", use a sparse LU decomposition.

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

    Examples
    --------
    >>> from chemotools.baseline import AsLs
    >>> import numpy as np
    >>> X = np.array([[1, 2, 3, 4, 5]])
    >>> asls = AsLs()
    >>> X_corrected = asls.fit_transform(X)

    References
    ----------
    [1] Sung-June Baek, Aaron Park, Young-Jin Ahn, Jaebum Choo.
        "Baseline correction using asymmetrically reweighted penalized
        least squares smoothing." Analyst 140 (1), 250–257 (2015).
    """

    _parameter_constraints: dict = {
        "lam": [Interval(Real, 0, None, closed="both")],
        "penalty": [Interval(Real, 0, 1, closed="both")],
        "nr_iterations": [Interval(Real, 1, None, closed="both")],
        "solver_type": StrOptions({"banded", "sparse"}),
        "max_iter_after_warmstart": [Interval(Real, 1, None, closed="both")],
    }

    def __init__(
        self,
        lam: float = 1e4,
        penalty: float = 1e-2,
        nr_iterations: int = 100,
        solver_type: Literal["banded", "sparse"] = "banded",
        max_iter_after_warmstart: int = 20,
    ):
        _BaseWhittaker.__init__(self, lam=lam, solver_type=solver_type)
        _BaselineWhittakerMixin.__init__(
            self,
            nr_iterations=nr_iterations,
            max_iter_after_warmstart=max_iter_after_warmstart,
        )
        self.penalty = penalty

    def fit(self, X: np.ndarray, y=None) -> "AsLs":
        """
        Fit AsLs model to spectra.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            The input spectra to fit the model to.

        y : None
            Ignored.

        Returns
        -------
        self : AsLs
            Fitted estimator.
        """
        return super().fit(X, y)

    def transform(self, X: np.ndarray, y=None, copy=True) -> np.ndarray:
        """Apply AsLs baseline correction.

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
        self, x: np.ndarray, w: np.ndarray, max_iter: int, solver: Callable
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Run AsLs iterations on a single spectrum.

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
            # Whittaker smoothing
            z = self._solve_whittaker(x, w, solver=solver)

            # Residuals
            d = x - z

            # Update weights
            new_w = np.where(d >= 0, self.penalty, 1 - self.penalty)

            # Convergence check
            if np.array_equal(new_w, w):
                break
            w = new_w

        return z, w
