"""
The :mod:`chemotools.baseline._as_ls` module implements the Asymmetric
Least Squares (AsLs) baseline correction algorithm
"""

# Authors: Niklas Zell <nik.zoe@web.de>, Pau Cabaneros
# License: MIT

from typing import Callable, Literal, Optional

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

    - a **banded solver** (fast and memory-efficient, recommended for most spectra)
    - a **sparse LU solver** (more stable for ill-conditioned problems)

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

    Attributes
    ----------
    n_features_in_ : int
        The number of features in the input data.

    DtD_ : np.ndarray
        The precomputed banded representation of :math:`D^T D` for the
        second-order difference operator.

        * Stored as a banded representation (``solveh_banded`` format) if ``solver_type='banded'``
        * Stored as a ``scipy.sparse`` CSC matrix if ``solver_type='sparse'``

    self.w_init_ : np.ndarray
        The weights set for warm-starting.

    References
    ----------
    [1] Sung-June Baek, Aaron Park, Young-Jin Ahn, Jaebum Choo.
        "Baseline correction using asymmetrically reweighted penalized
        least squares smoothing." Analyst 140 (1), 250–257 (2015).

    Examples
    --------
    >>> from chemotools.baseline import AsLs
    >>> from chemotools.datasets import load_fermentation_train
    >>> # Load sample data
    >>> X, _ = load_fermentation_train()
    >>> # Instantiate the transformer
    >>> transformer = AsLs(lam=1e4, nr_iterations=100)
    AsLs()
    >>> transformer.fit(X)
    >>> # Generate baseline-corrected data
    >>> X_corrected = transformer.transform(X)
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
            Ignored to align with API.

        Returns
        -------
        self : AsLs
            Fitted estimator.
        """
        return super().fit(X, y)

    def transform(self, X: np.ndarray, y=None) -> np.ndarray:
        """Apply AsLs baseline correction.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            The input spectra to transform.

        y : None
            Ignored to align with API.

        Returns
        -------
        X_transformed : np.ndarray of shape (n_samples, n_features)
            The baseline-corrected spectra.
        """
        return super().transform(X, y)

    def _fit_core(
        self,
        X: np.ndarray,
        y=None,
        nr_iterations: int = 1,
        solver: Optional[Callable] = None,
    ) -> "AsLs":
        """Fit core implementation: compute warm-start weights.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input spectra.
        y : None
            Ignored.
        nr_iterations : int
            Not used in this implementation.
        solver : Optional[Callable]
            Whittaker solver function.

        Returns
        -------
        self : AsLs
            Fitted instance.
        """
        self.w_init_ = self._compute_warmstart_weights(X, solver)
        return self

    def _transform_core(
        self,
        X: np.ndarray,
        y=None,
        nr_iterations: int = 1,
        solver: Optional[Callable] = None,
    ) -> np.ndarray:
        """Transform core implementation: apply baseline correction.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input spectra to correct.
        y : None
            Ignored.
        nr_iterations : int
            Not used in this implementation.
        solver : Callable
            Whittaker solver function.

        Returns
        -------
        X_corrected : np.ndarray of shape (n_samples, n_features)
            Baseline-corrected spectra.
        """
        return self._apply_baseline_correction(X, solver)

    def _calculate_baseline(
        self, x: np.ndarray, w: np.ndarray, max_iter: int, solver: Optional[Callable]
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
