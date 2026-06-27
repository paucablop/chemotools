"""
The :mod:`chemotools.baseline._as_ls` module implements the Asymmetric
Least Squares (AsLs) baseline correction algorithm
"""

# Authors: Niklas Zell <nik.zoe@web.de>, Pau Cabaneros
# License: MIT

from typing import Literal

import numpy as np
from sklearn.utils._param_validation import Interval, Real, StrOptions

from chemotools.smooth._base import _BaseWhittaker
from chemotools.utils._whittaker_solvers import WhittakerSolver

from ._base import _BaselineWhittakerMixin


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
        Backend used to solve the Whittaker linear system. Prefer
        ``"banded"`` (the default): it exploits the pentadiagonal structure
        of :math:`D^T D` with an O(n_features) LAPACK solve and is
        consistently faster. Use ``"sparse"`` only as a numerical fallback
        for ill-conditioned problems.

    max_iter_after_warmstart : int, default=20
        Maximum iterations allowed when warm-starting from previous weights.

    n_jobs : int, default=1
        Number of parallel jobs used during :meth:`transform`. Effective for
        both solver types because each sample is processed independently
        through the iteration loop. Benchmarks show roughly **4–5× speedup**
        with ``n_jobs=-1`` on 8 cores.

    Attributes
    ----------
    n_features_in_ : int
        The number of features in the input data.

    DtD_ : np.ndarray
        The precomputed banded representation of :math:`D^T D` for the
        second-order difference operator.

        * Stored as a banded representation (``solveh_banded``
          format) if ``solver_type='banded'``
        * Stored as a ``scipy.sparse`` CSC matrix if
          ``solver_type='sparse'``

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
        "solver_type": [StrOptions({"banded", "sparse"})],
        "max_iter_after_warmstart": [Interval(Real, 1, None, closed="both")],
        "n_jobs": _BaseWhittaker._parameter_constraints["n_jobs"],
    }

    def __init__(
        self,
        lam: float = 1e4,
        penalty: float = 1e-2,
        nr_iterations: int = 100,
        solver_type: Literal["banded", "sparse"] = "banded",
        max_iter_after_warmstart: int = 20,
        n_jobs: int = 1,
    ):
        _BaseWhittaker.__init__(self, lam=lam, solver_type=solver_type, n_jobs=n_jobs)
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
        solver: WhittakerSolver | None = None,
    ) -> "AsLs":
        """Fit core implementation: compute warm-start weights.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input spectra.
        y : None
            Ignored.
        solver : WhittakerSolver or None
            Whittaker solver instance, provided by ``_BaseWhittaker.fit``.

        Returns
        -------
        self : AsLs
            Fitted instance.
        """
        assert solver is not None
        self.w_init_ = self._compute_warmstart_weights(X, solver)
        return self

    def _transform_block(self, X_block: np.ndarray) -> np.ndarray:
        return self._apply_baseline_correction(X_block, self.solver_)

    def _calculate_baseline(
        self, x: np.ndarray, w: np.ndarray, max_iter: int, solver
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
            z = solver.solve(x, w)

            # Residuals
            d = x - z

            # Update weights
            new_w = np.where(d >= 0, self.penalty, 1 - self.penalty)

            # Convergence check
            if np.array_equal(new_w, w):
                break
            w = new_w

        return z, w
