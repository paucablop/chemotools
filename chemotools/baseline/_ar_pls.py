"""
The :mod:`chemotools.baseline._ar_pls` module implements the Asymmetrically Reweighted
Penalized Least Squares (ArPLS) baseline correction algorithm
"""

# Authors: Niklas Zell <nik.zoe@web.de>, Pau Cabaneros
# License: MIT

from typing import Callable, Literal, Optional
import numpy as np
from sklearn.utils._param_validation import Interval, Real, StrOptions

from ._base import _BaselineWhittakerMixin
from chemotools.smooth._base import _BaseWhittaker


class ArPls(_BaselineWhittakerMixin, _BaseWhittaker):
    """
    Asymmetrically Reweighted Penalized Least Squares (ArPLS) baseline correction.

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

    ratio : float, default=0.01
        Convergence threshold for weight updates.

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
    >>> from chemotools.baseline import ArPls
    >>> from chemotools.datasets import load_fermentation_train
    >>> # Load sample data
    >>> X, _ = load_fermentation_train()
    >>> # Instantiate the transformer
    >>> transformer = ArPls(lam=1e4, nr_iterations=100)
    ArPls()
    >>> transformer.fit(X)
    >>> # Generate baseline-corrected data
    >>> X_corrected = transformer.transform(X)
    """

    _parameter_constraints: dict = {
        "lam": [Interval(Real, 0, None, closed="both")],
        "ratio": [Interval(Real, 0, 1, closed="both")],
        "nr_iterations": [Interval(Real, 1, None, closed="both")],
        "solver_type": StrOptions({"banded", "sparse"}),
        "max_iter_after_warmstart": [Interval(Real, 1, None, closed="both")],
    }

    def __init__(
        self,
        lam: float = 1e4,
        ratio: float = 1e-2,
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
        self.ratio = ratio

    def fit(self, X: np.ndarray, y=None) -> "ArPls":
        """
        Fit ArPLS model to spectra.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            The input spectra to fit the model to.

        y : None
            Ignored to align with API.

        Returns
        -------
        self : ArPlS
            Fitted estimator.
        """
        return super().fit(X, y)

    def transform(self, X: np.ndarray, y=None) -> np.ndarray:
        """Apply ArPLS baseline correction.

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
    ) -> "ArPls":
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
        self : ArPls
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
            # Step 1: Whittaker smoothing
            z = self._solve_whittaker(x, w, solver=solver)

            # Step 2: Residuals
            d = x - z
            dn = d[d < 0]

            # Early stopping: no negative residuals
            if dn.size == 0:
                break

            # Early stopping: std is zero
            m, s = dn.mean(), dn.std()
            if s == 0:
                break

            # Step 3: Update weights
            exponent = np.clip(2 * (d - (2 * s - m)) / s, -709, 709)
            new_w = 1.0 / (1.0 + np.exp(exponent))

            # Convergence check
            if np.linalg.norm(w - new_w) / np.linalg.norm(w) < self.ratio:
                w = new_w
                break
            w = new_w

        return z, w
