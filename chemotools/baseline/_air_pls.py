"""
The :mod:`chemotools.baseline._air_pls` module implements the Adaptive Iteratively Reweighted
Penalized Least Squares (AirPLS) baseline correction algorithm
"""

# Authors: Niklas Zell <nik.zoe@web.de>, Pau Cabaneros
# License: MIT

from typing import Callable, Literal, Optional
import numpy as np
from sklearn.utils._param_validation import Interval, Real, StrOptions

from ._base import _BaselineWhittakerMixin
from chemotools.smooth._base import _BaseWhittaker


class AirPls(_BaselineWhittakerMixin, _BaseWhittaker):
    """
    Adaptive Iteratively Reweighted Penalized Least Squares (AirPls) baseline correction.

    AirPls is a widely used algorithm for removing baselines from spectroscopic
    signals. It iteratively reweights residuals to suppress positive deviations
    (peaks) while adapting baseline estimates using an exponential weight update.
    A second-order difference operator (recommended) is used as the penalty term,
    ensuring the estimated baseline is smooth.

    The Whittaker smoothing step can be solved using either:

    - a **banded solver** (fast and memory-efficient, recommended for most spectra)
    - a **sparse LU solver** (more stable for ill-conditioned problems)

    For efficiency, AirPls supports warm-starting: when processing multiple spectra
    with similar baseline structure, weights from a previous fit can be reused,
    typically reducing the number of iterations required.

    Parameters
    ----------
    lam : float, default=1e4
        Regularization parameter controlling smoothness of the baseline.
        Larger values yield smoother baselines.

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
    [1] Z.-M. Zhang, S. Chen, Y.-Z. Liang.
        "Baseline correction using adaptive iteratively reweighted penalized
        least squares." Analyst 135 (5), 1138–1146 (2010).

    Examples
    --------
    >>> from chemotools.baseline import AirPls
    >>> from chemotools.datasets import load_fermentation_train
    >>> # Load sample data
    >>> X, _ = load_fermentation_train()
    >>> # Instantiate the transformer
    >>> transformer = AirPls(lam=1e4, nr_iterations=100)
    AirPls()
    >>> transformer.fit(X)
    >>> # Remove baselines
    >>> X_corrected = transformer.transform(X)
    """

    _parameter_constraints: dict = {
        "lam": [Interval(Real, 0, None, closed="both")],
        "nr_iterations": [Interval(Real, 1, None, closed="both")],
        "solver_type": StrOptions({"banded", "sparse"}),
        "max_iter_after_warmstart": [Interval(Real, 1, None, closed="both")],
    }

    def __init__(
        self,
        lam: float = 1e4,
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

    def fit(self, X: np.ndarray, y=None) -> "AirPls":
        """
        Fit AirPls model to spectra.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            The input spectra to fit the model to.

        y : None
            Ignored to align with API.

        Returns
        -------
        self : AirPls
            Fitted estimator.
        """
        return super().fit(X, y)

    def transform(self, X: np.ndarray, y=None) -> np.ndarray:
        """Apply AirPls baseline correction.

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
    ) -> "AirPls":
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
        self : AirPls
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
        Compute AirPls baseline for a single spectrum.

        Parameters
        ----------
        x : ndarray
            Input spectrum.
        w : ndarray
            Initial weights.
        max_iter : int
            Maximum iterations.

        Returns
        -------
        z : ndarray
            Estimated baseline.
        w : ndarray
            Final weights.
        """
        x_abs_sum = np.abs(x).sum()

        for i in range(max_iter):
            # Step 1: Whittaker smoothing
            z = self._solve_whittaker(x, w, solver=solver)

            # Step 2: Residuals
            d = x - z
            if np.all(d == 0):
                break

            # Step 3: Negative residuals only
            mask = d < 0
            d_neg = d * mask
            dssn = -d_neg.sum()

            # Stopping criterion
            if dssn < 0.001 * x_abs_sum:
                break

            # Step 4: Update weights
            new_w = np.zeros_like(w)
            if dssn > 0:
                new_w[mask] = np.exp(i * (-d_neg[mask]) / dssn)

                # Boundary handling
                neg_vals = d[mask]
                if neg_vals.size > 0:
                    new_w[0] = np.exp(i * (-neg_vals).max() / dssn)
                new_w[-1] = new_w[0]

            w = new_w

        return z, w
