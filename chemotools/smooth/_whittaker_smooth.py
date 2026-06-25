"""
The :mod:`chemotools.smooth._whittaker_smooth` module implements
the Whittaker smoothing algorithm.
"""

# Authors: Niklas Zell <nik.zoe@web.de>, Pau Cabaneros
# License: MIT

from typing import Literal

import numpy as np
from sklearn.utils._param_validation import Interval, Real, StrOptions

from chemotools._parallel import apply_rows
from chemotools.utils._whittaker_solvers import WhittakerSolver

from ._base import _BaseWhittaker


class WhittakerSmooth(_BaseWhittaker):
    """
    Whittaker smoothing for noise reduction and signal trend estimation.

    Whittaker smoothing is a penalized least squares method that estimates
    smooth trends from noisy data by balancing fidelity to the input signal
    with a smoothness constraint. A second-order difference operator is used
    as the penalty term, ensuring that the estimated signal is smooth while
    preserving overall shape.

    The Whittaker smoothing step can be solved using either:
    - a **banded solver** (fast and memory-efficient, recommended for most spectra), or
    - a **sparse LU solver** (more stable for ill-conditioned problems).

    Optional weights can be provided to emphasize or downweight certain
    observations during smoothing. If no weights are supplied, all points
    are treated equally.

    Parameters
    ----------
    lam : float, default=1e4
        Regularization parameter controlling smoothness of the fitted signal.
        Larger values yield smoother trends.

    weights : ndarray of shape (n_features,), optional, default=None
        Non-negative weights applied to each observation. If None,
        all observations are weighted equally.

    solver_type : Literal["banded", "sparse"], default="banded"
        Backend used to solve the Whittaker linear system. Prefer
        ``"banded"`` (the default): it solves all rows in a single batched
        LAPACK call and is roughly **27× faster** than ``"sparse"`` at
        scale. Use ``"sparse"`` only as a numerical fallback for
        ill-conditioned problems.

    n_jobs : int, default=1
        Number of parallel jobs used during :meth:`transform`.
        Only effective when ``solver_type="sparse"``: rows are split across
        workers via joblib. When ``solver_type="banded"`` (the default), a
        single vectorised LAPACK batch solve is used regardless of this value,
        because it is already faster than spawning parallel workers.
        With ``solver_type="sparse"`` and ``n_jobs=-1``, benchmarks show
        roughly **4× speedup** on 8 cores.

    Attributes
    ----------
    n_features_in_ : int
        The number of features in the training data.

    References
    ----------
    [1] Eilers, P.H. (2003).
        "A perfect smoother." Analytical Chemistry 75 (14), 3631–3636.

    Examples
    --------
    >>> from chemotools.datasets import load_fermentation_train
    >>> from chemotools.smooth import WhittakerSmooth
    >>> # Load sample data
    >>> X, _ = load_fermentation_train()
    >>> # Initialize WhittakerSmooth
    >>> ws = WhittakerSmooth()
    WhittakerSmooth()
    >>> # Fit and transform the data
    >>> X_smoothed = ws.fit_transform(X)
    """

    _parameter_constraints: dict = {
        "lam": [Interval(Real, 0, None, closed="neither")],
        "weights": ["array-like", None],
        "solver_type": [StrOptions({"banded", "sparse"})],
        "n_jobs": _BaseWhittaker._parameter_constraints["n_jobs"],
    }

    def __init__(
        self,
        lam: float = 1e4,
        weights: np.ndarray | None = None,
        solver_type: Literal["banded", "sparse"] = "banded",
        n_jobs: int = 1,
    ):
        super().__init__(
            lam=lam, weights=weights, solver_type=solver_type, n_jobs=n_jobs
        )

    def fit(self, X: np.ndarray, y=None) -> "WhittakerSmooth":
        """
        Fit the Whittaker smoother to input data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input data matrix, where rows correspond to samples
            and columns correspond to features (e.g., spectra).

        y : None
            Ignored, present for API consistency with scikit-learn.

        Returns
        -------
        self : WhittakerSmooth
            Fitted estimator.
        """
        return super().fit(X, y)

    def transform(self, X: np.ndarray, y=None) -> np.ndarray:
        """
        Apply Whittaker smoothing to input data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input data matrix to smooth.

        y : None
            Ignored, present for API consistency with scikit-learn.

        Returns
        -------
        X_transformed : ndarray of shape (n_samples, n_features)
            The smoothed version of the input data.
        """
        from sklearn.utils.validation import check_is_fitted, validate_data

        check_is_fitted(self, ["DtD_", "solver_", "weights_"])
        X_ = validate_data(
            self, X, ensure_2d=True, copy=True, reset=False, dtype=np.float64
        )
        # Keep the fast batched LAPACK path for banded systems.
        if self.solver_type == "banded":
            return self.solver_.solve_batch(X_, self.weights_)

        # Sparse backend solves one row at a time; chunk rows across jobs.
        return apply_rows(X_, n_jobs=self.n_jobs, fn=self._transform_block)

    def _fit_core(
        self,
        X: np.ndarray,
        y=None,
        solver: WhittakerSolver | None = None,
    ) -> "WhittakerSmooth":
        """
        Core fitting logic for Whittaker smoothing.

        Stores the observation weights to be used in subsequent
        smoothing operations. If no custom weights were provided,
        uniform weights are applied.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input data matrix.

        y : None
            Ignored.

        solver : WhittakerSolver or None
            Whittaker solver instance, provided by ``_BaseWhittaker.fit``.
            Not used by this implementation.

        Returns
        -------
        self : WhittakerSmooth
            Fitted smoother with stored weights.
        """
        # Default weights if not provided
        self.weights_ = (
            self.weights if self.weights is not None else np.ones(X.shape[1])
        )
        return self

    def _transform_block(self, X_block: np.ndarray) -> np.ndarray:
        return self.solver_.solve_batch(X_block, self.weights_)
