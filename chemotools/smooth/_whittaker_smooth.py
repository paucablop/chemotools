"""
The :mod:`chemotools.smooth._whittaker_smooth` module implements the Whittaker smoothing algorithm.
"""

# Authors: Niklas Zell <nik.zoe@web.de>, Pau Cabaneros
# License: MIT

from typing import Callable, Literal
import numpy as np
from sklearn.utils._param_validation import Interval, StrOptions, Real

from chemotools.utils._linear_algebra import (
    whittaker_smooth_banded,
)
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
        If "banded", use the banded solver for Whittaker smoothing.
        If "sparse", use a sparse LU decomposition.

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
    }

    def __init__(
        self,
        lam: float = 1e4,
        weights: np.ndarray | None = None,
        solver_type: Literal["banded", "sparse"] = "banded",
    ):
        super().__init__(lam=lam, weights=weights, solver_type=solver_type)

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
        return super().transform(X, y)

    def _fit_core(
        self,
        X: np.ndarray,
        y=None,
        nr_iterations: int = 1,
        solver: Callable = whittaker_smooth_banded,
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

        nr_iterations : int, default=1
            Not used. Present for API consistency with subclasses.

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

    def _transform_core(
        self,
        X: np.ndarray,
        y=None,
        nr_iterations: int = 1,
        solver: Callable = whittaker_smooth_banded,
    ) -> np.ndarray:
        """
        Core transformation logic for Whittaker smoothing.

        Applies Whittaker smoothing to each input sample using
        the stored weights and regularization parameter.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input data to smooth.

        y : None
            Ignored.

        nr_iterations : int, default=1
            Not used. Present for API consistency with subclasses.

        Returns
        -------
        X_smooth : ndarray of shape (n_samples, n_features)
            The smoothed input data.
        """
        for i, x in enumerate(X):
            X[i] = self._solve_whittaker(x, self.weights_, solver)
        return X
