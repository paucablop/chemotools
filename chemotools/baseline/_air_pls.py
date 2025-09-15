import logging

import numpy as np

from ._base import _BaseWhittaker


logger = logging.getLogger(__name__)


class AirPls(_BaseWhittaker):
    """
    Adaptive Iteratively Reweighted Penalized Least Squares (AirPls) baseline correction.

    AirPls is a widely used algorithm for removing baselines from spectroscopic
    signals. It iteratively reweights residuals to suppress positive deviations
    (peaks) while adapting baseline estimates using an exponential weight update.
    A second-order difference operator (recommended) is used as the penalty term,
    ensuring the estimated baseline is smooth.

    The Whittaker smoothing step can be solved using either:
    - a **banded solver** (fast and memory-efficient, recommended for most spectra), or
    - a **sparse LU solver** (more stable for ill-conditioned problems).

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
        Internal method: compute the baseline for a single spectrum
        using the AirPls exponential reweighting scheme.

    References
    ----------
    [1] Z.-M. Zhang, S. Chen, Y.-Z. Liang.
        "Baseline correction using adaptive iteratively reweighted penalized
        least squares." Analyst 135 (5), 1138–1146 (2010).
    """

    def __init__(
        self,
        lam: float = 1e4,
        nr_iterations: int = 100,
        use_banded: bool = True,
        max_iter_after_warmstart: int = 20,
    ):
        super().__init__(
            lam=lam,
            nr_iterations=nr_iterations,
            use_banded=use_banded,
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
            Ignored.

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
        Run vectorized AirPLS iterations (keeps original exponential weighting).

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
        x_abs_sum = np.abs(x).sum()  # reused for stopping

        for i in range(max_iter):
            # Solve Whittaker
            z = self._solve_whittaker(x, w)

            # Calculate residuals
            d = x - z

            # Early exit if all residuals are non-negative
            if np.all(d == 0):
                break

            # vectorized negative mask: mask True where d < 0
            mask = d < 0
            # negative part (non-positive elsewhere)
            d_neg = d * mask  # negatives are negative numbers, positives zeroed
            dssn = -d_neg.sum()  # same as abs(sum(d[d<0]))

            # stopping criterion (same threshold as original)
            if dssn < 0.001 * x_abs_sum:
                break

            # ensure we don't try to use iteration index beyond configured nr_iterations
            if i == self.nr_iterations - 1:
                break

            # build new weights vectorized
            new_w = np.zeros_like(w)
            if dssn > 0:
                # compute exponential only for negative positions without allocating
                # a masked subarray repeatedly (vectorized)
                # note: i is 0-based; original code used i in exp, keep same semantics
                # absolute of negative entries is -d_neg (since those entries are negative)
                new_w[mask] = np.exp(i * (-d_neg[mask]) / dssn)

                # boundary handling: use max of negative d (most negative -> largest abs)
                # extract negative d values once
                neg_vals = d[mask]
                if neg_vals.size > 0:
                    new_w[0] = np.exp(i * (-neg_vals).max() / dssn)
                new_w[-1] = new_w[0]

            w = new_w

        # return last z and weights
        return z, w
