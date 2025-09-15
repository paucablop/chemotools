import logging

import numpy as np

from ._base import _BaseWhittaker


logger = logging.getLogger(__name__)


class AirPls(_BaseWhittaker):
    """AirPLS baseline correction with exponential weight update."""

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
        """Fit AirPLS model to spectra."""
        return super().fit(X, y)

    def transform(self, X: np.ndarray, y=None) -> np.ndarray:
        """Apply AirPLS baseline correction."""
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
