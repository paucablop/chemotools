import logging

import numpy as np

from ._base import _BaseWhittaker


logger = logging.getLogger(__name__)


class ArPls(_BaseWhittaker):
    """ARPLS baseline correction with logistic weight update."""

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
        """Fit ARPLS model to spectra."""
        return super().fit(X, y)

    def transform(self, X: np.ndarray, y=None, copy=True) -> np.ndarray:
        """Apply ARPLS baseline correction."""
        return super().transform(X, y)

    def _calculate_baseline(
        self, x: np.ndarray, w: np.ndarray, max_iter: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Run ARPLS iterations on a single spectrum.

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
