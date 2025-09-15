import logging

from ._base import (
    _BaseWhittaker,
    _whittaker_smooth_banded,
    _whittaker_smooth_sparse,
    _precompute_DtD_sparse,
)
import numpy as np

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

    def fit(self, X, y=None):
        """Fit ARPLS model to spectra."""
        return super().fit(X, y)

    def transform(self, X, y=None):
        """Apply ARPLS baseline correction."""
        return super().transform(X, y)

    def _calculate_baseline(self, x, w, max_iter):
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
            try:
                if self.use_banded:
                    z = _whittaker_smooth_banded(x, w, self.lam, self.DtD_ab_)
                else:
                    z = _whittaker_smooth_sparse(x, w, self.lam, self.DtD_ab)
            except Exception as e:
                logger.debug("Banded solver failed (%s); fallback to sparse LU.", e)
                DtD_ab = _precompute_DtD_sparse(self.n_features_in_)
                z = _whittaker_smooth_sparse(x, w, self.lam, DtD_ab)

            d = x - z

            #
            dn = d[d < 0]
            if dn.size == 0:
                break

            m, s = dn.mean(), dn.std()
            if s == 0:
                break

            exponent = np.clip(2 * (d - (2 * s - m)) / s, -709, 709)
            wt = 1.0 / (1.0 + np.exp(exponent))

            # Early stopping
            if np.linalg.norm(w - wt) / np.linalg.norm(w) < self.ratio:
                w = wt
                break
            w = wt

        return z, w
