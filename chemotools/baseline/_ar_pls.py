import logging
import numpy as np
from scipy.linalg import solveh_banded
import scipy.sparse as sp
from scipy.sparse.linalg import splu

from sklearn.base import BaseEstimator, TransformerMixin, OneToOneFeatureMixin
from sklearn.utils.validation import check_is_fitted, validate_data

logger = logging.getLogger(__name__)


def _precompute_DtD_banded(N: int):
    """Precompute the banded representation of D^T D (upper form, u=2)."""
    if N < 3:
        return np.zeros((3, N))

    if N >= 5:
        DtD_main = np.concatenate(([1, 5], np.repeat(6, N - 4), [5, 1]))
    elif N == 4:
        DtD_main = np.array([1, 5, 5, 1])
    else:  # N == 3
        DtD_main = np.array([1, 5, 1])

    if N == 3:
        DtD_sup1 = np.array([-2, -2])
    elif N == 4:
        DtD_sup1 = np.array([-2, -4, -2])
    else:
        DtD_sup1 = np.concatenate(([-2], np.repeat(-4, N - 3), [-2]))

    DtD_sup2 = np.ones(N - 2)

    ab = np.zeros((3, N))
    ab[0, 2:] = DtD_sup2
    ab[1, 1:] = DtD_sup1
    ab[2, :] = DtD_main
    return ab


def _whittaker_smooth_banded(x, w, lam, DtD_ab):
    """Solve (diag(w) + lam*D^T D) z = w*x with banded solver."""
    ab = np.empty_like(DtD_ab)
    ab[...] = DtD_ab
    ab[2, :] = lam * ab[2, :] + w  # main diag updated
    ab[1, 1:] = lam * ab[1, 1:]  # superdiag
    ab[0, 2:] = lam * ab[0, 2:]  # 2nd superdiag
    return solveh_banded(ab, w * x, lower=False, overwrite_ab=True, overwrite_b=True)


def _whittaker_smooth_sparse(x, w, lam):
    """Fallback: sparse LU solve."""
    N = len(x)
    D = sp.diags([1, -2, 1], [0, 1, 2], shape=(N - 2, N), format="csc")
    H = lam * (D.T @ D)
    W = sp.diags(w, 0, shape=(N, N), format="csc")
    C = (H + W).tocsc()
    solver = splu(C)
    return solver.solve(w * x)


class ArPls(TransformerMixin, OneToOneFeatureMixin, BaseEstimator):
    """ARPLS with banded Whittaker smoothing and warm-started weights."""

    def __init__(
        self,
        lam=1e4,
        ratio=0.01,
        nr_iterations=100,
        use_banded=True,
        max_iter_after_warmstart=20,
    ):
        self.lam = lam
        self.ratio = ratio
        self.nr_iterations = nr_iterations
        self.use_banded = use_banded
        self.max_iter_after_warmstart = max_iter_after_warmstart

    def fit(self, X, y=None):
        X = validate_data(
            self, X, y="no_validation", ensure_2d=True, reset=True, dtype=np.float64
        )

        N = X.shape[1]
        self.DtD_ab_ = _precompute_DtD_banded(N)

        # Warm-start weights from first spectrum
        x0 = X[0]
        z, w = self._run_arpls(x0, np.ones_like(x0), self.nr_iterations)
        self.w_init_ = w
        return self

    def transform(self, X, y=None):
        check_is_fitted(self, ["DtD_ab_", "w_init_"])
        X_ = validate_data(
            self, X, y="no_validation", ensure_2d=True, copy=True, reset=False
        )
        for i, x in enumerate(X_):
            z, _ = self._run_arpls(
                x,
                self.w_init_.copy(),
                max_iter=min(self.nr_iterations, self.max_iter_after_warmstart),
            )
            X_[i] = x - z
        return X_

    def _run_arpls(self, x, w, max_iter):
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
                    z = _whittaker_smooth_sparse(x, w, self.lam)
            except Exception as e:
                logger.debug("Banded solver failed (%s); fallback to sparse LU.", e)
                z = _whittaker_smooth_sparse(x, w, self.lam)

            d = x - z
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
