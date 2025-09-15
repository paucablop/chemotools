from abc import ABC, abstractmethod
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


def _precompute_DtD_sparse(N: int):
    D = sp.diags([1, -2, 1], [0, 1, 2], shape=(N - 2, N), format="csc")
    return D.T @ D


def _whittaker_smooth_banded(x, w, lam, DtD_ab):
    """Solve (diag(w) + lam*D^T D) z = w*x with banded solver."""
    ab = np.empty_like(DtD_ab)
    ab[...] = DtD_ab
    ab[2, :] = lam * ab[2, :] + w  # main diag updated
    ab[1, 1:] = lam * ab[1, 1:]  # superdiag
    ab[0, 2:] = lam * ab[0, 2:]  # 2nd superdiag
    return solveh_banded(ab, w * x, lower=False, overwrite_ab=True, overwrite_b=True)


def _whittaker_smooth_sparse(x, w, lam, DtD_sparse):
    """Fallback: sparse LU solve."""
    N = len(x)
    H = lam * DtD_sparse
    W = sp.diags(w, 0, shape=(N, N), format="csc")
    C = (H + W).tocsc()
    solver = splu(C)
    return solver.solve(w * x)


class _BaseWhittaker(TransformerMixin, OneToOneFeatureMixin, BaseEstimator, ABC):
    """Abstract base class for Whittaker-based baseline correction."""

    def __init__(
        self,
        lam=10000.0,
        nr_iterations=100,
        use_banded=True,
        max_iter_after_warmstart=20,
    ):
        self.lam = lam
        self.nr_iterations = nr_iterations
        self.use_banded = use_banded
        self.max_iter_after_warmstart = max_iter_after_warmstart

    def fit(self, X, y=None):
        """Fit model to data, precomputing matrices and warm-start weights."""
        X = validate_data(
            self, X, y="no_validation", ensure_2d=True, reset=True, dtype=np.float64
        )

        n_features = X.shape[1]
        self.DtD_ab_ = _precompute_DtD_banded(n_features)

        # warm-start weights from first spectrum
        x0 = X[0]
        _, w = self._calculate_baseline(x0, np.ones_like(x0), self.nr_iterations)
        self.w_init_ = w
        return self

    def transform(self, X, y=None):
        """Apply baseline correction to input spectra."""
        check_is_fitted(self, ["DtD_ab_", "w_init_"])
        X_ = validate_data(
            self, X, y="no_validation", ensure_2d=True, copy=True, reset=False
        )

        for i, x in enumerate(X_):
            z, _ = self._calculate_baseline(
                x,
                self.w_init_.copy(),
                max_iter=min(self.nr_iterations, self.max_iter_after_warmstart),
            )
            X_[i] = x - z
        return X_

    @abstractmethod
    def _calculate_baseline(self, x, w, max_iter):
        """Subclasses must implement algorithm-specific baseline estimation."""
        ...
