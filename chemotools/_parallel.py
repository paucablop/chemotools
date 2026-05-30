from __future__ import annotations

from typing import Callable

import numpy as np
from joblib import Parallel, delayed, effective_n_jobs


def parallel_apply_by_rows(
    X: np.ndarray,
    *,
    n_jobs: int,
    block_fn: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    """Apply block_fn to row chunks of X, preserving row order."""
    n_effective_jobs = effective_n_jobs(n_jobs)
    if n_effective_jobs <= 1 or X.shape[0] < 2:
        return block_fn(X)

    n_samples = X.shape[0]
    chunk_size = max(1, (n_samples + n_effective_jobs - 1) // n_effective_jobs)
    chunks = [
        X[start : start + chunk_size] for start in range(0, n_samples, chunk_size)
    ]

    out_chunks = Parallel(n_jobs=n_jobs)(delayed(block_fn)(chunk) for chunk in chunks)

    X_out = np.empty_like(X)
    start = 0
    for c in out_chunks:
        stop = start + c.shape[0]
        X_out[start:stop] = c
        start = stop
    return X_out
