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


def parallel_apply_by_row_slices(
    *,
    n_rows: int,
    n_jobs: int,
    block_fn: Callable[[int, int], np.ndarray],
    empty_shape: tuple[int, int],
    dtype: np.dtype | type = np.float64,
) -> np.ndarray:
    """Apply block_fn(start, stop) to row slices, preserving row order.

    Unlike ``parallel_apply_by_rows``, this helper supports outputs whose
    feature dimension differs from the input and lets callers access external
    per-row state (for example, aligned metadata arrays).
    """
    if n_rows == 0:
        return np.empty(empty_shape, dtype=dtype)

    n_effective_jobs = effective_n_jobs(n_jobs)
    if n_effective_jobs <= 1 or n_rows < 2:
        return block_fn(0, n_rows)

    chunk_size = max(1, (n_rows + n_effective_jobs - 1) // n_effective_jobs)
    row_slices = [
        (start, min(start + chunk_size, n_rows))
        for start in range(0, n_rows, chunk_size)
    ]

    out_chunks = Parallel(n_jobs=n_jobs)(
        delayed(block_fn)(start, stop) for start, stop in row_slices
    )
    return np.concatenate(out_chunks, axis=0)
