from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np
from joblib import Parallel, delayed, effective_n_jobs


def parallel_map_reduce(
    items: Sequence[Any],
    *,
    n_jobs: int,
    map_fn: Callable[[Any], Any],
    reduce_fn: Callable[[list[Any]], Any],
) -> Any:
    """Map map_fn over items in parallel then reduce results in arrival order.

    Workers never share state; all signalling must go through return values.
    reduce_fn always receives results in the same order as items, regardless
    of actual execution order.
    """
    n_eff = effective_n_jobs(n_jobs)
    if n_eff <= 1 or len(items) <= 1:
        return reduce_fn([map_fn(item) for item in items])

    parts = Parallel(n_jobs=n_jobs)(delayed(map_fn)(item) for item in items)
    return reduce_fn(list(parts))


def row_chunks(X: np.ndarray, n_jobs: int) -> list[np.ndarray]:
    """Partition X into one contiguous row-chunk per effective job."""
    n_eff = max(1, effective_n_jobs(n_jobs))
    n_rows = X.shape[0]
    if n_eff <= 1 or n_rows < 2:
        return [X]
    chunk_size = max(1, (n_rows + n_eff - 1) // n_eff)
    return [X[start : start + chunk_size] for start in range(0, n_rows, chunk_size)]


def row_slices(n_rows: int, n_jobs: int) -> list[tuple[int, int]]:
    """Partition n_rows into (start, stop) pairs, one per effective job."""
    n_eff = max(1, effective_n_jobs(n_jobs))
    if n_eff <= 1 or n_rows < 2:
        return [(0, n_rows)]
    chunk_size = max(1, (n_rows + n_eff - 1) // n_eff)
    return [
        (start, min(start + chunk_size, n_rows))
        for start in range(0, n_rows, chunk_size)
    ]


def apply_rows(
    X: np.ndarray,
    *,
    n_jobs: int,
    fn: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    """Apply fn to row chunks of X and concatenate results.

    Output row count matches X; feature dimension is determined by fn.
    """
    return parallel_map_reduce(
        row_chunks(X, n_jobs),
        n_jobs=n_jobs,
        map_fn=fn,
        reduce_fn=lambda parts: np.concatenate(parts, axis=0),
    )


def apply_row_slices(
    *,
    n_rows: int,
    n_jobs: int,
    fn: Callable[[int, int], np.ndarray],
    empty_shape: tuple[int, int],
    dtype: np.dtype | type = np.float64,
) -> np.ndarray:
    """Apply fn(start, stop) to row slices and concatenate results.

    Supports outputs whose feature dimension differs from the input.
    Returns an empty array of empty_shape when n_rows == 0.
    """
    if n_rows == 0:
        return np.empty(empty_shape, dtype=np.dtype(dtype))
    return parallel_map_reduce(
        row_slices(n_rows, n_jobs),
        n_jobs=n_jobs,
        map_fn=lambda s: fn(s[0], s[1]),
        reduce_fn=lambda parts: np.concatenate(parts, axis=0),
    )
