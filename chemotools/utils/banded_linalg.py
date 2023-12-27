from numbers import Integral

import numpy as np
from scipy.sparse import spmatrix
from sklearn.utils import check_array, check_scalar


def _check_full_arr_n_diag_counts_for_lu_banded(
    a_shape: tuple[int, int],
    l_and_u: tuple[int, int],
) -> None:
    """Validates the shape of a full array and the number of sub- and superdiagonals
    for LU-decomposition of a banded (sparse) matrix.
    """
    num_rows, num_cols = a_shape
    num_low_diags, num_upp_diags = l_and_u

    check_scalar(
        x=num_rows,
        name="num_rows",
        target_type=Integral,
        min_val=1,
        include_boundaries="left",
    )
    check_scalar(
        x=num_cols,
        name="num_cols",
        target_type=Integral,
        min_val=1,
        include_boundaries="left",
    )
    check_scalar(
        x=num_low_diags,
        name="num_low_diags",
        target_type=Integral,
        min_val=0,
        max_val=num_rows - 1,
        include_boundaries="both",
    )
    check_scalar(
        x=num_upp_diags,
        name="num_upp_diags",
        target_type=Integral,
        min_val=0,
        max_val=num_rows - 1,
        include_boundaries="both",
    )

    if num_rows != num_cols:
        raise ValueError(f"\nThe matrix must be square, but it has shape {a_shape}.")
    # else nothing


def conv_to_lu_banded_storage(
    a: np.ndarray | spmatrix,
    l_and_u: tuple[int, int],
) -> np.ndarray:
    """Converts a (sparse) square banded matrix A to its banded storage required for
    LU-decomposition in LAPACK-routines like the function ``lu_banded`` or SciPy's
    ``solve_banded``. This format is identical for pentapy where it is referred to as
    "column-wise flattened".
    Cholesky-decompositions require a different format.

    Parameters
    ----------
    a : np.ndarray or sparse matrix of shape (n, n)
        A square banded NumPy-2D-Array or SciPy sparse matrix. "Square" means that the
        row count equals the column count while "banded" implies that only the main
        diagonal and a few sub- and/or superdiagonals are non-zero (see `l_and_u`).
    l_and_u : tuple[int, int]
        The number of "non-zero" sub- (first) and superdiagonals (second element) aside
        the main diagonal which does not need to be considered here. "Non-zero" can be
        a bit misleading in this context. These numbers should count up to the diagonal
        after which all following diagonals are zero. Zero-diagonals that come before
        still need to be included.
        Wrong specification of this can lead to non-zero-diagonals being ignored or
        zero-diagonals being included which corrupts the results or reduces the
        performance.

    Returns
    -------
    ab : np.ndarray of shape (l_and_u[0] + 1 + l_and_u[1], n)
        A NumPy-2D-Array resembling `a` in banded storage format (see Notes).

    Raises
    ------
    ValueError
        If `a` is not square.
    ValueError
        If the number of rows of `a` does not match the number of rows given by
        the diagonal number.

    Notes
    -----
    For LAPACK LU-decomposition, the matrix `a` is stored in `ab` using the matrix
    diagonal ordered form:

        ```python
        ab[u + i - j, j] == a[i,j] # see below for u
        ```

    An example of `ab` (shape of a is ``(7,7)``, `u`=3 superdiagonals, `l`=2
    subdiagonals) looks like:

        ```python
         *    *    *   a03  a14  a25  a36
         *    *   a02  a13  a24  a35  a46
         *   a01  a12  a23  a34  a45  a56   # ^ superdiagonals
        a00  a11  a22  a33  a44  a55  a66   # main diagonal
        a10  a21  a32  a43  a54  a65   *    # v subdiagonals
        a20  a31  a42  a53  a64   *    *
        ```

    where all entries marked with ``*`` are ``0`` when returned by this function.
    Internally LAPACK relies on an expanded version of this format to perform inplace
    operations, but the respective functions handle the conversion themselves.

    """

    # the matrix is checked for being square and for having the correct number of rows
    num_low_diags, num_upp_diags = l_and_u
    a = check_array(array=a, accept_sparse=True, ensure_2d=True)
    _check_full_arr_n_diag_counts_for_lu_banded(
        a_shape=a.shape, l_and_u=l_and_u  # type: ignore
    )

    # first, the number of lower and upper diagonals is extracted and turned into two
    # offset vectors
    main_diag_idx = num_upp_diags
    num_cols = a.shape[-1]

    # now, the diagonal extraction method is specified based and the banded storage is
    # filled by it
    diag_method = a.diagonal  # type: ignore
    ab = np.zeros(
        shape=(num_low_diags + 1 + num_upp_diags, num_cols),
        dtype=a.dtype,  # type: ignore
    )

    # the superdiagonals and the main diagonal
    for offset in range(num_upp_diags, -1, -1):
        ab[main_diag_idx - offset, offset::] = diag_method(offset)

    # the subdiagonals
    for offset in range(-1, -num_low_diags - 1, -1):
        ab[main_diag_idx - offset, 0:offset] = diag_method(offset)

    return ab


def slodget_cho_banded(decomposition: tuple[np.ndarray, bool]) -> tuple[float, float]:
    """Computes the logarithm of the absolute value of the determinant of a banded
    hermitian matrix `A` using its Cholesky-decomposition. This is way more efficient
    than computing the determinant directly because the Cholesky factors' main
    diagonals already encode the determinant as the product of the diagonal entries.

    Parameters
    ----------
    (cb, lower) : tuple, (np.ndarray, bool)
        `cb` is a NumPy-2D-Array resembling the Cholesky-decomposition of `A` in banded
        storage format as returned by ``cholesky_banded``.
        `lower` is a boolean indicating whether the Cholesky-decomposition the lower
        triangular form (``True``) or the upper triangular form was of `A` was used
        (``False``).

    Returns
    -------
    sign : float
        A number representing the sign of the determinant. It is always +1 since
        the matrix under consideration is positive definite.
    logabsdet : float
        The natural log of the absolute value of the determinant. It cannot be zero
        since the matrix under consideration is positive definite.

    """

    lower = decomposition[1]
    main_diag_idx = 0 if lower else -1

    return 1.0, 2.0 * np.sum(np.log(decomposition[0][main_diag_idx, ::]))
