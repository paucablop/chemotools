"""
This utility submodule provides functions for the linear algebra with banded matrices,
namely

- conversion from the upper banded storage for LAPACK's banded Cholesky decomposition
    to the banded storage for LAPACK's banded LU decomposition,
- wrappers for SciPy's LAPACK-routines for the banded Cholesky decomposition and the
    corresponding linear solver,
- LU decomposition of a banded matrix and the corresponding linear solver,
- computation of the log-determinant of a banded matrix using its Cholesky or LU
    decomposition

The decomposition functions return dataclasses that facilitate the handling of the
factorizations.

"""

### Imports ###

from numbers import Integral

import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import cho_solve_banded as scipy_cho_solve_banded
from scipy.linalg import cholesky_banded as scipy_cholesky_banded
from scipy.linalg import lapack
from scipy.sparse import spmatrix
from sklearn.utils import check_array, check_scalar

from chemotools.utils.models import BandedCholeskyFactorization, BandedLUFactorization

### Type Aliases ###

LAndUBandCounts = tuple[int, int]


### Auxiliary Functions ###


def _datacopied(arr, original):
    """
    Strictly check for ``arr`` not sharing any data wit ``original``, under the
    assumption that ``arr = asarray(original)``

    Was copied from Scipy to be consistent in the LAPACK-wrappers implemented here.

    """

    if arr is original:
        return False

    if not isinstance(original, np.ndarray) and hasattr(original, "__array__"):
        return False

    return arr.base is None


def _check_full_arr_n_diag_counts_for_lu_banded(
    a_shape: tuple[int, int],
    l_and_u: tuple[int, int],
) -> None:
    """Validates the shape of a full array and the number of sub- and superdiagonals
    for LU decomposition of a banded (sparse) matrix.
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


def conv_upper_chol_banded_to_lu_banded_storage(
    ab: np.ndarray,
) -> tuple[LAndUBandCounts, np.ndarray]:
    """
    Converts the upper banded storage format used by LAPACK's banded Cholesky
    decomposition to the banded storage format used by LAPACK's banded LU
    decomposition.

    Parameters
    ----------
    ab : np.ndarray of shape (n_upp_bands + 1, n_cols)
        The matrix ``A`` stored in the upper banded storage format used by LAPACK's
        banded Cholesky decomposition (see Notes for details).

    Returns
    -------
    l_and_u : (int, int)
        The number of sub- (first) and superdiagonals (second element) aside the main
        diagonal which does not need to be considered here.
    ab : np.ndarray of shape (l_and_u[0] + 1 + l_and_u[1], n_cols)
        The matrix ``A`` stored in the banded storage format used by LAPACK's banded LU
        decomposition (see Notes for details).

    Notes
    -----
    The upper diagonal ordered form for LAPACK's Cholesky decomposition is given by the
    following ordering

    ```python
    ab[u + i - j, j] == a[i,j]
    ```

    e.g., for a symmetric matrix ``A`` of shape (7, 7) with in total 3 superdiagonals,
    3 subdiagonals, and the main diagonal, the ordering is as follows:

    ```python
    *   *   *   a03 a14 a25 a36
    *   *   a02 a13 a24 a35 a46
    *   a01 a12 a23 a34 a45 a56 # ^ superdiagonals
    a00 a11 a22 a33 a44 a55 a66 # main diagonal
    ```

    where each `*` denotes a zero element.

    For LAPACK's LU decomposition, the matrix `A` is stored in `ab` using the matrix
    diagonal ordered form:

    ```python
    ab[u + i - j, j] == a[i,j]
    ```

    The example from above would then look like this where basicall, all the
    superdiagonal rows are just copied to the subdiagonal rows and moved to the left so
    that the first non-zero element of each row is in the first column:

    ```python
    *   *   *   a03 a14 a25 a36
    *   *   a02 a13 a24 a35 a46
    *   a01 a12 a23 a34 a45 a56 # ^ superdiagonals
    a00 a11 a22 a33 a44 a55 a66 # main diagonal
    a01 a12 a23 a34 a45 a56 *   # v subdiagonals
    a02 a13 a24 a35 a46 *   *
    a03 a14 a25 a36 *   *   *
    ```

    where all entries marked with `*` are as well zero elements although they will be
    set to arbitrary values by this function.

    """

    # an Array is initialised to store the subdiagonal part
    num_low_diags = ab.shape[0] - 1
    main_diag_idx = num_low_diags
    n_cols = ab.shape[1]
    ab_subdiags = np.zeros(shape=(num_low_diags, n_cols), dtype=ab.dtype)

    for offset in range(1, num_low_diags + 1):
        ab_subdiags[offset - 1, 0 : n_cols - offset] = ab[
            main_diag_idx - offset, offset:None
        ]

    # the subdiagonal part is then concatenated to the original array and the result is
    # returned
    l_and_u = (num_low_diags, num_low_diags)
    return l_and_u, np.row_stack((ab, ab_subdiags))


def conv_to_lu_banded_storage(
    a: np.ndarray | spmatrix,
    l_and_u: tuple[int, int],
) -> np.ndarray:
    """Converts a (sparse) square banded matrix A to its banded storage required for
    LU decomposition in LAPACK-routines like the function ``lu_banded`` or SciPy's
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
    For LAPACK's LU decomposition, the matrix `a` is stored in `ab` using the matrix
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


### LAPACK-Wrappers for banded LU decomposition ###


def lu_banded(
    l_and_u: LAndUBandCounts,
    ab: ArrayLike,
    *,
    check_finite: bool = True,
) -> BandedLUFactorization:
    """
    Computes the LU decomposition of a banded matrix ``A`` using LAPACK-routines.
    This function is a wrapper of the LAPACK-routine ``gbtrf`` which computes the LU
    decomposition of a banded matrix ``A`` in-place. It wraps the routine in an
    analogous way to SciPy's ``scipy.linalg.cholesky_banded``.

    Parameters
    ----------
    l_and_u : (int, int)
        The number of "non-zero" sub- (first) and superdiagonals (second element) aside
        the main diagonal which does not need to be considered here. "Non-zero" can be
        a bit misleading in this context. These numbers should count up to the diagonal
        after which all following diagonals are all zero. Zero-diagonals that come
        before still need to be included.
        Neither of both may exceed ``num_rows``.
        Wrong specification of this can lead to non-zero-diagonals being ignored or
        zero-diagonals being included which corrupts the results or reduces the
        performance.
    ab : array_like of shape (l_and_u[0] + 1 + l_and_u[1], n)
        A 2D-Array resembling the matrix ``A`` in banded storage format (see Notes).
    check_finite : bool, default=True
        Whether to check that the input matrix contains only finite numbers. Disabling
        may give a performance gain, but may result in problems (crashes,
        non-termination) if the inputs do contain infinities or NaNs.

    Returns
    -------
    lub_factorization : BandedLUFactorization
        A dataclass containing the LU factorization of the matrix ``A`` as follows:
            ``lub``: The LU decomposition of ``A`` in banded storage format (see Notes).
            ``ipiv``: The pivoting indices.
            ``l_and_u``: The number of sub- and superdiagonals of the matrix ``A`` that
                are non-zero.
            ``singular``: A boolean indicating whether the matrix is singular.

    Notes
    -----
    For LAPACK's banded LU decomposition, the matrix ``a`` is stored in ``ab`` using the
    matrix diagonal ordered form:

        ```python
        ab[u + i - j, j] == a[i,j] # see below for u
        ```

    An example of ``ab`` (shape of a is ``(7,7)``, ``u``=3 superdiagonals, ``l``=2
    subdiagonals) looks like:

        ```python
             *    *    *   a03  a14  a25  a36
             *    *   a02  a13  a24  a35  a46
             *   a01  a12  a23  a34  a45  a56   # ^ superdiagonals
            a00  a11  a22  a33  a44  a55  a66   # main diagonal
            a10  a21  a32  a43  a54  a65   *    # v subdiagonals
            a20  a31  a42  a53  a64   *    *
        ```

    where all entries marked with `*` are zero elements although they will be set to
    arbitrary values by this function.

    Internally LAPACK relies on an expanded version of this format to perform inplace
    operations that adds another ``l`` superdiagonals to the matrix in order to
    overwrite them for the purpose of pivoting. The output is thus an expanded version
    of the LU decomposition of ``A`` in the same format where the main diagonal of
    ``L`` is implicitly taken to be a vector of ones. The output can directly be used
    for the LAPACK-routine ``gbtrs`` to solve linear systems of equations based on this
    decomposition.

    """

    # the (optional) finite check and Array-conversion are performed
    if check_finite:
        ab = np.asarray_chkfinite(ab)
    else:
        ab = np.asarray(ab)

    # then, the number of lower and upper subdiagonals needs to be checked for being
    # consistent with the shape of ``ab``
    num_low_diags, num_upp_diags = l_and_u
    if num_low_diags + num_upp_diags + 1 != ab.shape[0]:
        raise ValueError(
            f"\nInvalid values for the number of lower and upper "
            f"diagonals: l+u+1 ({num_low_diags + num_upp_diags + 1}) does not equal "
            f"ab.shape[0] ({ab.shape[0]})."
        )

    # now, the LAPACK-routines can be called
    # to make ``ab`` compatible with the shape the LAPACK expects in this case, it
    # needs to be re-written into a larger Array that has zeros elsewhere
    # FIXME: for tridiagonal matrices, the SciPy wrapper for ``gttrf`` should be used
    lapack_routine = "gbtrf"
    (gbtrf,) = lapack.get_lapack_funcs((lapack_routine,), (ab,))
    lpkc_ab = np.row_stack(
        (
            np.zeros((num_low_diags, ab.shape[1]), dtype=ab.dtype),
            ab,
        )
    )
    lub, ipiv, info = gbtrf(
        ab=lpkc_ab, kl=num_low_diags, ku=num_upp_diags, overwrite_ab=True
    )

    # then, the results needs to be validated and returned
    # Case 1: the factorisation could be completed, which does not imply that the
    # solution can be used for solving a linear system
    if info >= 0:
        return BandedLUFactorization(
            lub=lub,
            ipiv=ipiv,
            l_and_u=l_and_u,
            singular=info > 0,
        )

    # Case 2: the factorisation was not completed due to invalid input
    raise ValueError(f"\nIllegal value in {-info}-th argument of internal gbtrf.")


def lu_solve_banded(
    lub_factorization: BandedLUFactorization,
    b: ArrayLike,
    *,
    overwrite_b: bool = False,
    check_finite: bool = True,
) -> np.ndarray:
    """
    Solves a linear system of equations ``Ax=b`` with a banded matrix ``A`` using its
    precomputed LU decomposition.
    This function wraps the LAPACK-routine ``gbtrs`` in an analogous way to SciPy's
    ``scipy.linalg.cho_solve_banded``.

    Parameters
    ----------
    lub_factorization : BandedLUFactorization
        The LU decomposition of the matrix ``A`` in banded storage format as returned by
        the function :func:`lu_banded`.
    b : ndarray of shape (n,)
        A 1D-Array containing the right-hand side of the linear system of equations.
    overwrite_b : bool, default=False
        If ``True``, the contents of ``b`` can be overwritten by the routine. Otherwise,
        a copy of ``b`` is created and overwritten.
    check_finite : bool, default=True
        Whether to check that the input contains only finite numbers. Disabling may give
        a performance gain, but may result in problems (crashes, non-termination) if the
        inputs do contain infinities or NaNs.

    Returns
    -------
    x : ndarray of shape (n,)
        The solution to the system ``A x = b``.

    Raises
    ------
    LinAlgError
        If the system to solve is singular.

    """

    # if the matrix is singular, the solution cannot be computed
    if lub_factorization.singular:
        raise np.linalg.LinAlgError("\nSystem is singular.")

    # the (optional) finite check and Array-conversion are performed
    if check_finite:
        lub_factorization.lub = np.asarray_chkfinite(lub_factorization.lub)
        lub_factorization.ipiv = np.asarray_chkfinite(lub_factorization.ipiv)
        b_inter = np.asarray_chkfinite(b)
    else:
        lub_factorization.lub = np.asarray(lub_factorization.lub)
        lub_factorization.ipiv = np.asarray(lub_factorization.ipiv)
        b_inter = np.asarray(b)

    overwrite_b = overwrite_b or _datacopied(b_inter, b)

    # then, the shapes of the LU decomposition and ``b`` need to be validated against
    # each other
    if lub_factorization.n_cols != b_inter.shape[0]:
        raise ValueError(
            f"\nShapes of lub ({lub_factorization.n_cols}) and b "
            f"({b_inter.shape[0]}) are not compatible."
        )

    # now, the LAPACK-routine is called
    (gbtrs,) = lapack.get_lapack_funcs(("gbtrs",), (lub_factorization.lub, b))
    x, info = gbtrs(
        ab=lub_factorization.lub,
        kl=lub_factorization.l_and_u[0],
        ku=lub_factorization.l_and_u[1],
        b=b,
        ipiv=lub_factorization.ipiv,
        overwrite_b=overwrite_b,
    )

    # then, the results needs to be validated and returned
    # Case 1: the solution could be computed truly successfully, i.e., without any
    # NaN-values
    if info == 0 and not np.any(np.isnan(x)):
        return x

    # Case 2: the solution was computed, but there were NaN-values in it
    elif info == 0:
        raise np.linalg.LinAlgError("\nMatrix is singular.")

    # Case 3: the solution could not be computed due to invalid input
    elif info < 0:
        raise ValueError(f"\nIllegal value in {-info}-th argument of internal gbtrs.")

    # Case 4: unexpected error
    raise AssertionError(
        f"\nThe internal gbtrs returned info > 0 ({info}) which should not happen."
    )


def slogdet_lu_banded(
    lub_factorization: BandedLUFactorization,
) -> tuple[float, float]:
    """
    Computes the logarithm of the absolute value and the sign of the determinant of a
    banded matrix A using its LU decomposition. This is way more efficient than
    computing the determinant directly because the LU decompositions main diagonals
    already encode the determinant as the product of the diagonal entries of the
    factors.

    Parameters
    ----------
    lub_factorization : BandedLUFactorization
        The LU decomposition of the matrix ``A`` in banded storage format as returned by
        the function :func:`lu_banded`.

    Returns
    -------
    sign : float
        A number representing the sign of the determinant.
    logabsdet : float
        The natural log of the absolute value of the determinant.
        If the determinant is zero, then `sign` will be 0 and `logabsdet` will be
        -Inf. In all cases, the determinant is equal to ``sign * np.exp(logabsdet)``.

    Raises
    ------
    OverflowError
        If any of the diagonal entries of the LU decomposition leads to an overflow in
        the natural logarithm.

    """

    # first, the number of actual row exchanges needs to be counted
    unchanged_row_idxs = np.arange(
        start=0,
        stop=lub_factorization.n_rows,
        step=1,
        dtype=lub_factorization.ipiv.dtype,
    )
    num_row_exchanges = np.count_nonzero(lub_factorization.ipiv - unchanged_row_idxs)

    # the sign-prefactor of the determinant is either +1 or -1 depending on whether the
    # number of row exchanges is even or odd
    sign = -1.0 if num_row_exchanges % 2 == 1 else 1.0

    # since the determinant (without sign prefactor) is just the product of the diagonal
    # product of L and the diagonal product of U, the calculation simplifies. As the
    # main diagonal of L is a vector of ones, only the diagonal product of U is required
    main_diag = lub_factorization.lub[lub_factorization.main_diag_row_idx, ::]
    u_diaprod_sign = np.prod(np.sign(main_diag))
    with np.errstate(divide="ignore", over="ignore"):
        logabsdet = np.sum(np.log(np.abs(main_diag)))

    # logarithms of zero are already properly handled, so there is not reason to worry
    # about, since they are -inf which will result in a zero determinant in exp();
    # overflow however needs to lead to a raise and in this case the log(det) is either
    # +inf in case of overflow only or NaN in case of the simultaneous occurrence of
    # zero and overflow
    if np.isnan(logabsdet) or np.isposinf(logabsdet):
        raise OverflowError(
            "\nFloating point overflow in natural logarithm. At least 1 main diagonal "
            "entry results in overflow, thereby corrupting the determinant."
        )

    # finally, the absolute value of the natural logarithm of the determinant is
    # returned together with its sign
    if np.isneginf(logabsdet):
        return 0.0, logabsdet
    elif float(u_diaprod_sign) > 0.0:
        return sign, logabsdet

    return -sign, logabsdet


### SciPy-Wrappers for banded Cholesky-decomposition ###


def cholesky_banded(
    ab: np.ndarray,
    overwrite_ab: bool = False,
    lower: bool = False,
    check_finite: bool = True,
) -> BandedCholeskyFactorization:
    """
    A drop-in replacement for SciPy's ``cholesky_banded`` that stores the factorization
    in a dataclass.

    Please refer to the SciPy documentation for further information that is not
    mentioned here.

    Returns
    -------
    chob_factorization : BandedCholeskyFactorization
        A dataclass containing the Cholesky-factorization of the matrix ``A`` as
        follows:
            ``lb``: The Cholesky-decomposition of ``A`` in banded storage format.
            ``lower``: A boolean indicating whether the Cholesky-decomposition is in
                lower triangular form (``True``) or in upper triangular form
                (``False``).
    """

    return BandedCholeskyFactorization(
        lb=scipy_cholesky_banded(**locals()),
        lower=lower,
    )


def cho_solve_banded(
    chob_factorization: BandedCholeskyFactorization,
    b: np.ndarray,
    overwrite_b: bool = False,
    check_finite: bool = True,
) -> np.ndarray:
    """
    A drop-in replacement for SciPy's ``cho_solve_banded`` that relies on the
    factorization being stored in a dataclass.

    Please refer to the SciPy documentation for further information that is not
    mentioned here.

    Parameters
    ----------
    chob_factorization : BandedCholeskyFactorization
        The Cholesky-factorization of the matrix ``A`` in banded storage format as
        returned by the function :func:`cholesky_banded`.

    """

    return scipy_cho_solve_banded(
        cb_and_lower=(chob_factorization.lb, chob_factorization.lower),
        b=b,
        overwrite_b=overwrite_b,
        check_finite=check_finite,
    )


def slodget_cho_banded(
    chob_factorization: BandedCholeskyFactorization,
) -> tuple[float, float]:
    """Computes the logarithm of the absolute value of the determinant of a banded
    hermitian matrix `A` using its Cholesky-decomposition. This is way more efficient
    than computing the determinant directly because the Cholesky factors' main
    diagonals already encode the determinant as the product of the diagonal entries.

    Parameters
    ----------
    chob_factorization : BandedCholeskyFactorization
        The Cholesky-factorization of the matrix `A` in banded storage format as
        returned by the function :func:`cholesky_banded`.

    Returns
    -------
    sign : float
        A number representing the sign of the determinant. It is always +1 since
        the matrix under consideration is positive definite.
    logabsdet : float
        The natural log of the absolute value of the determinant. It cannot be zero
        since the matrix under consideration is positive definite.

    Raises
    ------
    OverflowError
        If any of the diagonal entries of the Cholesky-decomposition leads to an
        overflow in the natural logarithm.

    """

    # the sign-prefactor of the determinant is always +1 since the matrix is positive
    # definite, so only the diagonal product of the Cholesky-decomposition is required
    main_diag = chob_factorization.lb[chob_factorization.main_diag_row_idx, ::]
    with np.errstate(divide="ignore", over="ignore"):
        logabsdet = 2.0 * np.sum(np.log(main_diag))

    # logarithms of zero are already properly handled, so there is not reason to worry
    # about, since they are -inf which will result in a zero determinant in exp();
    # overflow however needs to lead to a raise and in this case the log(det) is either
    # +inf in case of overflow only or NaN in case of the simultaneous occurrence of
    # zero and overflow
    if np.isnan(logabsdet) or np.isposinf(logabsdet):
        raise OverflowError(
            "\nFloating point overflow in natural logarithm. At least 1 main diagonal "
            "entry results in overflow, thereby corrupting the determinant."
        )

    return 1.0, logabsdet
