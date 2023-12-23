from numbers import Integral
from typing import Optional, Union
from warnings import warn

import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import eigvals_banded, lapack
from scipy.sparse import eye as speye
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
    a: Union[np.ndarray, spmatrix],
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
    ab = np.zeros(shape=(num_low_diags + 1 + num_upp_diags, num_cols))

    # the superdiagonals and the main diagonal
    for offset in range(num_upp_diags, -1, -1):
        ab[main_diag_idx - offset, offset::] = diag_method(offset)

    # the subdiagonals
    for offset in range(-1, -num_low_diags - 1, -1):
        ab[main_diag_idx - offset, 0:offset] = diag_method(offset)

    return ab


def lu_banded(
    l_and_u: tuple[int, int],
    ab: ArrayLike,
    *,
    overwrite_ab: bool = False,
    check_finite: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Computes the LU-decomposition of a banded matrix A using LAPACK-routines.

    This function is a wrapper of the LAPACK-routine ``gbtrf`` which computes the LU-
    decomposition of a banded matrix `A` in-place. It wraps the routine in an analogous
    way to SciPy's ``cholesky_banded``.

    Parameters
    ----------
    l_and_u : tuple[int, int]
        The number of "non-zero" sub- (first) and superdiagonals (second element) aside
        the main diagonal which does not need to be considered here. "Non-zero" can be
        a bit misleading in this context. These numbers should count up to the diagonal
        after which all following diagonals are zero. Zero-diagonals that come before
        still need to be included.
        Neither of both may exceed `num_rows`.
        Wrong specification of this can lead to non-zero-diagonals being ignored or
        zero-diagonals being included which corrupts the results or reduces the
        performance.
    ab : np.ndarray of shape (l_and_u[0] + 1 + l_and_u[1], n)
        A NumPy-2D-Array resembling the matrix `A` in banded storage format
        (see Notes).

    overwrite_ab : bool, default=False
        If ``True``, the contents of `ab` can be overwritten by the routine. Otherwise,
        a copy of `ab` is created and overwritten.

    check_finite : bool, default=True
        Whether to check that the input matrix contains only finite numbers. Disabling
        may give a performance gain, but may result in problems (crashes,
        non-termination) if the inputs do contain infinities or NaNs.

    Returns
    -------
    lu : np.ndarray of shape (l_and_u[0] + 1 + 2 * l_and_u[1], n)
        A NumPy-2D-Array resembling the LU-decomposition of `A` in banded storage
        format (see Notes).
    ipiv : np.ndarray of shape (n,)
        A NumPy-1D-Array containing the pivoting indices. It's `i`-th entry resembles
        gives the row that was used for pivoting the `i`-th row of `A`.

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
    operations that adds another `l` superdiagonals to the matrix in order to overwrite
    them for the purpose of pivoting. The output is thus an expanded version
    of the LU-decomposition of `A` in the same format where the main diagonal of
    `L` is implicitly taken to be a vector of ones. The output can directly be used
    for the LAPACK-routine ``gbtrs`` to solve linear systems of equations based on this
    decomposition.

    """
    # the (optional) finite check and Array-conversion are performed
    if check_finite:
        inter_ab = np.asarray_chkfinite(ab)
    else:
        inter_ab = np.asarray(ab)

    # then, the number of lower and upper subdiagonals needs to be checked for being
    # consistent with the shape of ``ab``
    num_low_diags, num_upp_diags = l_and_u
    if num_low_diags + num_upp_diags + 1 != inter_ab.shape[0]:
        raise ValueError(
            f"\nInvalid values for the number of lower and upper "
            f"diagonals: l+u+1 ({num_low_diags + num_upp_diags + 1}) does not equal "
            f"ab.shape[0] ({inter_ab.shape[0]})."
        )
    # else nothing

    # now, the LAPACK-routines can be called
    # to make ``ab`` compatible with the shape the LAPACK expects in this case, it
    # needs to be re-written into a larger Array that has zeros elsewhere
    # FIXME: for tridiagonal matrices, the SciPy wrapper for ``gttrf`` should be used
    lapack_routine = "gbtrf"
    (gbtrf,) = lapack.get_lapack_funcs((lapack_routine,), (inter_ab,))
    lpkc_ab = np.zeros(
        shape=(2 * num_low_diags + num_upp_diags + 1, inter_ab.shape[1]),
        dtype=gbtrf.dtype,
    )
    lpkc_ab[num_low_diags::, ::] = inter_ab
    lu, ipiv, info = gbtrf(
        ab=lpkc_ab, kl=num_low_diags, ku=num_upp_diags, overwrite_ab=overwrite_ab
    )

    # then, the results needs to be validated and returned
    # Case 1: the factorisation could be completed, which does not imply that the
    # solution can be used for solving a linear system
    if info >= 0:
        if info > 0:
            warn(
                f"\nThe ({info-1}, {info-1})-th entry of the factor U is exactly 0, "
                f"which makes it singular.\n"
                f"Linear systems cannot be solved with this factor."
            )
        # else nothing

        return lu, ipiv

    # Case 2: the factorisation was not completed due to invalid input
    else:
        raise ValueError(f"\nIllegal value in {-info}-th argument of internal gbtrf.")


def slogdet_lu_banded(
    decomposition: tuple[tuple[int, int], np.ndarray, np.ndarray],
) -> tuple[float, float]:
    """Computes the logarithm of the absolute value of the determinant of a banded
    matrix A using its LU-decomposition. This is way more efficient than computing the
    determinant directly because the LU-decompositions main diagonals already encode
    the determinant as the product of the diagonal entries of the factors.

    Parameters
    ----------
    (l_and_u, lub, ipiv) : tuple, (tuple[int, int], np.ndarray, np.ndarray)
        `l_and_u` is a tuple of two integers specifying the number of sub- and
        superdiagonals of the matrix `A` that are non-zero.
        `lub` is a NumPy-2D-Array resembling the LU-decomposition of `A` in banded
        storage format as returned by ``lu_banded``.
        `ipiv` is a NumPy-1D-Array containing the pivoting indices as returned by
        ``lu_banded``.

    Returns
    -------
    sign : float
        A number representing the sign of the determinant.
    logabsdet : float
        The natural log of the absolute value of the determinant.

    If the determinant is zero, then `sign` will be 0 and `logabsdet` will be
    -Inf. In all cases, the determinant is equal to ``sign * np.exp(logabsdet)``.

    """
    # first, the number of lower and upper diagonals is extracted
    l_and_u, lub, ipiv = decomposition
    num_low_diags, num_upp_diags = l_and_u
    num_rows = lub.shape[-1]

    # then, the number of actual row exchanges needs to be counted
    unchanged_row_idxs = np.arange(start=0, stop=num_rows, step=1, dtype=ipiv.dtype)
    num_row_exchanges = np.count_nonzero(ipiv - unchanged_row_idxs)

    # the sign-prefactor of the determinant is either +1 or -1 depending on whether the
    # number of row exchanges is even or odd
    if num_row_exchanges % 2 == 1:
        sign = -1.0
    else:
        sign = 1.0

    # since the determinant (without sign prefactor) is just the product of the diagonal
    # product of L and the diagonal product of U, the calculation simplifies. As the
    # main diagonal of L is a vector of ones, only the diagonal product of U is required
    main_diag_idx = num_low_diags + num_upp_diags
    u_diaprod_sign = np.prod(np.sign(lub[main_diag_idx, ::]))
    with np.errstate(divide="ignore", over="ignore"):
        logabsdet = np.sum(np.log(np.abs(lub[main_diag_idx, ::])))

    # logarithms of zero are already properly handled, so there is not reason to worry
    # about, since they are -inf which will result in a zero determinant in exp()
    # overflow however needs to lead to a raise and in this case the log(det) is either
    # +inf in case of overflow only or NaN in case of the simultaneous occurrence of
    # zero and overflow
    if np.isnan(logabsdet) or np.isposinf(logabsdet):
        raise ValueError(
            "\nFloating point overflow in natural logarithm. At least 1 main diagonal "
            "entry results in overflow, thereby corrupting the determinant."
        )
    # else nothing

    # finally, the absolute value of the natural logarithm of the determinant is
    # returned together with its sign
    if np.isneginf(logabsdet):
        sign = 0.0
    elif float(u_diaprod_sign) > 0.0:
        pass
    else:
        sign *= -1.0

    return sign, logabsdet


def lu_solve_banded(
    decomposition: tuple[np.ndarray, np.ndarray, tuple[int, int]],
    b: ArrayLike,
    *,
    overwrite_b: bool = False,
    check_finite: bool = True,
) -> np.ndarray:
    """Solves a linear system of equations ``Ax=b`` with a banded matrix `A` using its
    precomputed LU-decomposition.
    This function wraps the LAPACK-routine ``gbtrs`` in an analogous way to SciPy's
    ``cho_solve_banded``.

    Parameters
    ----------
    (l_and_u, lub, ipiv) : tuple, (np.ndarray, np.ndarray, tuple[int, int])
        `lub` is a NumPy-2D-Array resembling the LU-decomposition of `A` in banded
            storage format as returned by ``lu_banded``.
        `ipiv` is a NumPy-1D-Array containing the pivoting indices as returned by
            ``lu_banded``.
        `l_and_u` is a tuple of two integers specifying the number of sub- and
            superdiagonals of the matrix `A` that are non-zero.
    b : np.ndarray of shape (n,)
        A 1D-Array containing the right-hand side of the linear system of equations.
    overwrite_b : bool, default=False
        If ``True``, the contents of `b` can be overwritten by the routine. Otherwise,
        a copy of `b` is created and overwritten.
    check_finite : bool, default=True
        Whether to check that the input matrix contains only finite numbers. Disabling
        may give a performance gain, but may result in problems (crashes,
        non-termination) if the inputs do contain infinities or NaNs.

    Returns
    -------
    x : np.ndarray of shape (n,)
        The solution to the system A x = b

    """
    # the (optional) finite check and Array-conversion are performed
    lub, ipiv, l_and_u = decomposition
    if check_finite:
        inter_lub = np.asarray_chkfinite(lub)
        inter_ipiv = np.asarray_chkfinite(ipiv)
        inter_b = np.asarray_chkfinite(b)
    else:
        inter_lub = np.asarray(lub)
        inter_ipiv = np.asarray(ipiv)
        inter_b = np.asarray(b)

    # then, the shapes of the LU-decomposition and ``b`` need to be validated against
    # each other
    if inter_lub.shape[-1] != inter_b.shape[0]:
        raise ValueError(
            f"\nShapes of lub ({inter_lub.shape[-1]}) and b ({inter_b.shape[0]}) are "
            f"not compatible."
        )
    # else nothing

    # now, the LAPACK-routine is called
    num_low_diags, num_upp_diags = l_and_u
    (gbtrs,) = lapack.get_lapack_funcs(("gbtrs",), (inter_lub, inter_b))
    x, info = gbtrs(
        ab=inter_lub,
        kl=num_low_diags,
        ku=num_upp_diags,
        b=inter_b,
        ipiv=inter_ipiv,
        overwrite_b=overwrite_b,
    )

    # then, the results needs to be validated and returned
    # Case 1: the solution could be computed successfully
    if info == 0:
        return x

    # Case 2: the solution could not be computed due to invalid input
    elif info < 0:
        raise ValueError(f"\nIllegal value in {-info}-th argument of internal gbtrs.")

    # Case 3: unexpected error
    else:
        raise AssertionError(
            f"\nThe internal gbtrs returned info > 0 ({info}) which should not happen."
        )


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


def _find_largest_symm_sparse_banded_spd_eigval(
    ab: np.ndarray, check_finite: bool = True
) -> float:
    """Finds the largest eigenvalue of a symmetric sparse banded matrix `A` using
    SciPy's ``sparse.linalg.eigsh``.

    Notes
    -----
    This function is intended for matrices that are known to be at least positive
    semi-definite from a mathematical point of view (all eigenvalues >= 0). However, due
    to numerical inaccuracies, the smallest eigenvalue may be negative. Such a
    restriction is not critical in this context since the largest eigenvalue is
    typically positive.
    From a performance perspective, this function relies on LAPACK's banded eigensolver
    and it thus highly efficient already.

    """

    return eigvals_banded(
        a_band=ab,
        lower=False,
        select="i",
        select_range=(ab.shape[1] - 1, ab.shape[1] - 1),
        check_finite=check_finite,
    )[0]


def _find_smallest_symm_sparse_banded_spd_eigval(
    ab: np.ndarray, check_finite: bool = True
) -> float:
    """Finds the smallest eigenvalue of a symmetric sparse banded matrix `A` using
    SciPy's ``sparse.linalg.eigsh``.

    Notes
    -----
    This function is intended for matrices that are known to be at least positive
    semi-definite from a mathematical point of view (all eigenvalues >= 0). However, due
    to numerical inaccuracies, the smallest eigenvalue may be negative.
    From a performance perspective, this function relies on LAPACK's banded eigensolver
    and it thus highly efficient already.

    """
    return eigvals_banded(
        a_band=ab,
        lower=False,
        select="i",
        select_range=(0, 0),
        check_finite=check_finite,
    )[0]


def conv_symm_sparse_banded_sposdef_to_posdef(
    a: spmatrix,
    *,
    l_and_u: tuple[int, int],
    rcond: Optional[float] = None,
    check_finite: bool = True,
) -> spmatrix:
    """Converts a symmetric sparse banded matrix `A` to a positive definite matrix
    `B` by adding a small multiple of the identity matrix to `A` (see Notes).

    Parameters
    ----------
    a : spmatrix of shape (n, n)
        A square symmetric sparse banded matrix.

    l_and_u : tuple[int, int]
        The number of "non-zero" sub- (first) and superdiagonals (second element) aside
        the main diagonal which does not need to be considered here. "Non-zero" can be
        a bit misleading in this context. These numbers should count up to the diagonal
        after which all following diagonals are zero. Zero-diagonals that come before
        still need to be included.
        Wrong specification of this can lead to non-zero-diagonals being ignored or
        zero-diagonals being included which corrupts the results or reduces the
        performance.
        Both its entries must coincide.

    rcond : float, default=None
        The relative condition number of the positive definite matrix `B`.
        If ``None``, the default value of ``scipy.linalg.pinvh`` is used which is
        ``eps * n`` where ``eps`` is the machine precision of the datatype of `a`.

    Returns
    -------
    b : spmatrix of shape (n, n)
        A positive definite matrix which is identical to ``a`` except for main diagonal.

    Raises
    ------
    ValueError
        If `a` is not square or symmetric.

    Notes
    -----
    This function performs no checks on `a`.
    It is intended for matrices that are known to be at least positive semi-definite
    from a mathematical point of view (all eigenvalues >= 0). However, due to numerical
    inaccuracies, the smallest eigenvalue may be negative.
    For making `A` positive definite, a small multiple of the identity matrix is added
    to it as ``B = A + delta * I`` where `delta` is chosen to be the smallest positive
    number such that the smallest eigenvalue of ``A + delta * I`` is numerically
    positive when compared to the largest eigenvalue of ``A + delta * I``. In other
    words, given the smallest and largest eigenvalue of `A` `lam_min` and `lam_max`,
    respectively, `delta` is is chosen such that
    ``(lam_min + delta) / (lam_max + delta) >= rcond`` because ``lam_min + delta`` and
    ``lam_max + delta`` are the smallest and largest eigenvalue of the resulting `B`.
    Internally ``1.1 * rcond`` is used as the smallest ratio to account for numerical
    inaccuracies in the conducted and potential future computations of eigenvalues.

    """

    # first, the rcond-parameter is determined
    if rcond is None:
        inter_rcond = np.finfo(a.dtype).eps * a.shape[0]  # type: ignore
    else:
        inter_rcond = rcond

    # then, the smallest and largest eigenvalue are computed
    ab = conv_to_lu_banded_storage(a=a, l_and_u=l_and_u)[0 : l_and_u[1] + 1, ::]
    max_eigenvalue = _find_largest_symm_sparse_banded_spd_eigval(
        ab=ab, check_finite=check_finite
    )
    min_eigenvalue = _find_smallest_symm_sparse_banded_spd_eigval(
        ab=ab, check_finite=check_finite
    )

    # if the ratio is fine already, the matrix is returned
    if (min_eigenvalue / max_eigenvalue) >= inter_rcond:
        return a.copy()  # type: ignore
    # else nothing

    # otherwise, the smallest multiple of the identity matrix is computed that makes
    # the ratio fine and the resulting matrix is returned
    inter_rcond *= 1.1
    delta = (min_eigenvalue - inter_rcond * max_eigenvalue) / (inter_rcond - 1.0)

    return a + delta * speye(
        m=a.shape[0], dtype=a.dtype, format=a.format  # type: ignore
    )
