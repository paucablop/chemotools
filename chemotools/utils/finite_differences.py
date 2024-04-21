from math import comb
from numbers import Integral

import numpy as np
from sklearn.utils import check_scalar


def calc_forward_diff_kernel(
    *,
    differences: int,
) -> np.ndarray:
    """
    Computes the kernel for forward finite differences which can be applied to a
    series by means of a convolution, e.g.,

    ```python
        kernel = calc_forward_fin_diff_kernel(differences=2)
        differences = np.convolve(series, np.flip(kernel), mode="valid")
        # NOTE: NumPy flips the kernel internally due to the definition of convolution
    ```

    Parameters
    ----------
    differences : int
        The order of the differences starting from 0 for the original curve, 1 for the
        first order, 2 for the second order, ..., and ``m`` for the ``m``-th order
        differences.
        Values below 1 are not allowed.

    Returns
    -------
    fin_diff_kernel : ndarray of shape (differences + 1,)
        A NumPy-1D-vector resembling the kernel from the code example above. To avoid
        loss of precision, the data type is ``np.int64``.

    Raises
    ------
    ValueError
        If the difference order is below 1.

    """
    # the input is validated
    check_scalar(
        differences,
        name="differences",
        target_type=Integral,
        min_val=1,
        include_boundaries="left",
    )

    # afterwards, the kernel is computed using the binomial coefficients
    return np.array(
        [
            ((-1) ** iter_i) * comb(differences, iter_i)
            for iter_i in range(differences, -1, -1)
        ],
        dtype=np.int64,
    )


def _gen_squ_fw_fin_diff_mat_cho_banded_transp_first(
    *,
    n_data: int,
    differences: int,
) -> np.ndarray:
    """
    Generates the squared forward finite differences matrix ``D.T @ D`` from the
    forward finite difference matrix ``D`` of order ``differences``. It will be cast to
    to the upper banded storage format used for LAPACK's banded Cholesky decomposition.

    All unused elements in the banded matrix are set to zero.

    """

    # the problems has to be split into a leading, a central, and a trailing part
    # first, the leading part is computed because it might be that this is already
    # enough
    # first, the kernel for the forward differences is computed and the bandwidth is
    # determined
    kernel = calc_forward_diff_kernel(differences=differences)
    n_bands = 1 + 2 * differences
    lead_n_rows = min(kernel.size, n_data - kernel.size + 1)
    lead_n_cols = kernel.size + lead_n_rows - 1

    # the leading matrix is computed as a dense matrix
    dtd = np.zeros(shape=(lead_n_rows, lead_n_cols), dtype=np.int64)
    for row_idx in range(0, lead_n_rows):
        dtd[row_idx, row_idx : row_idx + kernel.size] = kernel

    # its squared form is computed
    dtd = dtd.T @ dtd

    # now, the leading matrix is converted to a banded matrix
    dtd_banded = np.zeros(shape=(differences + 1, lead_n_cols), dtype=np.int64)
    for diag_idx in range(0, differences + 1):
        offset = differences - diag_idx
        dtd_banded[diag_idx, offset:None] = np.diag(dtd, k=offset)

    # if the number of data points is less than the kernel size minus one, the
    # leading matrix is already the final matrix
    if n_data <= n_bands:
        return dtd_banded

    # otherwise, a central part has to be inserted
    # this turns out to be just a column-wise repetition of the kernel computed with
    # double the difference order, so this matrix can simple be inserted into the
    # computed leading D.T @ D matrix
    # NOTE: the doubled kernel is the most central column of the banded D.T @ D already
    #       computed
    central_n_cols = n_data - dtd_banded.shape[1]
    kernel_double = dtd_banded[::, kernel.size - 1].reshape((-1, 1))
    return np.concatenate(
        (
            dtd_banded[::, 0 : kernel.size],
            np.tile(kernel_double, (1, central_n_cols)),
            dtd_banded[::, kernel.size :],
        ),
        axis=1,
    )


def _gen_squ_fw_fin_diff_mat_cho_banded_orig_first(
    *,
    n_data: int,
    differences: int,
) -> np.ndarray:
    """
    Generates the squared forward finite differences matrix ``D @ D.T`` from the
    forward finite difference matrix ``D`` of order ``differences``. It will be cast to
    to the upper banded storage format used for LAPACK's banded Cholesky decomposition.

    All unused elements in the banded matrix are set to zero.

    """

    # this case is simpler than the transposed case because the matrix is just a
    # Toeplitz matrix with the kernel of double the difference order on the diagonal
    kernel_double = calc_forward_diff_kernel(differences=2 * differences)[
        differences:None
    ]
    # for an odd difference order, the sign of the kernel has to be flipped
    if differences % 2 == 1:
        kernel_double = np.negative(kernel_double)

    n_rows = n_data - kernel_double.size + 1
    n_upp_plus_central_bands = min(n_rows, 1 + differences)

    # the matrix is computed as a dense and simple filled by means of a loop
    ddt_banded = np.zeros(shape=(n_upp_plus_central_bands, n_rows), dtype=np.int64)
    main_diag_idx = min(differences, n_upp_plus_central_bands - 1)
    for offset in range(0, n_upp_plus_central_bands):
        ddt_banded[main_diag_idx - offset, offset:None] = kernel_double[offset]

    return ddt_banded


def gen_squ_fw_fin_diff_mat_cho_banded(
    *,
    n_data: int,
    differences: int,
    orig_first: bool,
) -> np.ndarray:
    """
    Generates the squared forward finite differences matrix ``D @ D.T`` or ``D.T @ D``
    from the forward finite difference matrix ``D`` of order ``differences``. It will be
    cast to to the upper banded storage format used for LAPACK's banded Cholesky
    decomposition.

    All unused elements in the banded matrix are set to zero.

    Parameters
    ----------
    n_data : int
        The number of data points in the series to which the forward finite differences
        are applied.
    differences : int
        The order of the differences starting from 0 for the original curve, 1 for the
        first order, 2 for the second order, ..., and ``m`` for the ``m``-th order
        differences.
        Values below 1 are not allowed.
    orig_first : bool
        If ``True``, the squared forward finite differences matrix ``D @ D.T`` is
        computed. Otherwise, the squared forward finite differences matrix ``D.T @ D``
        is computed.

    Returns
    -------
    squ_fw_fin_diff_mat_cho_banded : ndarray of shape (n_bands, n_data - differences + 1) or (n_bands, n_data)
        The squared forward finite differences matrix in the upper banded storage format
        used for LAPACK's banded Cholesky decomposition (see Notes for details).
        When ``orig_first`` is ``True``, the matrix has at maximum ``differences + 1``
        bands (rows) and ``n_data - differences + 1`` columns.
        Otherwise, the matrix has at maximum ``differences + 1`` bands (rows) and
        ``n_data`` columns.

    Raises
    ------
    ValueError
        If ``n_data`` is below ``differences + 1``, i.e., the kernel does not fit into
        the data at least once.
    ValueError
        If the difference order is below 1.

    Notes
    -----
    The squared forward finite differences matrix is stored in the upper banded storage
    format used for LAPACK's banded Cholesky decomposition.
    This upper diagonal ordered form is given by the following ordering

    ```python
    ab[u + i - j, j] == a[i,j]
    ```

    e.g., for a symmetric matrix of shape (6, 6) with in total 3 superdiagonals,
    3 subdiagonals, and the main diagonal, the ordering is as follows:

    ```python
    *   *   *   a03 a14 a25
    *   *   a02 a13 a24 a35
    *   a01 a12 a23 a34 a45 # ^ superdiagonals
    a00 a11 a22 a33 a44 a55 # main diagonal
    ```

    where each `*` denotes a zero element.

    Written out, this would give the following matrix:

    ```python
    a00 a01 a02 a03 0   0
    a01 a11 a12 a13 a14 0
    a02 a12 a22 a23 a24 a25
    a03 a13 a23 a33 a34 a35
    0   a14 a24 a34 a44 a45
    0   0   a25 a35 a45 a55
    ```

    """  # noqa: E501

    # first, it needs to be ensured that the number of data points is enough to
    # support the kernel for the respective difference order at least once
    check_scalar(
        n_data,
        name="n_data",
        target_type=Integral,
        min_val=differences + 1,
        include_boundaries="left",
    )

    # afterwards, the squared forward finite differences matrix is computed
    if orig_first:
        return _gen_squ_fw_fin_diff_mat_cho_banded_orig_first(
            n_data=n_data,
            differences=differences,
        )

    return _gen_squ_fw_fin_diff_mat_cho_banded_transp_first(
        n_data=n_data,
        differences=differences,
    )
