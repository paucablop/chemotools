from math import comb
from numbers import Integral

import numpy as np
from scipy.sparse import csr_matrix, dia_matrix
from scipy.sparse import diags as spdiags
from scipy.sparse import eye as speye
from sklearn.utils import check_scalar


def calc_forward_diff_kernel(
    *,
    differences: int,
) -> np.ndarray:
    """Computes the kernel for forward finite differences which can be applied to a
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
        Values below 0 are not allowed.

    Returns
    -------
    fin_diff_kernel : np.ndarray of shape (differences + 1,)
        A NumPy-1D-vector resembling the kernel from the code example above. To avoid
        loss of precision, the data type is ``np.int64``.

    Raises
    ------
    ValueError
        If the difference order is below 0.

    """
    # the input is validated
    check_scalar(
        differences,
        name="differences",
        target_type=Integral,
        min_val=0,
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


def forward_finite_diff_conv_matrix(
    *,
    differences: int,
    series_size: int,
) -> dia_matrix:
    """Computes the convolution matrix for forward finite differences which can be
    applied to a series by means of a matrix multiplication, e.g.,

    ```python
        conv_mat = finite_diff_conv_matrix(differences=2, series_size=10)
        differences = conv_mat @ series # boundaries require special care
    ```

    this is equivalent to

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
        Values below 0 are not allowed.
    series_size : int
        The number of data points in the series to which the convolution matrix is
        applied.

    Returns
    -------
    diff_mat : dia_matrix of shape (series_size - differences, series_size)
        A sparse matrix resembling the convolution matrix from the code example above.
        To avoid loss of precision, the data type is ``np.int64``.

    Raises
    ------
    ValueError
        If the difference order is below 0, or ``series_size`` is not sufficient to
        support the respective difference order.

    """

    # the input is validated
    kernel_size = differences + 1
    try:
        check_scalar(
            series_size,
            name="n_features",  # for compatibility with sklearn
            target_type=Integral,
            min_val=kernel_size,
            include_boundaries="left",
        )
    except ValueError:
        raise ValueError(f"Got n_features = {series_size}, must be >= {kernel_size}.")

    # afterwards, the kernel is computed ...
    kernel = calc_forward_diff_kernel(differences=differences)
    # ... and the convolution matrix is created
    return spdiags(
        diagonals=kernel,
        offsets=np.arange(start=0, stop=kernel_size, step=1),  # type: ignore
        shape=(series_size - kernel_size + 1, series_size),
        format="dia",
        dtype=np.int64,
    )


def calc_limit_max_eigval_fin_diff_mat(differences: int) -> int:
    """Computes the maximum eigenvalue of the forward finite difference matrix as
    computed by ``forward_finite_diff_conv_matrix`` for the given difference order. It
    only uses the limit value as the series size tends to infinity, but from some
    tests, this seems to be an upper limit for the maximum singular value for any
    series size which makes it ideal for thresholding.

    Parameters
    ----------
    differences : int
        The order of the differences starting from 0 for the original curve, 1 for the
        first order, 2 for the second order, ..., and ``m`` for the ``m``-th order
        differences.
        Values below 0 are not allowed.

    Returns
    -------
    max_eigval : int
        The maximum eigenvalue of the forward finite difference matrix.

    """

    # NOTE: this was found rather empirically, but it works
    return 4**differences


def posdef_mod_squared_fw_fin_diff_conv_matrix(
    *,
    fw_fin_diff_mat: dia_matrix,
    differences: int,
    dia_mod_matrix: dia_matrix | None,
    max_eigval_mult: float,
    dtype: type,
) -> csr_matrix:
    """Computes the modified squared forward finite difference matrix ``P`` for the
    given difference order and series size. It is computed as

    ```python
    # the pre-computation is obtained which might still be positive semi-definite
    P = D.T @ M @ D
    # the maximum eigenvalue of P is estimated to make it positive definite
    max_lam_p = max_lam_dtd * max_lam_m
    # by lifting the main diagonal, P is made numerical positive definite
    P += max_lam_p * max_eigval_mult * I
    ```

    where ``D`` is the convolution matrix for forward finite differences, ``M`` is the
    diagonal matrix of the modified weights, ``max_lam_dtd`` is the maximum eigenvalue
    ``D.T @ D``, ``max_lam_m`` is the maximum eigenvalue of ``M``, i.e., the maximum
    weight (since diagonal matrix), and ``max_lam_p`` is the maximum eigenvalue of
    ``P``. For details on this approximation, please see Notes.

    Parameters
    ----------
    fw_fin_diff_mat : dia_matrix
        The convolution matrix for forward finite differences resembling ``D`` from the
        description above. It can be computed by ``forward_finite_diff_conv_matrix``.
    differences : int
        The order of the differences starting from 0 for the original curve, 1 for the
        first order, 2 for the second order, ..., and ``m`` for the ``m``-th order
        differences.
        Values below 0 are not allowed.
    dia_mod_matrix : dia_matrix or None
        The sparse diagonal matrix of the modification weights resembling ``M`` from the
        description above. If ``None``, this multiplication is skipped.
    max_eigval_mult : float
        The multiple of the maximum eigenvalue of the modified squared forward finite
        differences matrix that is added to the main diagonal of the output matrix to
        make it positive definite according to the description above.
    dtype : type
        The data type of the output matrix.

    Returns
    -------
    posdef_squ_diff_mat : csr_matrix of shape (series_size - differences, series_size)
        A positive definite sparse matrix resembling the squared forward finite
        difference matrix ``P`` from the description above. It will be of data type
        ``dtype``.

    Raises
    ------
    ValueError
        If the difference order is below 0, or the number of grid points is not
        sufficient to support the respective difference order.

    Notes
    -----
    The approximation of ``max_lam_p`` is based on the spectral norm of the
    matrix product. Since the spectral norm ``||P||2`` is submultiplicative, the
    estimate ``||D.T||2 * ||M||2 * ||D||2`` is an upper bound for ``||P||2``.
    As ``||D||2 = ||D.T||2 = sqrt(max_lam_dtd)`` and
    ``||M||2 = sqrt(max_lam_m**2) = max_lam_m = M.max()``, the maximum eigenvalue of
    ``P`` is estimated as ``max_lam_p = max_lam_dtd * max_lam_m``. For just ensuring
    numerical stability, this is perfectly fine and it also won't overestimate the
    maximum eigenvalue of ``P`` too much and therefore, the perturbation of the main
    diagonal is kept small.

    """

    # first, the maximum eigenvalue of the finite difference matrix is computed
    squ_diff_mat_eigval_max = calc_limit_max_eigval_fin_diff_mat(
        differences=differences
    )

    # afterwards, the squared convolution matrix is computed
    if dia_mod_matrix is None:
        squ_diff_mat = fw_fin_diff_mat.T @ fw_fin_diff_mat
        m_eigval_max = 1.0
    else:
        squ_diff_mat = fw_fin_diff_mat.T @ dia_mod_matrix @ fw_fin_diff_mat
        m_eigval_max = dia_mod_matrix.data.max()

    # the main diagonal is lifted by a multiple of the machine epsilon
    lift_mat = speye(m=fw_fin_diff_mat.shape[1], dtype=dtype, format="csr")
    lift_mat *= max_eigval_mult * squ_diff_mat_eigval_max * m_eigval_max

    # the positive definite matrix is returned
    return squ_diff_mat + lift_mat
