"""
This utility submodule provides functions for the computation of forward finite
differences, namely

- the kernel for forward and central finite differences,
- computation of related kernel matrices
- estimation of the noise standard deviation of a series

"""

### Imports ###

from math import comb, factorial
from numbers import Integral, Real
from typing import Any, Callable, Dict, Literal, Optional, Tuple, Union

import numpy as np
from scipy.ndimage import median_filter
from sklearn.utils import check_scalar

### Constants ###

_MAD_PREFACTOR = 1.482602

### Functions ###


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
        If ``differences`` is below 1.

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


def calc_central_diff_kernel(*, differences: int, accuracy: int = 2) -> np.ndarray:
    """
    Computes the kernel for central finite differences which can be applied to a
    series by means of a convolution, e.g.,

    ```python
        kernel = calc_central_fin_diff_kernel(differences=2, accuracy=2)
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
    accuracy : int, default=2
        The accuracy of the finite difference approximation, which has to be an even
        integer ``>= 2``.
        The higher the accuracy, the better the approximation.

    Returns
    -------
    fin_diff_kernel : ndarray of shape (kernel_size,)
        A NumPy-1D-vector resembling the kernel from the code example above. Since the
        elements are not necessarily integers, the data type is ``np.float64``.
        Its size is given by ``2 * floor((differences + 1) / 2) - 1 + accuracy`` where
        ``floor`` returns the next lower integer.

    Raises
    ------
    ValueError
        If ``differences`` is below 1.
    ValueError
        If ``accuracy`` is not an even integer ``>= 2``.

    References
    ----------
    The computation is based on the description in [1]_.

    .. [1] Wikipedia, "Finite difference coefficient - Central finite difference",
    URL: https://en.wikipedia.org/wiki/Finite_difference_coefficient#Central_finite_difference

    """  # noqa: E501

    ### Input Validation ###

    # first, difference order and accuracy are validated
    check_scalar(
        differences,
        name="differences",
        target_type=Integral,
        min_val=1,
        include_boundaries="left",
    )

    check_scalar(
        accuracy,
        name="accuracy",
        target_type=Integral,
        min_val=2,
        include_boundaries="left",
    )
    if accuracy % 2 == 1:
        raise ValueError("Got accuracy = {accuracy}, expected an even integer.")

    ### Central Difference Kernel Computation ###

    # first, the size of the kernel is computed
    kernel_size = 2 * ((differences + 1) // 2) - 1 + accuracy
    half_kernel_size = kernel_size // 2

    # then, the linear system to solve for the coefficients is set up
    grid_vect = np.arange(
        start=-half_kernel_size,
        stop=half_kernel_size + 1,
        step=1,
        dtype=np.int64,
    )
    lhs_mat = np.vander(grid_vect, N=kernel_size, increasing=True).transpose()
    rhs_vect = np.zeros(shape=(kernel_size,), dtype=np.int64)
    rhs_vect[differences] = factorial(differences)

    # the coefficients are computed by solving the linear system
    return np.linalg.solve(lhs_mat, rhs_vect)


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
        If ``differences`` is below 1.

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
    try:
        check_scalar(
            n_data,
            name="n_data",
            target_type=Integral,
            min_val=differences + 1,
            include_boundaries="left",
        )

    # NOTE: this is only for Sklearn compatibility
    except ValueError:
        raise ValueError(f"Got n_features = {n_data}, must be >= {differences + 1}.")

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


def estimate_noise_stddev(
    series: np.ndarray,
    differences: int = 6,
    diff_accuracy: int = 2,
    window_length: Optional[int] = None,
    extrapolator: Callable[..., np.ndarray] = np.pad,
    extrapolator_args: Tuple[Any, ...] = ("reflect",),
    extrapolator_kwargs: Optional[Dict[str, Any]] = None,
    power: Literal[-2, -1, 1, 2] = 1,
    stddev_min: Union[float, int] = 1e-10,
) -> np.ndarray:
    """
    EXPERIMENTAL FEATURE

    Estimates the local/global noise standard deviation of a series even in the presence
    of trends, like baselines and peaks, as well as outliers by using central finite
    differences.
    Please see the Notes section for further details.

    Parameters
    ----------
    series : ndarray of shape (n_data,)
        The series for which the noise standard deviation is estimated.
    differences : int, default=6
        The order of the differences starting from 0 for the original curve, 1 for the
        first order, 2 for the second order, ..., and ``m`` for the ``m``-th order
        differences.
        Empirically, 5-6 was found as a sweet spot, but even numbers work better with
        the default ``extrapolator``.
        Values below 1 are not allowed.
    diff_accuracy : int, default=2
        The accuracy of the finite difference approximation, which has to be an even
        integer ``>= 2``.
        Higher values will enhance the effect of outliers that will corrupt the noise
        estimation of their neighborhood.
    window_length : int or None, default=None
        The odd window size around a datapoint to estimate its local noise standard
        deviation.
        Higher values will lead to a smoother noise standard deviation estimate by
        sacrificing the local resolution. At the same time, edge effects start blurring
        in if the ``extrapolator`` does not provide a good extrapolation.
        If provided, it has to be at least 1.
        If ``None``, the global noise standard deviation is estimated, i.e., it will
        be the same for each data point.
    extrapolator : callable, default=np.pad
        The extrapolator function that is used to pad the series before the finite
        differences and the median filter are applied. It will pad the signal with
        ``pad_width = (diff_kernel_size // 2) + (window_length // 2)`` elements on each
        side where ``diff_kernel_size`` is the size of the central finite differences
        kernel (see the Notes for details).
        It has to be a callable with the following signature:

        ```python
        series_extrap = extrapolator(
            series,
            pad_width,
            *extrapolator_args,
            **extrapolator_kwargs,
        )
        ```

        If ``window_length`` is ``None``, only the central finite differences kernel is
        considered.
        By default, the signal is padded by reflecting ``series`` at the edges on either
        side, but of course the quality of the noise estimation can be improved by using
        a more sophisticated extrapolation method.
    extrapolator_args : tuple, default=("reflect",)
        Additional positional arguments that are passed to the extrapolator function as
        described for ``extrapolator``.
    extrapolator_kwargs : dict or None, default=None
        Additional keyword arguments that are passed to the extrapolator function as
        described for ``extrapolator``.
        If ``None``, no additional keyword arguments are passed.
    power : {-2, -1, 1, 2}, default=1
        The power to which the noise standard deviation is raised.
        This can be used to compute the:

        - original noise standard deviation (``power=1``),
        - the noise variance (``power=2``),
        - the inverse noise standard deviation (``power=-1``), or
        - the inverse noise variance (``power=-2``; typically used as weights).

    stddev_min : float or int, default=1e-10
        The minimum noise standard deviation that is allowed.
        Any estimated noise standard deviation below this value will be set to this
        value.
        Borrowing an idea from image processing, the minimum noise standard deviation
        can, e.g., be estimated from one or more feature-free regions of ``series``.
        It must be at least ``1e-15``.

    Returns
    -------
    noise_stddev : ndarray of shape (n_data,)
        The estimated noise standard deviation raised to ``power`` for each data point
        in the series.

    Raises
    ------
    ValueError
        If ``series.size`` is below less than the kernel or window size (see Notes for
        details).
    ValueError
        If ``differences`` is below 1.
    ValueError
        If ``diff_accuracy`` is not an even integer ``>= 2``.
    ValueError
        If ``window_length`` is below 1.


    References
    ----------
    The estimation algorithm is an adaption of the global estimation logic applied for
    the "DER SNR" proposed in [1]_ (see the Notes for further details).

    .. [1] Stoehr F., et al., "DER SNR: A Simple & General Spectroscopic Signal-to-Noise
    Measurement Algorithm", Astronomical Data Analysis Software and Systems XVII P5.4
    ASP Conference Series, Vol. XXX, 2008

    Notes
    -----
    The "DER SNR" algorithm estimates a global noise level in a robust fashion by
    applying a modified version of the Median Absolute Deviation (MAD) to the
    derivative/differences of the signal. By using a moving MAD filter, the local noise
    level can be estimated as well.

    From a workflow perspective, the following steps are performed on the signal:

    - The signal is extrapolated to avoid edge effects.
    - The central finite differences are computed.
    - Their absolute values are taken.
    - The median (global) or median filter (local) is applied to these absolute
        differences. With proper scaling, this will give an estimate of the noise level.

    There is one limitation, namely that the algorithm does not work well for signals
    that are perfectly noise-free, but this is a rare case in practice.

    The kernel size for the central finite difference kernel is given by
    ``2 * floor((differences + 1) / 2) - 1 + diff_accuracy``.

    """

    ### Input Validation ###

    # first, the window size, power, and minimum standard deviation are validated
    # NOTE: the difference order and accuracy are by the central finite differences
    #       kernel function
    # window size
    if window_length is not None:
        check_scalar(
            window_length,
            name="window_length",
            target_type=Integral,
            min_val=1,
            include_boundaries="left",
        )
        if window_length % 2 == 0:
            raise ValueError(
                f"Got window_length = {window_length}, expected an odd integer."
            )

    # power
    if power not in {-2, -1, 1, 2}:
        raise ValueError(f"Got power = {power}, expected -2, -1, 1, or 2.")

    # minimum standard deviation
    check_scalar(
        stddev_min,
        name="stddev_min",
        target_type=Real,
        min_val=1e-15,
        include_boundaries="left",
    )

    # for validation of the series, the central finite differences kernel has to be
    # computed
    diff_kernel = calc_central_diff_kernel(
        differences=differences,
        accuracy=diff_accuracy,
    )

    # afterwards, the series is validated
    if series.size < diff_kernel.size:
        raise ValueError(
            f"Got series.size = {series.size}, must be >= {diff_kernel.size} (kernel "
            f"size)."
        )

    if window_length is not None:
        if series.size < window_length:
            raise ValueError(
                f"Got series.size = {series.size}, must be >= {window_length} (window "
                "size)."
            )

    ### Preparation ###

    # the keyword arguments for the extrapolator are set up
    extrapolator_kwargs = (
        extrapolator_kwargs if extrapolator_kwargs is not None else dict()
    )

    ### Noise Standard Deviation Estimation ###

    # the signal is extrapolated to avoid edge effects
    pad_width = diff_kernel.size // 2
    pad_width += 0 if window_length is None else window_length // 2
    series_extrap = extrapolator(
        series,
        pad_width,
        *extrapolator_args,
        **extrapolator_kwargs,
    )

    # the absolute central finite differences are computed ...
    abs_diff_series = np.abs(
        np.convolve(series_extrap, np.flip(diff_kernel), mode="valid")
    )
    size_after_diff = abs_diff_series.size

    # ... and the median filter is applied to theses differences
    prefactor = _MAD_PREFACTOR / np.linalg.norm(diff_kernel)
    # Case 1: the global noise standard deviation is estimated
    if window_length is None:
        noise_stddev = np.full_like(
            series,
            fill_value=prefactor * np.median(abs_diff_series),
        )

    # Case 2: the local noise standard deviation is estimated
    else:
        half_window_length = window_length // 2
        noise_stddev = (
            prefactor
            * median_filter(
                abs_diff_series,
                size=window_length,
                mode="constant",
            )[half_window_length : size_after_diff - half_window_length]
        )

    # the minimum-bounded noise standard deviation is raised to the power
    noise_stddev = np.maximum(noise_stddev, stddev_min)

    if power in {-2, 2}:
        noise_stddev = np.square(noise_stddev)

    if power in {-2, -1}:
        noise_stddev = np.reciprocal(noise_stddev)

    return noise_stddev
