from math import factorial
from numbers import Integral

import numpy as np
from scipy.sparse import dia_matrix
from scipy.sparse import diags as spdiags
from sklearn.utils import check_scalar


def _calc_arbitrary_fin_diff_kernel(
    *,
    grid_points: np.ndarray,
    differences: int,
) -> np.ndarray:
    """Computes the kernel for finite differences with arbitrary grid points."""
    # the number of grid points is counted
    num_grid_points = grid_points.size

    # if the grid points cannot support the respective difference, an error is raised
    if differences >= num_grid_points:
        raise ValueError(
            f"\n{num_grid_points} grid points cannot support a {differences}-th order "
            f"difference."
        )
    # else nothing

    # then, the system of linear equations to solve is set up as A@x = b where x is
    # the kernel vector
    lhs_mat_a = np.vander(x=grid_points, N=num_grid_points, increasing=True).T
    rhs_vect_b = np.zeros(shape=(num_grid_points,), dtype=np.float64)
    rhs_vect_b[differences] = factorial(differences)

    # the kernel is computed and returned
    return np.linalg.solve(a=lhs_mat_a, b=rhs_vect_b)


def calc_forward_diff_kernel(
    *,
    differences: int,
    accuracy: int = 1,
) -> np.ndarray:
    """Computes the kernel for forward finite differences which can be applied to a
    series by means of a convolution, e.g.,

    ```python
        kernel = calc_forward_fin_diff_kernel(differences=2, accuracy=1)
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
    accuracy : int, default=1
        The accuracy of the approximation which must be a positive integer starting
        from 1.

    Returns
    -------
    fin_diff_kernel : np.ndarray of shape (differences + accuracy,)
        A NumPy-1D-vector resembling the kernel from the code example above.

    Raises
    ------
    ValueError
        If the difference order is below 0, the accuracy is below 1, or the number of
        grid points is not sufficient to support the respective difference order.

    """
    # the input is validated
    check_scalar(
        differences,
        name="differences",
        target_type=Integral,
        min_val=0,
        include_boundaries="left",
    )
    check_scalar(
        accuracy,
        name="accuracy",
        target_type=Integral,
        min_val=1,
        include_boundaries="left",
    )

    # afterwards, the number of grid points is evaluated, which is simply the sum of the
    # difference order and the accuracy
    num_grid_points = differences + accuracy

    # then, the system of linear equations is solved for the x in A@x = b since x is
    # the kernel vector
    grid_points = np.arange(
        start=0,
        stop=num_grid_points,
        step=1,
        dtype=np.float64,
    )
    fin_diff_kernel = _calc_arbitrary_fin_diff_kernel(
        grid_points=grid_points, differences=differences
    )

    return fin_diff_kernel


def forward_finite_diff_conv_matrix(
    *,
    differences: int,
    accuracy: int = 1,
    series_size: int,
) -> dia_matrix:
    """Computes the convolution matrix for forward finite differences which can be
    applied to a series by means of a matrix multiplication, e.g.,

    ```python
        conv_mat = finite_diff_conv_matrix(differences=2, accuracy=1, series_size=10)
        differences = conv_mat @ series # boundaries require special care
    ```

    this is equivalent to

    ```python
        kernel = calc_forward_fin_diff_kernel(differences=2, accuracy=1)
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
    accuracy : int, default=1
        The accuracy of the approximation which must be a positive integer starting
        from 1.
    series_size : int
        The number of data points in the series to which the convolution matrix is
        applied.

    Returns
    -------
    conv_mat : dia_matrix of shape (series_size - differences, series_size)
        A sparse matrix resembling the convolution matrix from the code example above.

    Raises
    ------
    ValueError
        If the difference order is below 0, the accuracy is below 1, or the number of
        grid points is not sufficient to support the respective difference order.
    ValueError
        If ``series_size`` is not enough to support the respective ``differences`` and
        ``accuracy``.

    """
    # the input is validated (``differences`` and ``accuracy`` are validated in the
    # function ``calc_forward_diff_kernel``)
    kernel_size = differences + accuracy
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
    kernel = calc_forward_diff_kernel(differences=differences, accuracy=accuracy)
    # ... and the convolution matrix is created
    return spdiags(
        diagonals=kernel,
        offsets=np.arange(start=0, stop=kernel_size, step=1),  # type: ignore
        shape=(series_size - kernel_size + 1, series_size),
        format="dia",
    )
