import numpy as np
import pytest
from scipy.linalg import cholesky_banded
from scipy.sparse import eye as speye

from chemotools.utils.banded_linalg import (
    _find_largest_symm_sparse_banded_spd_eigval,
    _find_smallest_symm_sparse_banded_spd_eigval,
    conv_to_lu_banded_storage,
    lu_banded,
    lu_solve_banded,
    slodget_cho_banded,
    slogdet_lu_banded,
)
from chemotools.utils.finite_differences import (
    calc_forward_diff_kernel,
    forward_finite_diff_conv_matrix,
)

# from chemotools.utils.whittaker_base import WhittakerLikeSolver
from tests.fixtures import reference_finite_differences  # noqa: F401


def test_forward_diff_kernel(
    reference_finite_differences: list[tuple[int, int, np.ndarray]]  # noqa: F811
) -> None:
    # Arrange
    for differences, accuracy, reference in reference_finite_differences:
        # Act
        kernel = calc_forward_diff_kernel(differences=differences, accuracy=accuracy)

        # Assert
        assert kernel.size == reference.size, (
            f"Difference order {differences} with accuracy {accuracy} "
            f"expected kernel size {reference.size} but got {kernel.size}"
        )
        assert np.allclose(kernel, reference, atol=1e-8), (
            f"Difference order {differences} with accuracy {accuracy} "
            f"expected kernel\n{reference.tolist()}\n"
            f"but got\n{kernel.tolist()}"
        )


@pytest.mark.parametrize("accuracy", list(range(1, 21)))
@pytest.mark.parametrize("difference", list(range(0, 21)))
@pytest.mark.parametrize("size", [1, 2, 10, 50, 100, 500, 1000, 5000])
def test_forward_finite_diff_conv_matrix(
    size: int, difference: int, accuracy: int
) -> None:
    """Tests the generated convolution matrix for forward finite differences by
    comparing it to NumPy's ``convolve``.
    """

    # the test is skipped if the kernel is too large
    if difference + accuracy > size:
        pytest.skip(
            f"Test skipped because the kernel size {difference + 1} is larger than the "
            f"series size {size}."
        )
    # else nothing

    # the random signal is generated
    np.random.seed(seed=42)
    series = np.random.rand(size)

    # the kernel is computed ...
    kernel = calc_forward_diff_kernel(differences=difference, accuracy=accuracy)
    # ... and the random series is convolved with the kernel ...
    # NOTE: the kernel is flipped because of the way NumPy's convolve works
    numpy_convolved_series = np.convolve(series, np.flip(kernel), mode="valid")

    # the convolution matrix is computed ...
    conv_matrix = forward_finite_diff_conv_matrix(
        differences=difference, accuracy=accuracy, series_size=series.size
    )
    # ... and the series is convolved with the convolution matrix
    matrix_convolved_series = conv_matrix @ series

    # the actual test is performed
    assert np.allclose(matrix_convolved_series, numpy_convolved_series), (
        f"Differences by matrix product for Difference order {difference} with "
        f"accuracy {accuracy} for series of size {size} failed."
    )


@pytest.mark.parametrize("with_finite_check", [True, False])
@pytest.mark.parametrize("difference", list(range(0, 11)))
@pytest.mark.parametrize("size", [1, 2, 10, 50, 100, 500, 1000, 5000])
def test_stepwise_lu_banded_solve(
    size: int, difference: int, with_finite_check: bool
) -> None:
    """Tests the LU decomposition of a banded matrix by comparing the solution of the
    linear systems involved in Whittaker smoothing with the solution obtained by NumPy's
    ``solve``.
    It gets ill-condition for ``differences`` >> 10, but this is not the intended use
    case.
    """

    # the test is skipped if the kernel is too large
    if difference + 1 > size:
        pytest.skip(
            f"Test skipped because the kernel size {difference + 1} is larger than the "
            f"series size {size}."
        )
    # else nothing

    # a random right hand side vector is generated
    np.random.seed(seed=42)
    b = np.random.rand(
        size,
    )

    # a finite difference matrix is generated with an updated diagonal to
    # ensure positive definiteness
    l_and_u = (difference, difference)
    d = forward_finite_diff_conv_matrix(
        differences=difference, accuracy=1, series_size=size
    )
    a = d.T @ d + speye(size)

    # it is converted to LU banded storage ...
    ab = conv_to_lu_banded_storage(a=a, l_and_u=l_and_u)
    # ... its LU decomposition is computed ...
    lub, ipiv = lu_banded(
        l_and_u=l_and_u,
        ab=ab,
        overwrite_ab=False,
        check_finite=with_finite_check,
    )
    # ... and the linear system is solved
    x = lu_solve_banded(
        decomposition=(lub, ipiv, l_and_u),
        b=b,
        check_finite=with_finite_check,
    )

    # the solution is compared to the solution obtained by NumPy's
    # solve
    np_x = np.linalg.solve(a=a.toarray(), b=b)

    assert np.allclose(x, np_x), (
        f"Banded LU decomposition for matrix of size {size} with {difference} sub- and "
        f"superdiagonals failed."
    )


@pytest.mark.parametrize("with_finite_check", [True, False])
@pytest.mark.parametrize("difference", list(range(0, 11)))
@pytest.mark.parametrize("size", [1, 2, 10, 50, 100, 500, 1000, 5000])
def test_lu_banded_slogdet(size: int, difference: int, with_finite_check: bool) -> None:
    """Tests the computation of the sign and log determinant of a banded matrix from
    its LU decomposition by comparing it to NumPy's ``slogdet``.
    """

    # the test is skipped if the kernel is too large
    if difference + 1 > size:
        pytest.skip(
            f"Test skipped because the kernel size {difference + 1} is larger than the "
            f"series size {size}."
        )
    # else nothing

    # a finite difference matrix is generated with an updated diagonal to ensure
    # positive definiteness
    l_and_u = (difference, difference)
    d = forward_finite_diff_conv_matrix(
        differences=difference, accuracy=1, series_size=size
    )
    a = d.T @ d + speye(size)

    # it is converted to LU banded storage ...
    ab = conv_to_lu_banded_storage(a=a, l_and_u=l_and_u)
    # ... its LU decomposition is computed ...
    lub, ipiv = lu_banded(
        l_and_u=l_and_u,
        ab=ab,
        overwrite_ab=False,
        check_finite=with_finite_check,
    )
    # ... and the sign and log determinant are determined
    sign, logabsdet = slogdet_lu_banded(
        decomposition=(l_and_u, lub, ipiv),
    )

    # the sign and log determinant are compared to the values obtained by NumPy's
    # slogdet
    np_sign, np_logabsdet = np.linalg.slogdet(a=a.toarray())  # type: ignore

    assert np_sign > 0, (
        f"Sign of log determinant for matrix of size {size} with {difference} sub- and "
        f"superdiagonals failed."
    )

    assert np.isclose(sign, np_sign), (
        f"Sign of log determinant for matrix of size {size} with {difference} sub- and "
        f"superdiagonals failed."
    )
    assert np.isclose(logabsdet, np_logabsdet), (
        f"Log determinant for matrix of size {size} with {difference} sub- and "
        f"superdiagonals failed."
    )


@pytest.mark.parametrize("with_finite_check", [True, False])
@pytest.mark.parametrize("difference", list(range(0, 11)))
@pytest.mark.parametrize("size", [1, 2, 10, 50, 100, 500, 1000, 5000])
def test_cho_banded_slogdet(
    size: int, difference: int, with_finite_check: bool
) -> None:
    """Tests the computation of the sign and log determinant of a banded matrix from
    its Cholesky decomposition by comparing it to NumPy's ``slogdet``.
    """

    # the test is skipped if the kernel is too large
    if difference + 1 > size:
        pytest.skip(
            f"Test skipped because the kernel size {difference + 1} is larger than the "
            f"series size {size}."
        )

    # a finite difference matrix is generated with an updated diagonal to
    # ensure positive definiteness
    l_and_u = (difference, difference)
    d = forward_finite_diff_conv_matrix(
        differences=difference, accuracy=1, series_size=size
    )
    a = d.T @ d + speye(size)

    # it is converted to LU banded storage ...
    ab = conv_to_lu_banded_storage(a=a, l_and_u=l_and_u)
    # ... its Cholesky decomposition is computed ...
    lower = False
    chob = cholesky_banded(ab=ab[0 : difference + 1, ::], lower=lower)
    # ... and the sign and log determinant are determined
    sign, logabsdet = slodget_cho_banded(decomposition=(chob, lower))

    # the sign and log determinant are compared to the values obtained by
    # NumPy's slogdet
    np_sign, np_logabsdet = np.linalg.slogdet(a=a.toarray())  # type: ignore

    assert np.isclose(sign, np_sign), (
        f"Sign of log determinant for matrix of size {size} with {difference} sub- and "
        f"superdiagonals failed."
    )
    assert np.isclose(logabsdet, np_logabsdet), (
        f"Log determinant for matrix of size {size} with {difference} sub- and "
        f"superdiagonals failed."
    )


# FIXME: this test takes forever and is currently not even required, so the differences
#        screened was limited a lot
@pytest.mark.parametrize("with_finite_check", [True, False])
# @pytest.mark.parametrize("difference", list(range(0, 11)))
@pytest.mark.parametrize("difference", [0, 1, 2])
@pytest.mark.parametrize("size", [1, 2, 10, 50, 100, 500, 1000, 5000])
def test_largest_smallest_eigval_of_spbanded(
    size: int, difference: int, with_finite_check: bool
) -> None:
    """Tests the computation of the largest and smallest eigenvalues of a symmetric
    mathematically positive semi-definite banded matrix by comparing it to NumPy's
    ``eigvalsh``. Squared finite difference matrices are used for this test since they
    are symmetric and mathematically positive semi-definite.
    """

    # the test is skipped if the kernel is too large
    if difference + 1 > size:
        pytest.skip(
            f"Test skipped because the kernel size {difference + 1} is larger than the "
            f"series size {size}."
        )

    # a finite difference matrix is generated and squared
    l_and_u = (difference, difference)
    d = forward_finite_diff_conv_matrix(
        differences=difference, accuracy=1, series_size=size
    )
    a = d.T @ d
    ab = conv_to_lu_banded_storage(a=a, l_and_u=l_and_u)[0 : difference + 1, ::]

    # now, its largest and smallest eigenvalues are computed ...
    max_eigval = _find_largest_symm_sparse_banded_spd_eigval(
        ab=ab, check_finite=with_finite_check
    )
    min_eigval = _find_smallest_symm_sparse_banded_spd_eigval(
        ab=ab, check_finite=with_finite_check
    )

    # ... and compared to the values obtained by NumPy's ``eigvalsh``
    np_eigvals = np.linalg.eigvalsh(a=a.toarray())
    np_max_eigval = np_eigvals.max()
    np_min_eigval = np_eigvals.min()

    assert np.isclose(max_eigval, np_max_eigval), (
        f"Largest eigenvalue for matrix of size {size} with {difference} "
        f"sub- and superdiagonals failed. "
        f"Chemotools solution: {max_eigval} vs."
        f"NumPy's solution: {np_max_eigval}"
    )
    assert np.isclose(min_eigval, np_min_eigval), (
        f"Smallest eigenvalue for matrix of size {size} with {difference} "
        f"sub- and superdiagonals failed. "
        f"Chemotools solution {min_eigval} vs."
        f"NumPy's solution {np_min_eigval}"
    )
