from typing import Literal

import numpy as np
import pytest
from scipy.linalg import cholesky_banded, eigvals_banded, solve_banded, solveh_banded
from scipy.sparse import eye as speye

from chemotools.utils.banded_linalg import (
    conv_to_lu_banded_storage,
    lu_banded,
    lu_solve_banded,
    slodget_cho_banded,
)
from chemotools.utils.finite_differences import (
    calc_forward_diff_kernel,
    calc_limit_max_eigval_fin_diff_mat,
    forward_finite_diff_conv_matrix,
    posdef_mod_squared_fw_fin_diff_conv_matrix,
)
from chemotools.utils.whittaker_base import WhittakerLikeSolver
from tests.fixtures import reference_finite_differences  # noqa: F401


def test_forward_diff_kernel(
    reference_finite_differences: list[tuple[int, int, np.ndarray]]  # noqa: F811
) -> None:
    # Arrange
    for differences, _, reference in reference_finite_differences:
        # Act
        kernel = calc_forward_diff_kernel(differences=differences)

        # Assert
        assert kernel.size == reference.size, (
            f"Difference order {differences} with accuracy 1 expected kernel size "
            f"{reference.size} but got {kernel.size}"
        )
        assert np.allclose(kernel, reference, atol=1e-8), (
            f"Difference order {differences} with accuracy 1 expected kernel "
            f"{reference.tolist()} but got {kernel.tolist()}"
        )


@pytest.mark.parametrize("difference", list(range(0, 21)))
@pytest.mark.parametrize("size", [1, 2, 10, 50, 100, 500, 1_000, 5_000])
def test_forward_finite_diff_conv_matrix(size: int, difference: int) -> None:
    """Tests the generated convolution matrix for forward finite differences by
    comparing it to NumPy's ``convolve``.
    """

    # the test is skipped if the kernel is too large
    if difference + 1 > size:
        pytest.skip(
            f"Test skipped because the kernel size {difference + 1} is larger than the "
            f"series size {size}."
        )
    # else nothing

    # the random signal is generated
    np.random.seed(seed=42)
    series = np.random.rand(size)

    # the kernel is computed ...
    kernel = calc_forward_diff_kernel(differences=difference)
    # ... and the random series is convolved with the kernel ...
    # NOTE: the kernel is flipped because of the way NumPy's convolve works
    numpy_convolved_series = np.convolve(series, np.flip(kernel), mode="valid")

    # the convolution matrix is computed ...
    conv_matrix = forward_finite_diff_conv_matrix(
        differences=difference, series_size=series.size
    )
    # ... and the series is convolved with the convolution matrix
    matrix_convolved_series = conv_matrix @ series

    # the actual test is performed
    assert conv_matrix.dtype == np.int64, (
        f"Convolution matrix for difference order {difference} with accuracy 1 for "
        f"series of size {size} expected data type np.int64 but got "
        f"{conv_matrix.dtype}."
    )

    assert np.allclose(matrix_convolved_series, numpy_convolved_series), (
        f"Differences by matrix product for difference order {difference} with "
        f"accuracy 1 for series of size {size} failed."
    )


@pytest.mark.parametrize("with_finite_check", [True, False])
@pytest.mark.parametrize("overwrite_b", [True, False])
@pytest.mark.parametrize("n_rhs", [0, 1, 2])
@pytest.mark.parametrize("n_upp_bands", [1, 2, 3, 4, 5, 6])
@pytest.mark.parametrize("n_low_bands", [1, 2, 3, 4, 5, 6])
@pytest.mark.parametrize(
    "n_rows", [1, 2, 10, 11, 50, 51, 100, 101, 500, 501, 1_000, 1001, 5_000, 5001]
)
def test_lu_banded_solve(
    n_rows: int,
    n_low_bands: int,
    n_upp_bands: int,
    n_rhs: int,
    overwrite_b: bool,
    with_finite_check: bool,
) -> None:
    """
    Tests the separate LU decomposition followed by solving a system of linear equations
    for banded matrices.

    NOTE: A number of 0 right-hand sides are used for making the vector to solve a
    NOTE: 1D-Array.

    """

    # if the matrix cannot exist with the given shape, the test is skipped
    n_rows_min = n_low_bands + n_upp_bands + 1
    if n_rows < n_rows_min:
        pytest.skip(
            f"Test skipped because the number of rows {n_rows} is smaller than the "
            f"minimum number of rows {n_rows_min} required by the number of sub- "
            f"{n_low_bands} and superdiagonals {n_upp_bands}."
        )

    # a random banded matrix and right-hand-side-vector/-matrix are generated
    np.random.seed(seed=42)
    ab = np.random.rand(n_low_bands + n_upp_bands + 1, n_rows)
    b = np.random.rand(n_rows) if n_rhs == 0 else np.random.rand(n_rows, n_rhs)

    # first, the Scipy solution is computed because if this fails due to singularity,
    # the test has to not test for equivalent results, but for failure
    # NOTE: failure is indicated by the solution being ``None``
    # NOTE: this order of evaluation is also better for testing if the overwrite flag
    # NOTE: is working correctly because otherwise SciPy would get the overwritten b
    l_and_u = (n_low_bands, n_upp_bands)
    x_ref = None
    try:
        x_ref = solve_banded(
            l_and_u=l_and_u,
            ab=ab,
            b=b,
            check_finite=True,
        )

        # NOTE: even if SciPy computes the solution "successfully", there might be NaNs
        # NOTE: in the result, so the test has to check for that as well
        if np.any(np.isnan(x_ref)):
            x_ref = None

    except np.linalg.LinAlgError:
        pass

    # the banded matrix is LU decomposed with the respective Chemotools function
    lu_fact = lu_banded(
        l_and_u=l_and_u,
        ab=ab,
        check_finite=with_finite_check,
    )

    # the linear system is solved with the respective Chemotools function
    # Case 1: Scipy failed
    if x_ref is None:
        # in this case, the Chemotools function has to raise an exception as well
        with pytest.raises(np.linalg.LinAlgError):
            x = lu_solve_banded(
                lub_factorization=lu_fact,
                b=b,
                overwrite_b=overwrite_b,
                check_finite=with_finite_check,
            )
        return

    # Case 2: Scipy succeeded
    # in this case, the Chemotools function has to return the same result as Scipy
    x = lu_solve_banded(
        lub_factorization=lu_fact,
        b=b,
        overwrite_b=overwrite_b,
        check_finite=with_finite_check,
    )

    # NOTE: the following check has to be fairly strict when it comes to equivalence
    # NOTE: since the SciPy and Chemotools are basically doing the same under the hood
    # NOTE: when it comes to the solution process (first LU, then triangular solve)
    assert np.allclose(x, x_ref, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("difference", list(range(0, 21)))
def test_limit_eigval_squ_fin_diff_mat(difference: int) -> None:
    """Tests the computation of the limit of the maximum eigenvalue of the squared
    forward finite difference matrix.
    """

    # the limit of the maximum eigenvalue is computed empirically
    series_size = 10000
    eigval_max_empirical = calc_limit_max_eigval_fin_diff_mat(differences=difference)
    # ... and compared to the reference value
    squ_diff_mat = forward_finite_diff_conv_matrix(
        differences=difference, series_size=series_size
    )
    squ_diff_mat = squ_diff_mat.T @ squ_diff_mat
    squ_diff_mat_b = conv_to_lu_banded_storage(
        a=squ_diff_mat, l_and_u=(difference, difference)
    ).astype(np.float64)[difference:, ::]
    eigval_max_reference = eigvals_banded(
        a_band=squ_diff_mat_b,
        lower=True,
        select="i",
        select_range=(series_size - 1, series_size - 1),
    )[0]

    assert np.isclose(eigval_max_empirical, eigval_max_reference), (
        f"Empirical limit of the maximum eigenvalue for difference order {difference} "
        f"with accuracy 1 for series of size {series_size} failed."
    )


@pytest.mark.parametrize("difference", list(range(0, 21)))
@pytest.mark.parametrize(
    "size",
    np.arange(start=1, stop=1001, step=1).tolist()
    + np.arange(start=1000, stop=100000, step=2500).tolist(),
)
def test_posdef_squ_fin_diff_conv_matrix(size: int, difference: int) -> None:
    """Tests the generated convolution matrix for forward finite differences by
    comparing it against itself after SciPy's ``solveh_banded`` has been applied.
    """

    # the test is skipped if the kernel is too large
    if difference + 1 > size:
        pytest.skip(
            f"Test skipped because the kernel size {difference + 1} is larger than the "
            f"series size {size}."
        )
    # else nothing

    # the random signal is generated
    min_eigval_size = 5000
    np.random.seed(seed=42)
    series = np.random.rand(size)

    # this is solved against a finite difference matrix with an updated diagonal to
    # ensure positive definiteness
    l_and_u = (difference, difference)
    max_eigval_mult = max(min_eigval_size, size) * np.finfo(np.float64).eps  # type: ignore # noqa: E501
    fw_fin_diff_mat = forward_finite_diff_conv_matrix(
        differences=difference,
        series_size=size,
    )
    squ_diff_mat = posdef_mod_squared_fw_fin_diff_conv_matrix(
        fw_fin_diff_mat=fw_fin_diff_mat,
        differences=difference,
        dia_mod_matrix=None,
        max_eigval_mult=max_eigval_mult,
        dtype=np.float64,
    )

    assert squ_diff_mat.dtype == np.float64, (  # type: ignore
        f"Convolution matrix for difference order {difference} with accuracy 1 for "
        f"series of size {size} expected data type np.float64 but got "
        f"{squ_diff_mat.dtype}."
    )

    # now, the linear system is solved
    ab = conv_to_lu_banded_storage(a=squ_diff_mat, l_and_u=l_and_u)
    x = solveh_banded(
        ab=ab[difference:, ::],
        b=series,
        lower=True,
    )

    # NOTE: ``atol`` is set relatively high because the matrix is not very well
    #       conditioned and an approximate result is expected but also fine
    assert np.allclose(series, squ_diff_mat @ x, atol=5e-4), (
        f"Positive definite squared convolution matrix for difference order "
        f"{difference} with accuracy 1 for series of size {size} failed."
    )


@pytest.mark.parametrize("with_finite_check", [True, False])
@pytest.mark.parametrize("difference", list(range(0, 11)))
@pytest.mark.parametrize("size", [1, 2, 10, 50, 100, 500, 1_000, 5_000])
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
    fw_fin_diff_mat = forward_finite_diff_conv_matrix(
        differences=difference,
        series_size=size,
    )
    a = posdef_mod_squared_fw_fin_diff_conv_matrix(
        fw_fin_diff_mat=fw_fin_diff_mat,
        differences=difference,
        dia_mod_matrix=None,
        max_eigval_mult=0.0,
        dtype=np.float64,
    )
    a += speye(size, dtype=np.int64)  # type: ignore

    # it is converted to LU banded storage ...
    ab = conv_to_lu_banded_storage(a=a, l_and_u=l_and_u).astype(np.float64)
    # ... its Cholesky decomposition is computed ...
    lower = False
    chob = cholesky_banded(
        ab=ab[0 : difference + 1, ::], lower=lower, check_finite=with_finite_check
    )
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


@pytest.mark.parametrize("same_weights_for_all", [True, False])
@pytest.mark.parametrize("with_weights", [True, False, "bad"])
@pytest.mark.parametrize("with_pentapy", [True, False])
@pytest.mark.parametrize("log10_lam", np.arange(-50.0, 170.0, step=20.0).tolist())
@pytest.mark.parametrize("nrows", [1, 2])
@pytest.mark.parametrize(
    "size", [3, 11, 50, 100, 500, 1_000, 5_000, 10_000, 50_000, 100_000]
)
@pytest.mark.parametrize("difference", [2, 10])
def test_whittaker_solve(
    difference: int,
    size: int,
    nrows: int,
    log10_lam: float,
    with_pentapy: bool,
    with_weights: bool | Literal["bad"],
    same_weights_for_all: bool,
) -> None:
    """Tests if the Whittaker smoothing still works for very low and large values of the
    smoothing parameter combined with different numerically challenging weights. If it
    survives this, arbitrary combinations can be considered safe.
    """

    # the test is skipped if the kernel is too large
    if difference + 1 > size:
        pytest.skip(
            f"Test skipped because the kernel size {difference + 1} is larger than the "
            f"series size {size}."
        )
    # else nothing

    # a Whittaker solver is instantiated ...
    whittaker_solver = WhittakerLikeSolver()
    # ... pentapy is enabled if requested ...
    whittaker_solver._WhittakerLikeSolver__allow_pentapy = with_pentapy  # type: ignore
    whittaker_solver._setup_for_fit(
        series_size=size,
        lam=10.0**log10_lam,
        differences=difference,
    )

    # ... weights are generated ...
    np.random.seed(seed=42)
    if with_weights:
        weights = np.random.rand(1, size)
        if with_weights == "bad":
            idxs = np.arange(start=0, stop=size, step=1, dtype=np.int64)
            weights[0, np.random.choice(idxs, size=int(size / 2), replace=False)] = 0.0
        # else nothing
    else:
        weights = None

    if not same_weights_for_all and weights is not None:
        weights = np.tile(weights.reshape((1, -1)), reps=(nrows, 1))
    # else nothing

    # ... and the linear system is solved
    z = whittaker_solver._whittaker_solve(
        X=np.random.rand(nrows, size),
        w=weights,
        use_same_w_for_all=same_weights_for_all,
    )[0]

    assert np.all(np.isfinite(z)), (
        f"Whittaker solver for series of size {size} with smoothing parameter "
        f"{10.0 ** log10_lam} and weights {weights} failed."
    )
