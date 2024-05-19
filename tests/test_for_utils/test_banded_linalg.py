"""
Test suite for the utility functions in the :mod:`chemotools.utils.banded_linalg`
module.

"""

### Imports ###

from typing import List, Union

import numpy as np
import pytest
from scipy.linalg import solve_banded as scipy_solve_banded

from chemotools.utils.banded_linalg import (
    _datacopied,
    conv_upper_chol_banded_to_lu_banded_storage,
    lu_banded,
    lu_solve_banded,
    slogdet_lu_banded,
)
from tests.test_for_utils.utils_funcs import get_banded_slogdet

### Constants ###

_ARRAY_TO_VIEW: np.ndarray = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
_VIEW = _ARRAY_TO_VIEW[::]

### Test Suite ###


@pytest.mark.parametrize(
    "arr, original, expected",
    [
        (  # Number 0 Different arrays
            np.array([1, 2, 3]),
            np.array([1, 2, 3]),
            True,
        ),
        (  # Number 1 Array and list
            np.array([1, 2, 3]),
            [1, 2, 3],
            True,
        ),
        (  # Number 2 Different data types
            np.array([1, 2, 3]),
            np.array([1, 2, 3], dtype=np.float64),
            True,
        ),
        (  # Number 3 Different view and array
            _ARRAY_TO_VIEW[0:3],
            np.array([1, 2, 3]),
            False,
        ),
        (  # Number 4 Same array
            _ARRAY_TO_VIEW,
            _ARRAY_TO_VIEW,
            False,
        ),
        (  # Number 5 Same view and array
            _VIEW,
            _ARRAY_TO_VIEW,
            False,
        ),
    ],
)
def test_datacopied(
    arr: np.ndarray,
    original: Union[np.ndarray, List],
    expected: bool,
) -> None:
    """
    Tests the function that checks if a NumPy array has been copied from another array
    or list.

    """

    assert _datacopied(arr, original) == expected


@pytest.mark.parametrize("with_finite_check", [True, False])
@pytest.mark.parametrize("overwrite_b", [True, False])
@pytest.mark.parametrize("n_rhs", [0, 1, 2])
@pytest.mark.parametrize("n_upp_bands", [1, 2, 3, 4, 5, 6])
@pytest.mark.parametrize("n_low_bands", [1, 2, 3, 4, 5, 6])
@pytest.mark.parametrize(
    "n_rows",
    [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        50,
        51,
        100,
        101,
        500,
        501,
        1_000,
        1001,
        5_000,
        5001,
    ],
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
    for banded matrices against the SciPy solution.

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
    ab = -1.0 + 2.0 * np.random.rand(n_low_bands + n_upp_bands + 1, n_rows)
    b = np.random.rand(n_rows) if n_rhs == 0 else np.random.rand(n_rows, n_rhs)

    # first, the Scipy solution is computed because if this fails due to singularity,
    # the test has to not test for equivalent results, but for failure
    # NOTE: failure is indicated by the solution being ``None``
    # NOTE: this order of evaluation is also better for testing if the overwrite flag
    #       is working correctly because otherwise SciPy would get the overwritten b
    l_and_u = (n_low_bands, n_upp_bands)
    x_ref = None
    try:
        x_ref = scipy_solve_banded(
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


@pytest.mark.parametrize("with_finite_check", [True, False])
@pytest.mark.parametrize("ensure_posdef", [True, False])
@pytest.mark.parametrize("n_upp_low_bands", [1, 2, 3, 4, 5, 6])
@pytest.mark.parametrize(
    "n_rows",
    [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        50,
        51,
        100,
        101,
        500,
        501,
        1_000,
        1_001,
        5_000,
        5_001,
    ],
)
def test_lu_banded_slogdet(
    n_rows: int,
    n_upp_low_bands: int,
    ensure_posdef: bool,
    with_finite_check: bool,
) -> None:
    """
    Tests the computation of the sign and log determinant of a banded matrix from
    its LU decomposition by comparing it to NumPy's ``slogdet``.

    """

    # if the matrix cannot exist with the given shape, the test is skipped
    n_rows_min = 2 * n_upp_low_bands + 1
    if n_rows < n_rows_min:
        pytest.skip(
            f"Test skipped because the number of rows {n_rows} is smaller than the "
            f"minimum number of rows {n_rows_min} required by the number of sub- "
            f"{n_upp_low_bands} and superdiagonals {n_upp_low_bands}."
        )

    # a random banded matrix is generated in the upper banded storage used for Cholesky
    # decomposition
    np.random.seed(seed=42)
    # NOTE: the diagonal lifting ensures that the matrix is positive and diagonally
    #       dominant, which makes it positive definite, but this is only done if the
    #       flag is set
    # NOTE: for an indefinite matrix, the matrix is shifted and scaled to be in the
    #       interval [-1, 1]
    ab_for_chol = np.random.rand(n_upp_low_bands + 1, n_rows)
    if ensure_posdef:
        ab_for_chol[n_upp_low_bands, ::] += 1.0 + 2.0 * float(n_upp_low_bands)
    else:
        ab_for_chol = -1.0 + 2.0 * ab_for_chol

    l_and_u, ab_for_lu = conv_upper_chol_banded_to_lu_banded_storage(ab=ab_for_chol)

    # first, the log determinant is computed with the literal definition as the sum of
    # the logarithms of the eigenvalues of the matrix
    sign_ref, logabsdet_ref = get_banded_slogdet(ab=ab_for_chol)

    # the banded matrix is LU decomposed with the respective Chemotools function ...
    lu_fact = lu_banded(
        l_and_u=l_and_u,
        ab=ab_for_lu,
        check_finite=with_finite_check,
    )
    # ... and the sign and log determinant are computed
    sign, logabsdet = slogdet_lu_banded(lub_factorization=lu_fact)

    # the results are compared
    assert np.isclose(sign, sign_ref, atol=1e-5, rtol=1e-5)
    assert np.isclose(logabsdet, logabsdet_ref, atol=1e-5, rtol=1e-5)
