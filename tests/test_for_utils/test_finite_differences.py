"""
Test suite for the utility functions in the :mod:`chemotools.utils.finite_differences`
module.

"""

### Imports ###

from typing import List, Tuple

import numpy as np
import pytest

from chemotools.utils.finite_differences import (
    calc_forward_diff_kernel,
    gen_squ_fw_fin_diff_mat_cho_banded,
)
from tests.fixtures import reference_finite_differences  # noqa: F401
from tests.test_for_utils.utils import (
    conv_upper_cho_banded_storage_to_sparse,
    multiply_vect_with_squ_fw_fin_diff_orig_first,
    multiply_vect_with_squ_fw_fin_diff_transpose_first,
)

### Test Suite ###


def test_forward_diff_kernel(
    reference_finite_differences: List[Tuple[int, int, np.ndarray]]  # noqa: F811
) -> None:
    # each kernel is calculated and compared to the reference
    for differences, _, reference in reference_finite_differences:
        kernel = calc_forward_diff_kernel(differences=differences)

        assert kernel.size == reference.size, (
            f"Difference order {differences} with accuracy 1 expected kernel size "
            f"{reference.size} but got {kernel.size}"
        )
        assert np.allclose(kernel, reference, atol=1e-8), (
            f"Difference order {differences} with accuracy 1 expected kernel "
            f"{reference.tolist()} but got {kernel.tolist()}"
        )


@pytest.mark.parametrize(
    "n_add_size",
    list(range(0, 11)) + list(range(20, 101, 10)) + list(range(200, 1001, 100)),
)
@pytest.mark.parametrize("differences", list(range(1, 11)))
def test_squ_fw_fin_diff_mat_cho_banded_orig_first(
    differences: int, n_add_size: int
) -> None:
    """
    Tests the generation of the squared forward finite difference matrix ``D @ D.T``
    where ``D`` is the forward finite difference matrix.
    Here, the original matrix ``D`` and not its transpose is used first.

    It can be effectively tested by means of a convolution of the matrix with a vector
    after it was converted from the banded storage to a sparse matrix.

    """

    # first, the finite difference kernel is calculated
    kernel = calc_forward_diff_kernel(differences=differences)

    # then, the banded matrix D @ D.T is generated ...
    n_data = kernel.size + n_add_size
    ddt_banded = gen_squ_fw_fin_diff_mat_cho_banded(
        n_data=n_data,
        differences=differences,
        orig_first=True,
    )
    # ... and converted to a sparse matrix
    ddt_sparse = conv_upper_cho_banded_storage_to_sparse(ab=ddt_banded)

    # a random vector is created
    np.random.seed(42)
    vector = np.random.rand(n_add_size + 1)

    # this vector is multiplied with the matrix
    result = ddt_sparse @ vector

    # afterwards, the result is compared to the result of the convolution
    result_conv = multiply_vect_with_squ_fw_fin_diff_orig_first(
        differences=differences,
        kernel=kernel,
        vector=vector,
    )

    # the results are compared
    # NOTE: the following check has to be fairly strict when it comes to equivalence
    #       since the NumPy and Chemotools are basically doing the same under the hood
    assert np.allclose(result, result_conv, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize(
    "n_add_size",
    list(range(0, 11)) + list(range(20, 101, 10)) + list(range(200, 1001, 100)),
)
@pytest.mark.parametrize("differences", list(range(1, 11)))
def test_squ_fw_fin_diff_mat_cho_banded_transpose_first(
    differences: int, n_add_size: int
) -> None:
    """
    Tests the generation of the squared forward finite difference matrix ``D.T @ D``
    where ``D`` is the forward finite difference matrix.
    Here, the transpose matrix ``D.T`` and not the original matrix is used first.

    It can be effectively tested by means of a convolution of the matrix with a vector
    after it was converted from the banded storage to a sparse matrix.

    """

    # first, the finite difference kernel is calculated
    kernel = calc_forward_diff_kernel(differences=differences)

    # then, the banded matrix D.T @ D is generated ...
    n_data = kernel.size + n_add_size
    dtd_banded = gen_squ_fw_fin_diff_mat_cho_banded(
        n_data=n_data,
        differences=differences,
        orig_first=False,
    )
    # ... and converted to a sparse matrix
    dtd_sparse = conv_upper_cho_banded_storage_to_sparse(ab=dtd_banded)

    # a random vector is created
    np.random.seed(42)
    vector = np.random.rand(n_data)

    # this vector is multiplied with the matrix
    result = dtd_sparse @ vector

    # afterwards, the result is compared to the result of the convolution
    result_conv = multiply_vect_with_squ_fw_fin_diff_transpose_first(
        differences=differences,
        kernel=kernel,
        vector=vector,
    )

    # the results are compared
    # NOTE: the following check has to be fairly strict when it comes to equivalence
    #       since the NumPy and Chemotools are basically doing the same under the hood
    assert np.allclose(result, result_conv, atol=1e-10, rtol=1e-10)
