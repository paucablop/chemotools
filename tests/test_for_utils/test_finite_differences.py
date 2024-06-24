"""
Test suite for the utility functions in the :mod:`chemotools.utils.finite_differences`
module.

"""

### Imports ###

from typing import List, Optional

import numpy as np
import pytest

from chemotools.utils._finite_differences import (
    central_finite_difference_coefficients,
    estimate_noise_stddev,
    forward_finite_difference_kernel,
    squared_forward_difference_matrix_banded,
)
from tests.fixtures import noise_level_estimation_references  # noqa: F401
from tests.fixtures import noise_level_estimation_signal  # noqa: F401
from tests.fixtures import reference_finite_differences  # noqa: F401
from tests.test_for_utils.utils_funcs import (
    conv_upper_cho_banded_storage_to_sparse,
    multiply_vect_with_squared_forward_finite_differences_original_first,
    multiply_vect_with_squared_forward_finite_differences_transpose_first,
)
from tests.test_for_utils.utils_models import (
    NoiseEstimationReference,
    RefDifferenceKernel,
)

### Test Suite ###


# parametrizes the fixture ``reference_finite_differences``
@pytest.mark.parametrize("kind", ["forward"])
def test_forward_diff_kernel(
    reference_finite_differences: List[RefDifferenceKernel],  # noqa: F811
) -> None:
    """
    Tests the calculation of the forward finite difference kernel.

    """

    # each kernel is calculated and compared to the reference
    for kernel_reference in reference_finite_differences:
        kernel_chemotools = forward_finite_difference_kernel(
            differences=kernel_reference.differences
        )

        # first, the size of the kernel is checked ...
        assert kernel_chemotools.size == kernel_reference.size, (
            f"Difference order {kernel_reference.differences} with accuracy 1 - "
            f"Expected kernel size {kernel_reference.size} but got "
            f"{kernel_chemotools.size}"
        )
        # ...  followed by the comparison of the kernel itself
        assert np.allclose(kernel_chemotools, kernel_reference.kernel, atol=1e-8), (
            f"Difference order {kernel_reference.differences} with accuracy 1 - "
            f"Expected kernel {kernel_reference.kernel.tolist()} but got "
            f"{kernel_chemotools.tolist()}"
        )


# parametrizes the fixture ``reference_finite_differences``
@pytest.mark.parametrize("kind", ["central"])
def test_central_diff_kernel(
    reference_finite_differences: List[RefDifferenceKernel],  # noqa: F811
) -> None:
    """
    Tests the calculation of the central finite difference kernel.

    """

    # each kernel is calculated and compared to the reference
    for kernel_reference in reference_finite_differences:
        kernel_chemotools = central_finite_difference_coefficients(
            differences=kernel_reference.differences,
            accuracy=kernel_reference.accuracy,
        )

        # first, the size of the kernel is checked ...
        assert kernel_chemotools.size == kernel_reference.size, (
            f"Difference order {kernel_reference.differences} with accuracy "
            f"{kernel_reference.accuracy} - Expected kernel size "
            f"{kernel_reference.size} but got {kernel_chemotools.size}"
        )
        # ...  followed by the comparison of the kernel itself
        assert np.allclose(kernel_chemotools, kernel_reference.kernel, atol=1e-8), (
            f"Difference order {kernel_reference.differences} with accuracy "
            f"{kernel_reference.accuracy} - Expected kernel "
            f"{kernel_reference.kernel.tolist()} but got {kernel_chemotools.tolist()}"
        )


@pytest.mark.parametrize(
    "num_additional_values",
    list(range(0, 11)) + list(range(20, 101, 10)) + list(range(200, 1001, 100)),
)
@pytest.mark.parametrize("differences", list(range(1, 11)))
def test_squ_fw_fin_diff_mat_cho_banded_orig_first(
    differences: int,
    num_additional_values: int,
) -> None:
    """
    Tests the generation of the squared forward finite difference matrix ``D @ D.T``
    where ``D`` is the forward finite difference matrix.
    Here, the original matrix ``D`` and not its transpose is used first.

    It can be effectively tested by means of a convolution of the matrix with a vector
    after it was converted from the banded storage to a sparse matrix.

    """

    # first, the finite difference kernel is calculated
    kernel = forward_finite_difference_kernel(differences=differences)

    # then, the banded matrix D @ D.T is generated ...
    num_data = kernel.size + num_additional_values
    d_dot_dt_banded = squared_forward_difference_matrix_banded(
        num_data=num_data,
        differences=differences,
        original_first=True,
    )
    # ... and converted to a sparse matrix
    d_dot_dt_sparse = conv_upper_cho_banded_storage_to_sparse(ab=d_dot_dt_banded)

    # a random vector is created
    np.random.seed(42)
    vector = np.random.rand(num_additional_values + 1)

    # this vector is multiplied with the matrix
    result_chemotools = d_dot_dt_sparse @ vector

    # afterwards, the result is compared to the result of the convolution
    result_reference = (
        multiply_vect_with_squared_forward_finite_differences_original_first(
            differences=differences,
            kernel=kernel,
            vector=vector,
        )
    )

    # the results are compared
    # NOTE: the following check has to be fairly strict when it comes to equivalence
    #       since the NumPy and Chemotools are basically doing the same under the hood
    assert np.allclose(
        result_chemotools,
        result_reference,
        atol=1e-10,
        rtol=1e-10,
    )


@pytest.mark.parametrize(
    "num_additional_values",
    list(range(0, 11)) + list(range(20, 101, 10)) + list(range(200, 1001, 100)),
)
@pytest.mark.parametrize("differences", list(range(1, 11)))
def test_squ_fw_fin_diff_mat_cho_banded_transpose_first(
    differences: int,
    num_additional_values: int,
) -> None:
    """
    Tests the generation of the squared forward finite difference matrix ``D.T @ D``
    where ``D`` is the forward finite difference matrix.
    Here, the transpose matrix ``D.T`` and not the original matrix is used first.

    It can be effectively tested by means of a convolution of the matrix with a vector
    after it was converted from the banded storage to a sparse matrix.

    """

    # first, the finite difference kernel is calculated
    kernel = forward_finite_difference_kernel(differences=differences)

    # then, the banded matrix D.T @ D is generated ...
    num_data = kernel.size + num_additional_values
    dt_dot_d_banded = squared_forward_difference_matrix_banded(
        num_data=num_data,
        differences=differences,
        original_first=False,
    )
    # ... and converted to a sparse matrix
    dt_dot_d_sparse = conv_upper_cho_banded_storage_to_sparse(ab=dt_dot_d_banded)

    # a random vector is created
    np.random.seed(42)
    vector = np.random.rand(num_data)

    # this vector is multiplied with the matrix
    result_chemotools = dt_dot_d_sparse @ vector

    # afterwards, the result is compared to the result of the convolution
    result_reference = (
        multiply_vect_with_squared_forward_finite_differences_transpose_first(
            differences=differences,
            kernel=kernel,
            vector=vector,
        )
    )

    # the results are compared
    # NOTE: the following check has to be fairly strict when it comes to equivalence
    #       since the NumPy and Chemotools are basically doing the same under the hood
    assert np.allclose(result_chemotools, result_reference, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize(
    "series, differences, accuracy, window_size, stddev_power, stddev_min",
    [
        (  # Number 0 series is too small for difference kernel
            np.arange(start=0, stop=5),
            10,
            2,
            3,
            1,
            1e-10,
        ),
        (  # Number 1 series is too small for difference kernel
            np.arange(start=0, stop=5),
            10,
            2,
            None,
            1,
            1e-10,
        ),
        (  # Number 2 series is too small for window size
            np.arange(start=0, stop=5),
            1,
            2,
            11,
            1,
            1e-10,
        ),
        (  # Number 3 the difference order is 0
            np.arange(start=0, stop=10),
            0,
            2,
            3,
            1,
            1e-10,
        ),
        (  # Number 4 the difference order is negative
            np.arange(start=0, stop=10),
            -1,
            2,
            3,
            1,
            1e-10,
        ),
        (  # Number 5 the accuracy is odd
            np.arange(start=0, stop=10),
            2,
            3,
            3,
            1,
            1e-10,
        ),
        (  # Number 6 the accuracy is odd
            np.arange(start=0, stop=10),
            2,
            5,
            3,
            1,
            1e-10,
        ),
        (  # Number 7 the accuracy is 1
            np.arange(start=0, stop=10),
            2,
            1,
            3,
            1,
            1e-10,
        ),
        (  # Number 8 the accuracy is 0
            np.arange(start=0, stop=10),
            2,
            0,
            3,
            1,
            1e-10,
        ),
        (  # Number 9 the accuracy is negative
            np.arange(start=0, stop=10),
            2,
            -1,
            3,
            1,
            1e-10,
        ),
        (  # Number 10 the window size is even
            np.arange(start=0, stop=10),
            1,
            2,
            6,
            1,
            1e-10,
        ),
        (  # Number 11 the window size is 0
            np.arange(start=0, stop=10),
            1,
            2,
            0,
            1,
            1e-10,
        ),
        (  # Number 12 the window size is negative
            np.arange(start=0, stop=10),
            1,
            2,
            -1,
            1,
            1e-10,
        ),
        (  # Number 13 the power is -3
            np.arange(start=0, stop=10),
            1,
            2,
            3,
            -3,
            1e-10,
        ),
        (  # Number 14 the power is 3
            np.arange(start=0, stop=10),
            1,
            2,
            3,
            3,
            1e-10,
        ),
        (  # Number 15 the minimum standard deviation is zero
            np.arange(start=0, stop=5),
            1,
            2,
            3,
            1,
            0.0,
        ),
        (  # Number 16 the minimum standard deviation is negative
            np.arange(start=0, stop=5),
            1,
            2,
            3,
            1,
            -10.0,
        ),
    ],
)
def test_estimate_noise_stddev_invalid_input(
    series: np.ndarray,
    differences: int,
    accuracy: int,
    window_size: Optional[int],
    stddev_power: int,
    stddev_min: float,
) -> None:
    """
    Tests the input validation of the function :func:`estimate_noise_stddev`.

    The combinations of

    - the series length,
    - the difference order,
    - the accuracy,
    - the window size,
    - the power to which the noise level is raised, and
    - the minimum standard deviation

    are chosen such that the input is invalid.

    """

    with pytest.raises(ValueError):
        estimate_noise_stddev(
            series=series,
            differences=differences,
            differences_accuracy=accuracy,
            window_size=window_size,
            stddev_power=stddev_power,  # type: ignore
            stddev_min=stddev_min,
        )

    return


def test_noise_level_estimation(
    noise_level_estimation_signal: np.ndarray,  # noqa: F811
    noise_level_estimation_references: List[NoiseEstimationReference],  # noqa: F811
) -> None:
    """
    Tests the noise level estimation function :func:`estimate_noise_stddev`.

    The function is tested for all the reference noise levels.

    """

    for noise_level_reference in noise_level_estimation_references:
        # the noise level is estimated
        noise_level_chemotools = estimate_noise_stddev(
            series=noise_level_estimation_signal,
            differences=noise_level_reference.differences,
            differences_accuracy=noise_level_reference.accuracy,
            window_size=noise_level_reference.window_size,
            stddev_min=noise_level_reference.min_noise_level,
        )
        # then, the noise level itself is compared to the reference in a quite strict
        # way because both results were computed in the same way with the only
        # difference being that Chemotools uses Python and the reference uses
        # LibreOffice Calc
        assert np.allclose(
            noise_level_chemotools,
            noise_level_reference.noise_level,
            rtol=1e-12,
        ), (
            f"Original noise level differs from reference noise for differences "
            f"{noise_level_reference.differences} with accuracy "
            f"{noise_level_reference.accuracy} and window size "
            f"{noise_level_reference.window_size} given a minimum standard deviation "
            f"of {noise_level_reference.min_noise_level}."
        )

        # then, all the available powers to which the noise level can be raised are
        # compared to the reference
        for (
            stddev_power,
            raised_noise_level_ref,
        ) in noise_level_reference.raised_noise_levels.items():
            raised_noise_level = estimate_noise_stddev(
                series=noise_level_estimation_signal,
                differences=noise_level_reference.differences,
                differences_accuracy=noise_level_reference.accuracy,
                window_size=noise_level_reference.window_size,
                stddev_min=noise_level_reference.min_noise_level,
                stddev_power=stddev_power,
            )

            # again, the comparison is quite strict
            assert np.allclose(
                raised_noise_level,
                raised_noise_level_ref,
                atol=1e-12,
            ), (
                f"Raised noise level differs from reference noise for differences "
                f"{noise_level_reference.differences} with accuracy "
                f"{noise_level_reference.accuracy} and window size "
                f"{noise_level_reference.window_size} given a minimum standard "
                f"deviation of {noise_level_reference.min_noise_level} and a power of "
                f"{stddev_power}."
            )

    return
