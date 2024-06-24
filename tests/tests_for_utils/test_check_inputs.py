"""
Test suite for the utility models in the :mod:`chemotools.utils.check_inputs` module.

"""

### Imports ###

from typing import Optional, Tuple, Union

import numpy as np
import pytest

from chemotools.utils.check_inputs import check_weights

### Test Suite ###


@pytest.mark.parametrize(
    "weights, expected_result",
    [
        (  # Number 0 (no weights; for all)
            None,
            (None, True),
        ),
        (  # Number 1 (valid 1D-weights; for all)
            np.array([1.0, 2.0, 3.0]),
            (np.array([[1.0, 2.0, 3.0]]), True),
        ),
        (  # Number 2 (valid 2D-weights; for all)
            np.array([[1.0, 2.0, 3.0]]),
            (np.array([[1.0, 2.0, 3.0]]), True),
        ),
        (  # Number 3 (valid 2D-weights; individual)
            np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]),
            (np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]), False),
        ),
        (  # Number 4 (invalid 1D-weights with wrong column number; for all)
            np.array([1.0, 2.0, 3.0, 4.0]),
            ValueError("Weights must have 3 columns, but they have"),
        ),
        (  # Number 5 (invalid 1D-weights with wrong column number; for all)
            np.array([1.0, 2.0]),
            ValueError("Weights must have 3 columns, but they have"),
        ),
        (  # Number 6 (invalid 2D-weights with wrong column number; for all)
            np.array([[1.0, 2.0, 3.0, 4.0]]),
            ValueError("Weights must have 3 columns, but they have"),
        ),
        (  # Number 7 (invalid 2D-weights with wrong column number; for all)
            np.array([[1.0, 2.0]]),
            ValueError("Weights must have 3 columns, but they have"),
        ),
        (  # Number 8 (invalid 2D-weights with wrong row number; individual)
            np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            ValueError("Weights must have either 1 or 3 rows, but they have"),
        ),
        (  # Number 9 (invalid 2D-weights with wrong row number; individual)
            np.array(
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]
            ),
            ValueError("Weights must have either 1 or 3 rows, but they have"),
        ),
        (
            # Number 10 (invalid 2D-weights with wrong row and column number;
            # individual)
            np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]),
            ValueError("Weights must have either 1 or 3 rows, but they have"),
        ),
        (
            # Number 10 (invalid 2D-weights with wrong row and column number;
            # individual)
            np.array([[1.0, 2.0], [3.0, 4.0]]),
            ValueError("Weights must have either 1 or 3 rows, but they have"),
        ),
        (
            # Number 11 (invalid 2D-weights with wrong row and column number;
            # individual)
            np.array(
                [
                    [1.0, 2.0, 3.0, 4.0],
                    [5.0, 6.0, 7.0, 8.0],
                    [9.0, 10.0, 11.0, 12.0],
                    [13.0, 14.0, 15.0, 16.0],
                ]
            ),
            ValueError("Weights must have either 1 or 3 rows, but they have"),
        ),
        (
            # Number 12 (invalid 2D-weights with wrong row and column number;
            # individual)
            np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]),
            ValueError("Weights must have either 1 or 3 rows, but they have"),
        ),
        (  # Number 13 (invalid 1D-weights with negative entry; for all)
            np.array([1.0, 2.0, -1_000.0]),
            ValueError("Weights may not be negative, but"),
        ),
        (  # Number 14 (invalid 2D-weights with negative entry; for all)
            np.array([[1.0, 2.0, -1_000.0]]),
            ValueError("Weights may not be negative, but"),
        ),
        (  # Number 15 (invalid 2D-weights with negative entry; individual)
            np.array([[1.0, 2.0, 3.0], [4.0, 5.0, -1_000.0], [7.0, 8.0, 9.0]]),
            ValueError("Weights may not be negative, but"),
        ),
        (  # Number 16 (invalid 1D-weights with NaN entry; for all)
            np.array([1.0, 2.0, np.nan]),
            ValueError("Input contains NaN"),
        ),
        (  # Number 17 (invalid 2D-weights with NaN entry; for all)
            np.array([[1.0, 2.0, np.nan]]),
            ValueError("Input contains NaN"),
        ),
        (  # Number 18 (invalid 2D-weights with NaN entry; individual)
            np.array([[1.0, 2.0, 3.0], [4.0, 5.0, np.nan], [7.0, 8.0, 9.0]]),
            ValueError("Input contains NaN"),
        ),
        (  # Number 19 (invalid 1D-weights with inf entry; for all)
            np.array([1.0, 2.0, np.inf]),
            ValueError("Input contains infinity or a value too large"),
        ),
        (  # Number 20 (invalid 2D-weights with inf entry; for all)
            np.array([[1.0, 2.0, np.inf]]),
            ValueError("Input contains infinity or a value too large"),
        ),
        (  # Number 21 (invalid 2D-weights with inf entry; individual)
            np.array([[1.0, 2.0, 3.0], [4.0, 5.0, np.inf], [7.0, 8.0, 9.0]]),
            ValueError("Input contains infinity or a value too large"),
        ),
        (  # Number 22 (invalid 1D-weights with -inf entry; for all)
            np.array([1.0, 2.0, -np.inf]),
            ValueError("Input contains infinity or a value too large"),
        ),
        (  # Number 23 (invalid 2D-weights with -inf entry; for all)
            np.array([[1.0, 2.0, -np.inf]]),
            ValueError("Input contains infinity or a value too large"),
        ),
        (  # Number 24 (invalid 2D-weights with -inf entry; individual)
            np.array([[1.0, 2.0, 3.0], [4.0, 5.0, -np.inf], [7.0, 8.0, 9.0]]),
            ValueError("Input contains infinity or a value too large"),
        ),
        (  # Number 25 (invalid 1D-weights with all zero entries; for all)
            np.array([0.0, 0.0, 0.0]),
            ValueError("At least one weights needs to be > 0, but"),
        ),
        (  # Number 26 (invalid 2D-weights with all zero entries; for all)
            np.array([[0.0, 0.0, 0.0]]),
            ValueError("At least one weights needs to be > 0, but"),
        ),
        (  # Number 27 (invalid 2D-weights with all zero entries; individual)
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
            ValueError("At least one weights needs to be > 0, but"),
        ),
    ],
)
def test_weight_checks(
    weights: Optional[np.ndarray],
    expected_result: Union[Tuple[Optional[np.ndarray], bool], Exception],
) -> None:
    """
    Tests the function :func:`chemotools.utils.check_inputs.check_weights` for different
    valid and invalid input combinations.

    """

    # the size of the matrix against which the weights are checked is set
    n_samples, n_features = 3, 3

    # if the expected output is an exception, the test is run in a context manager to
    # check if the respective exception is raised
    if isinstance(expected_result, Exception):
        error_catch_phrase = str(expected_result)
        with pytest.raises(
            type(expected_result),
            match=error_catch_phrase,
        ):
            check_weights(
                weights=weights,
                n_samples=n_samples,
                n_features=n_features,
            )

        return

    # otherwise, the output is compared to the expected output
    ref_weights, ref_same_weights_for_all = expected_result
    checked_weights, same_weights_for_all = check_weights(
        weights=weights,
        n_samples=n_samples,
        n_features=n_features,
    )

    # Case 1: the reference weights are None
    if ref_weights is None:
        assert checked_weights is None
        assert same_weights_for_all is ref_same_weights_for_all

        return

    # Case 2: the reference weights are an Array and the checked weights are as well
    if isinstance(ref_weights, np.ndarray) and isinstance(checked_weights, np.ndarray):
        assert np.array_equal(checked_weights, ref_weights)
        assert same_weights_for_all is ref_same_weights_for_all

        return

    raise AssertionError(
        "The weights could not be checked correctly due to a type mismatch."
    )
