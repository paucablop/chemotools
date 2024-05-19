"""
Test suite for the utility functions in the :mod:`chemotools.utils.whittaker_base`
module.

"""

### Imports ###

from math import log
from typing import Any, Tuple, Type, Union

import numpy as np
import pytest

from chemotools.utils import models
from chemotools.utils.whittaker_base.auto_lambda.shared import get_smooth_wrss
from chemotools.utils.whittaker_base.initialisation import (
    get_checked_lambda,
    get_penalty_log_pseudo_det,
)
from chemotools.utils.whittaker_base.main import WhittakerLikeSolver
from chemotools.utils.whittaker_base.misc import get_weight_generator
from chemotools.utils.whittaker_base.solvers import solve_normal_equations
from tests.fixtures import noise_level_whittaker_auto_lambda  # noqa: F401
from tests.fixtures import spectrum_whittaker_auto_lambda  # noqa: F401
from tests.test_for_utils.utils_funcs import (
    find_whittaker_smooth_opt_lambda_log_marginal_likelihood,
)
from tests.test_for_utils.utils_models import ExpectedWhittakerSmoothLambda

### Type Aliases ###

_RealNumeric = Union[float, int]
_WhittakerMethod = Union[str, models.WhittakerSmoothMethods]
_LambdaSpecs = Union[_RealNumeric, Tuple[_RealNumeric, _RealNumeric, _WhittakerMethod]]
_LambdaSpecsOrFlawed = Union[_LambdaSpecs, str]

### Constants ###

_NAN: float = float("nan")

### Test Suite ###


@pytest.mark.parametrize(
    "combination",
    [
        (  # Number 0 (fixed float)
            100.0,
            ExpectedWhittakerSmoothLambda(
                fixed_lambda=100.0,
                auto_bounds=(_NAN, _NAN),
                fit_auto=False,
                method_used=models.WhittakerSmoothMethods.FIXED,
                log_auto_bounds=(_NAN, _NAN),
            ),
        ),
        (  # Number 1 (fixed integer)
            100,
            ExpectedWhittakerSmoothLambda(
                fixed_lambda=100.0,
                auto_bounds=(_NAN, _NAN),
                fit_auto=False,
                method_used=models.WhittakerSmoothMethods.FIXED,
                log_auto_bounds=(_NAN, _NAN),
            ),
        ),
        (  # Number 2 (float search space, log marginal likelihood method enum)
            (100.0, 10_000.0, models.WhittakerSmoothMethods.LOGML),
            ExpectedWhittakerSmoothLambda(
                fixed_lambda=_NAN,
                auto_bounds=(100.0, 10_000.0),
                fit_auto=True,
                method_used=models.WhittakerSmoothMethods.LOGML,
                log_auto_bounds=(log(100.0), log(10_000.0)),
            ),
        ),
        (  # Number 3 (float search space, log marginal likelihood method string)
            (100.0, 10_000.0, "logml"),
            ExpectedWhittakerSmoothLambda(
                fixed_lambda=_NAN,
                auto_bounds=(100.0, 10_000.0),
                fit_auto=True,
                method_used=models.WhittakerSmoothMethods.LOGML,
                log_auto_bounds=(log(100.0), log(10_000.0)),
            ),
        ),
        (  # Number 4 (integer search space, log marginal likelihood method enum)
            (100, 10_000, models.WhittakerSmoothMethods.LOGML),
            ExpectedWhittakerSmoothLambda(
                fixed_lambda=_NAN,
                auto_bounds=(100.0, 10_000.0),
                fit_auto=True,
                method_used=models.WhittakerSmoothMethods.LOGML,
                log_auto_bounds=(log(100.0), log(10_000.0)),
            ),
        ),
        (  # Number 5 (integer search space, log marginal likelihood method string)
            (100, 10_000, "logml"),
            ExpectedWhittakerSmoothLambda(
                fixed_lambda=_NAN,
                auto_bounds=(100.0, 10_000.0),
                fit_auto=True,
                method_used=models.WhittakerSmoothMethods.LOGML,
                log_auto_bounds=(log(100.0), log(10_000.0)),
            ),
        ),
        (  # Number 6 (dataclass float specification; fixed method)
            models.WhittakerSmoothLambda(
                bounds=100.0,
                method=models.WhittakerSmoothMethods.FIXED,
            ),
            ExpectedWhittakerSmoothLambda(
                fixed_lambda=100.0,
                auto_bounds=(_NAN, _NAN),
                fit_auto=False,
                method_used=models.WhittakerSmoothMethods.FIXED,
                log_auto_bounds=(_NAN, _NAN),
            ),
        ),
        (  # Number 7 (dataclass integer specification; fixed method)
            models.WhittakerSmoothLambda(
                bounds=100,
                method=models.WhittakerSmoothMethods.FIXED,
            ),
            ExpectedWhittakerSmoothLambda(
                fixed_lambda=100.0,
                auto_bounds=(_NAN, _NAN),
                fit_auto=False,
                method_used=models.WhittakerSmoothMethods.FIXED,
                log_auto_bounds=(_NAN, _NAN),
            ),
        ),
        (  # Number 8 (dataclass float specification; log marginal likelihood method)
            models.WhittakerSmoothLambda(
                bounds=(100.0, 10_000.0),
                method=models.WhittakerSmoothMethods.LOGML,
            ),
            ExpectedWhittakerSmoothLambda(
                fixed_lambda=_NAN,
                auto_bounds=(100.0, 10_000.0),
                fit_auto=True,
                method_used=models.WhittakerSmoothMethods.LOGML,
                log_auto_bounds=(log(100.0), log(10_000.0)),
            ),
        ),
        (  # Number 9 (dataclass integer specification; log marginal likelihood method)
            models.WhittakerSmoothLambda(
                bounds=(100, 10_000),
                method=models.WhittakerSmoothMethods.LOGML,
            ),
            ExpectedWhittakerSmoothLambda(
                fixed_lambda=_NAN,
                auto_bounds=(100.0, 10_000.0),
                fit_auto=True,
                method_used=models.WhittakerSmoothMethods.LOGML,
                log_auto_bounds=(log(100.0), log(10_000.0)),
            ),
        ),
        (  # Number 10 (wrong length tuple)
            (100.0, 10_000.0),
            ValueError,
        ),
        (  # Number 11 (wrong type)
            "error",
            TypeError,
        ),
    ],
)
def test_get_checked_lambda(
    combination: Tuple[
        _LambdaSpecsOrFlawed, Union[ExpectedWhittakerSmoothLambda, Type[Exception]]
    ]
) -> None:
    """
    Tests the function that casts a penalty weight lambda to the respective dataclass.

    The ``combination`` parameter defines

    - the lambda specification to be used and
    - the expected result (will be an exception if the input should be considered
        invalid by the function).

    """

    # the input parameters are unpacked
    lam, expected_result = combination

    # if the expected output is an exception, the test is run in a context manager
    if not isinstance(expected_result, ExpectedWhittakerSmoothLambda):
        with pytest.raises(expected_result):
            get_checked_lambda(lam=lam)

        return

    # otherwise, the output dataclass is compared to the expected output
    lambda_model = get_checked_lambda(lam=lam)
    if isinstance(lambda_model, models.WhittakerSmoothLambda):
        expected_result.assert_is_equal_to(other=lambda_model)

        return

    raise AssertionError(
        "The lambda value could not be checked correctly since the returned value is "
        "not an instance of the class 'WhittakerSmoothLambda'."
    )


@pytest.mark.parametrize(
    "combination",
    [
        (None, 1.0),  # Number 0
        (  # Number 1
            np.ones(shape=(10, 1_000), dtype=np.float64),
            np.ones(shape=(1_000), dtype=np.float64),
        ),
        (  # Number 2
            np.ones(shape=(5, 1_000), dtype=np.float64),
            IndexError,
        ),
        (  # Number 3
            np.ones(shape=(1_000), dtype=np.float64),
            ValueError,
        ),
        (  # Number 4
            np.ones(shape=(1, 5, 1_000), dtype=np.float64),
            ValueError,
        ),
        ("error", TypeError),  # Number 5
    ],
)
def test_weight_generator(
    combination: Tuple[Any, Union[np.ndarray, float, Type[Exception]]]
) -> None:
    """
    Tests the weight generator.

    The ``combination`` parameter defines

    - the weights to be used and
    - the expected output at each iteration (will be an exception if the input should
        be considered invalid by the function).

    """

    # the input parameters are unpacked
    weights, expected_output = combination

    # the number of series is defined
    n_series = 10

    # if the expected output is an exception, the test is run in a context manager
    if not isinstance(expected_output, (np.ndarray, float, int)):
        with pytest.raises(expected_output):
            for _ in get_weight_generator(w=weights, n_series=n_series):
                pass

        return

    # otherwise, the output is compared to the expected output
    # Case 1: the expected output is a scalar
    if isinstance(expected_output, (float, int)):
        for w in get_weight_generator(w=weights, n_series=n_series):
            assert isinstance(w, (float, int))
            assert w == expected_output

        return

    # Case 2: the expected output is an array
    for w in get_weight_generator(w=weights, n_series=n_series):
        assert isinstance(w, np.ndarray)
        assert np.array_equal(w, expected_output)


@pytest.mark.parametrize("combination", [(True, 244_9755_000.0), (False, 490_000.0)])
def test_smooth_wrss(combination: Tuple[bool, float]) -> None:
    """
    Tests the weighted residual sum of squares calculation.

    The ``combination`` parameter defines

    - whether weights are used (``True``) or not (``False``) and
    - the expected weighted residual sum of squares.

    """

    # the input parameters are unpacked
    with_weights, wrss_expected = combination

    # two series are generated where the difference between the elements is 7.0
    np.random.seed(42)
    n_data = 10_000
    a_signs = np.random.choice([-1.0, 1.0], size=(n_data,), replace=True)
    a_series = a_signs * 4.5
    b_series = (-1.0) * a_signs * 2.5

    # the weights are generated
    weights = (
        np.arange(start=0, stop=n_data, step=1.0, dtype=np.float64)
        if with_weights
        else 1.0
    )

    # the wrss is calculated ...
    wrss = get_smooth_wrss(b=a_series, b_smooth=b_series, w=weights)

    # ... and compared to the expected value with a very strict tolerance
    assert np.isclose(wrss, wrss_expected, atol=1e-13, rtol=0.0)


# TODO: due to ill-conditioning, this is highly limited in the differences and number
#       of data points; in the future, this should be tackled by QR-decomposition for
#       extra numerical stability
@pytest.mark.parametrize(
    "differences_and_n_data_from_to",
    [
        (1, 0, 2_000),
        (1, 2_001, 4_000),
        (1, 4_001, 6_000),
        (1, 6_001, 8_000),
        (1, 8_001, 10_000),
        (2, 0, 2_000),
        (2, 2_001, 4_000),
        (2, 4_001, 6_000),
        (2, 6_001, 8_000),
        (2, 8_001, 10_000),
    ],
)
def test_penalty_log_pseudo_det_can_compute(
    differences_and_n_data_from_to: Tuple[int, int, int]
) -> None:
    """
    Tests the log pseudo-determinant of the penalty matrix for all the difference orders
    and number of data points.

    """

    differences, n_data_from, n_data_to = differences_and_n_data_from_to
    for nd in range(max(differences + 1, n_data_from), n_data_to + 1):
        get_penalty_log_pseudo_det(n_data=nd, differences=differences, dtype=np.float64)


# TODO: this test will not 100% reflect reality as intended; in the future this should
#       be tested with the LAPACK function ``dgbcon`` to check the condition number;
#       right now, it is set to a number of data points that causes the intended
#       failure, but in the future, the condition number has to be used to detect
#       ill-conditioning
def test_penalty_log_pseudo_det_breaks_ill_conditioned() -> None:
    """
    Tests that the log pseudo-determinant of the penalty matrix breaks when the matrix
    is ill-conditioned.

    """

    # the difference order and number of data points are set so high that the matrix
    # becomes ill-conditioned
    n_data = 1_000
    differences = 10

    # the function is tested for breaking
    with pytest.raises(RuntimeError):
        get_penalty_log_pseudo_det(
            n_data=n_data, differences=differences, dtype=np.float64
        )

        return


# TODO: this test will not 100% reflect reality as intended; in the future this should
#       be tested with the LAPACK function ``dgbcon`` to check the condition number;
#       right now, the matrix is heavily altered to cause the intended failure, but in
#       the future, the condition number has to be used to detect ill-conditioning
@pytest.mark.parametrize("with_pentapy", [True, False])
def test_normal_condition_solve_breaks_ill_conditioned(with_pentapy: bool) -> None:
    """
    Tests that the normal condition solver breaks when the matrix is ill-conditioned.

    Note that the conditions for the solver to break will never be met in practice.

    """

    # if pentapy is not installed but required, the test is skipped
    if with_pentapy:
        try:
            import pentapy  # noqa: F401
        except ImportError:
            pytest.skip("Pentapy is not installed.")

    # a banded ill-conditioned matrix is created that has zeros on the diagonal
    n_data = 10_000
    differences = 2
    a_banded = np.ones(shape=(2 * differences + 1, n_data), dtype=np.float64)
    a_banded[differences, :] = 0.0

    # some further required variables are initialised
    lam = 1e100
    b_vect = np.ones(shape=(n_data,), dtype=np.float64)
    weights = 0.0

    # Test that the solver breaks
    with pytest.raises(RuntimeError):
        solve_normal_equations(
            lam=lam,
            differences=differences,
            l_and_u=(differences, differences),
            penalty_mat_banded=a_banded,
            b_weighted=b_vect,
            w=weights,
            pentapy_enabled=with_pentapy,
        )


def test_whittakerlike_issues_warning_difference_order_too_high() -> None:
    """
    Tests that the class :class:`WhittakerLikeSolver` issues a warning when the
    difference order is greater than 2.

    """

    with pytest.warns(UserWarning):
        whitt_base = WhittakerLikeSolver()
        whitt_base._setup_for_fit(
            n_data=500,
            differences=3,
            lam=models.WhittakerSmoothLambda(
                bounds=(100.0, 10_000.0),
                method=models.WhittakerSmoothMethods.LOGML,
            ),
            child_class_name="pytest_run",
        )

    return


# TODO: this can only go to differences 2 due to ill-conditioning; in the future, this
#       should be tackled by QR-decomposition for extra numerical stability
@pytest.mark.parametrize("same_weights_for_all", [True, False])
@pytest.mark.parametrize("differences", [1, 2])
def test_auto_lambda_log_marginal_likelihood_refuses_no_weights(
    differences: int,
    same_weights_for_all: bool,
) -> None:
    """
    Tests that the automatic lambda calculation using the log marginal likelihood method
    refuses to work with no weights.

    """

    # the smoother is initialised ...
    n_data = 500
    whitt_base = WhittakerLikeSolver()
    whitt_base._setup_for_fit(
        n_data=n_data,
        differences=differences,
        lam=models.WhittakerSmoothLambda(
            bounds=(100.0, 10_000.0),
            method=models.WhittakerSmoothMethods.LOGML,
        ),
        child_class_name="pytest_run",
    )

    # ... and the log marginal likelihood method is called without weights
    np.random.seed(42)
    X = np.random.rand(n_data)
    with pytest.raises(ValueError):
        whitt_base._whittaker_solve(
            X=X,
            w=None,
            use_same_w_for_all=same_weights_for_all,
        )


@pytest.mark.parametrize("with_zero_weights", [True, False])
@pytest.mark.parametrize("same_weights_for_all", [True, False])
@pytest.mark.parametrize("differences", [1, 2])
@pytest.mark.parametrize("n_series", [1, 5])
def test_auto_lambda_log_marginal_likelihood(
    spectrum_whittaker_auto_lambda: np.ndarray,  # noqa: F811
    noise_level_whittaker_auto_lambda: np.ndarray,  # noqa: F811
    n_series: int,
    differences: int,
    same_weights_for_all: bool,
    with_zero_weights: bool,
) -> None:
    """
    Tests the automatic lambda calculation using the log marginal likelihood method.

    Some of the noise standard deviations in the respective fixture are set to NaN which
    allows for two different ways of handling them:

    - with zero weights, which will set the weights of the NaN values to zero, or
    - interpolated weights, which will replace the NaN values with linearly interpolated
        values which cannot be zero.

    This has slightly different effects on the log marginal likelihood calculation.

    Everything is tested against a from-scratch implementation based on  SciPy to ensure
    that the test is decoupled from the actual implementation used in Chemotools.

    """

    # first of all, the Nan values in the noise level are handled
    noise_level = noise_level_whittaker_auto_lambda.copy()

    # Case 1: Zero weights
    if with_zero_weights:
        # this can be achieved by replacing the NaN-values with +inf
        noise_level = np.where(np.isnan(noise_level), np.inf, noise_level)

    # Case 2: Interpolated weights
    else:
        # the NaN-values are replaced by linearly interpolated values
        nan_flags = np.isnan(noise_level)
        noise_level[nan_flags] = np.interp(
            x=np.where(nan_flags)[0],
            xp=np.where(~nan_flags)[0],
            fp=noise_level[~nan_flags],
        )

    # then, the weights are computed as the square of the inverse noise level ...
    weights = (1.0 / np.square(noise_level))[np.newaxis, ::]
    # ... and stacked as many times as required
    weights = np.tile(weights, reps=(n_series, 1))

    # then, the spectrum is repeated as many times as required
    X = np.tile(spectrum_whittaker_auto_lambda[np.newaxis, ::], reps=(n_series, 1))

    # the smoothing is performed using the chemotools implementation
    lambda_bounds = (1e-15, 1e10)
    whitt_base = WhittakerLikeSolver()
    whitt_base._setup_for_fit(
        n_data=X.shape[1],
        differences=differences,
        lam=models.WhittakerSmoothLambda(
            bounds=lambda_bounds,
            method=models.WhittakerSmoothMethods.LOGML,
        ),
        child_class_name="pytest_run",
    )
    _, lambda_opts = whitt_base._whittaker_solve(
        X=X,
        w=weights,
        use_same_w_for_all=same_weights_for_all,
    )

    # the reference optimum lambda is found by a from-scratch implementation that relies
    # on dense matrices
    lambda_opt_ref, _, _ = find_whittaker_smooth_opt_lambda_log_marginal_likelihood(
        b_vect=X[0, ::],
        weight_vect=weights[0, ::],
        differences=differences,
        log_lambda_bounds=(log(lambda_bounds[0]), log(lambda_bounds[1])),
        n_opts=100,
    )

    # the results are compared with 1% relative tolerance
    for lam_opts in lambda_opts:
        assert np.isclose(lam_opts, lambda_opt_ref, rtol=1e-2)
