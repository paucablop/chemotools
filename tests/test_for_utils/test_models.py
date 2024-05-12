"""
Test suite for the utility models in the :mod:`chemotools.utils.models` module.

"""

### Imports ###

from math import log
from typing import List, Tuple, Type, Union

import pytest

from chemotools.utils import models
from tests.test_for_utils.utils import ExpectedWhittakerSmoothLambda

### Type aliases ###

_RealNumeric = Union[float, int]
_LambdaValueNumeric = Union[_RealNumeric, Tuple[_RealNumeric, _RealNumeric]]
_LambdaValueNumericOrFlawed = Union[_LambdaValueNumeric, str]
_WhittakerMethod = Union[str, models.WhittakerSmoothMethods]
_WhittakerMethodSequence = List[_WhittakerMethod]
_LambdaTestCombination = Tuple[
    _LambdaValueNumericOrFlawed,
    _WhittakerMethodSequence,
    Union[ExpectedWhittakerSmoothLambda, Type[Exception]],
]

### Constants ###

_NAN: float = float("nan")
_FIXED_WHITTAKER_METHODS: _WhittakerMethodSequence = [
    "fixed",
    models.WhittakerSmoothMethods.FIXED,
]
_LOGML_WHITTAKER_METHODS: _WhittakerMethodSequence = [
    "logml",
    models.WhittakerSmoothMethods.LOGML,
]
# NOTE: "aauto" is not a typo, but helps to not confuse it with "all"
_aauto_whittaker_methods: _WhittakerMethodSequence = _LOGML_WHITTAKER_METHODS + []
_all_whittaker_methods: _WhittakerMethodSequence = (
    _FIXED_WHITTAKER_METHODS + _aauto_whittaker_methods
)


### Test Suite ###


@pytest.mark.parametrize(
    "combination",
    [
        (  # Number 0 (fixed float; fixed method)
            100.0,
            _FIXED_WHITTAKER_METHODS,
            ExpectedWhittakerSmoothLambda(
                fixed_lambda=100.0,
                auto_bounds=(_NAN, _NAN),
                fit_auto=False,
                method_used=models.WhittakerSmoothMethods.FIXED,
                log_auto_bounds=(_NAN, _NAN),
            ),
        ),
        (  # Number 1 (fixed integer; fixed method)
            100,
            _FIXED_WHITTAKER_METHODS,
            ExpectedWhittakerSmoothLambda(
                fixed_lambda=100.0,
                auto_bounds=(_NAN, _NAN),
                fit_auto=False,
                method_used=models.WhittakerSmoothMethods.FIXED,
                log_auto_bounds=(_NAN, _NAN),
            ),
        ),
        (  # Number 2 (coinciding floats; fixed method)
            (100.0, 100.0),
            _FIXED_WHITTAKER_METHODS,
            ExpectedWhittakerSmoothLambda(
                fixed_lambda=100.0,
                auto_bounds=(_NAN, _NAN),
                fit_auto=False,
                method_used=models.WhittakerSmoothMethods.FIXED,
                log_auto_bounds=(_NAN, _NAN),
            ),
        ),
        (  # Number 3 (coinciding integers; fixed method)
            (100, 100),
            _FIXED_WHITTAKER_METHODS,
            ExpectedWhittakerSmoothLambda(
                fixed_lambda=100.0,
                auto_bounds=(_NAN, _NAN),
                fit_auto=False,
                method_used=models.WhittakerSmoothMethods.FIXED,
                log_auto_bounds=(_NAN, _NAN),
            ),
        ),
        (  # Number 4 (virtually coinciding floats; fixed method)
            (100.0, 100.000001),
            _FIXED_WHITTAKER_METHODS,
            ExpectedWhittakerSmoothLambda(
                fixed_lambda=100.000001,
                auto_bounds=(_NAN, _NAN),
                fit_auto=False,
                method_used=models.WhittakerSmoothMethods.FIXED,
                log_auto_bounds=(_NAN, _NAN),
            ),
        ),
        (  # Number 5 (virtually coinciding floats; automated methods)
            (100.0, 100.000001),
            _aauto_whittaker_methods,
            ExpectedWhittakerSmoothLambda(
                fixed_lambda=100.000001,
                auto_bounds=(_NAN, _NAN),
                fit_auto=False,
                method_used=models.WhittakerSmoothMethods.FIXED,
                log_auto_bounds=(_NAN, _NAN),
            ),
        ),
        (  # Number 6 (flipped virtually coinciding floats; fixed method)
            (100.000001, 100.0),
            _FIXED_WHITTAKER_METHODS,
            ExpectedWhittakerSmoothLambda(
                fixed_lambda=100.000001,
                auto_bounds=(_NAN, _NAN),
                fit_auto=False,
                method_used=models.WhittakerSmoothMethods.FIXED,
                log_auto_bounds=(_NAN, _NAN),
            ),
        ),
        (  # Number 7 (flipped virtually coinciding floats; automated methods)
            (100.000001, 100.0),
            _aauto_whittaker_methods,
            ExpectedWhittakerSmoothLambda(
                fixed_lambda=100.000001,
                auto_bounds=(_NAN, _NAN),
                fit_auto=False,
                method_used=models.WhittakerSmoothMethods.FIXED,
                log_auto_bounds=(_NAN, _NAN),
            ),
        ),
        (  # Number 8 (search space floats; logml method)
            (100.0, 10_000.0),
            _LOGML_WHITTAKER_METHODS,
            ExpectedWhittakerSmoothLambda(
                fixed_lambda=_NAN,
                auto_bounds=(100.0, 10_000.0),
                fit_auto=True,
                method_used=models.WhittakerSmoothMethods.LOGML,
                log_auto_bounds=(log(100.0), log(10_000.0)),
            ),
        ),
        (  # Number 9 (search space integers; logml method)
            (100, 10_000),
            _LOGML_WHITTAKER_METHODS,
            ExpectedWhittakerSmoothLambda(
                fixed_lambda=_NAN,
                auto_bounds=(100.0, 10_000.0),
                fit_auto=True,
                method_used=models.WhittakerSmoothMethods.LOGML,
                log_auto_bounds=(log(100.0), log(10_000.0)),
            ),
        ),
        (  # Number 10 (flipped search space floats; logml method)
            (10_000.0, 100.0),
            _LOGML_WHITTAKER_METHODS,
            ExpectedWhittakerSmoothLambda(
                fixed_lambda=_NAN,
                auto_bounds=(100.0, 10_000.0),
                fit_auto=True,
                method_used=models.WhittakerSmoothMethods.LOGML,
                log_auto_bounds=(log(100.0), log(10_000.0)),
            ),
        ),
        (  # Number 11 (flipped search space integers; logml method)
            (10_000, 100),
            _LOGML_WHITTAKER_METHODS,
            ExpectedWhittakerSmoothLambda(
                fixed_lambda=_NAN,
                auto_bounds=(100.0, 10_000.0),
                fit_auto=True,
                method_used=models.WhittakerSmoothMethods.LOGML,
                log_auto_bounds=(log(100.0), log(10_000.0)),
            ),
        ),
        (  # Number 12 (fixed zero float; fixed method)
            0.0,
            _FIXED_WHITTAKER_METHODS,
            ValueError,
        ),
        (  # Number 13 (fixed zero integer; fixed method)
            0,
            _FIXED_WHITTAKER_METHODS,
            ValueError,
        ),
        (  # Number 14 (search space floats; fixed method)
            (100.0, 10_000.0),
            _FIXED_WHITTAKER_METHODS,
            ValueError,
        ),
        (
            # Number 15 (search space integers; fixed method)
            (100, 10_000),
            _FIXED_WHITTAKER_METHODS,
            ValueError,
        ),
        (  # Number 16 (fixed float; automated method)
            100.0,
            _aauto_whittaker_methods,
            ValueError,
        ),
        (
            # Number 17 (fixed integer; automated method)
            100,
            _aauto_whittaker_methods,
            ValueError,
        ),
        (  # Number 18 (search space floats with zero; all methods)
            (0.0, 100.0),
            _all_whittaker_methods,
            ValueError,
        ),
        (  # Number 19 (search space integers with zero; all methods)
            (0, 100),
            _all_whittaker_methods,
            ValueError,
        ),
        (  # Number 20 (flipped search space floats with zero; all methods)
            (100.0, 0.0),
            _all_whittaker_methods,
            ValueError,
        ),
        (  # Number 21 (flipped search space integer with zero; all methods)
            (100, 0),
            _all_whittaker_methods,
            ValueError,
        ),
        (  # Number 22 (all float zeros; all methods)
            (0.0, 0.0),
            _all_whittaker_methods,
            ValueError,
        ),
        (  # Number 23 (all float integers; all methods)
            (0, 0),
            _all_whittaker_methods,
            ValueError,
        ),
        (  # Number 24 (wrong type; all methods)
            "error",
            _all_whittaker_methods,
            TypeError,
        ),
        (  # Number 25 (fixed float; wrong method)
            100.0,
            "error",
            ValueError,
        ),
        (  # Number 26 (fixed integer; wrong method)
            100,
            "error",
            ValueError,
        ),
    ],
)
def test_whittaker_smooth_lambda_model(combination: _LambdaTestCombination) -> None:
    """
    Tests the class :class:`WhittakerSmoothLambda` for the correct behavior of its
    ``__post_init__`` method.

    The ``combination`` parameter defines

    - the lambda value(s) to be used,
    - the method(s) to be used, and
    - the expected result of the instantiation (will be an exception if the input
        should be considered invalid by the dataclass).

    """

    # the combination is unpacked
    lambda_value, methods, expected_result = combination

    # if the expected result is an exception, it is tested whether the correct exception
    # is raised
    if not isinstance(expected_result, ExpectedWhittakerSmoothLambda):
        for meth in methods:
            with pytest.raises(expected_result):
                models.WhittakerSmoothLambda(
                    bounds=lambda_value,  # type: ignore
                    method=meth,  # type: ignore
                )

        return

    # if the expected result is a valid result, the class is instantiated and the
    # generated object is compared to the expected result
    for meth in methods:
        lambda_model = models.WhittakerSmoothLambda(
            bounds=lambda_value,  # type: ignore
            method=meth,  # type: ignore
        )

        expected_result.assert_is_equal_to(other=lambda_model)
