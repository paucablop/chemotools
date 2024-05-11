"""
Test suite for the utility models in the :mod:`chemotools.utils.models` module.

"""

### Imports ###

from typing import List, Tuple, Type, Union

import numpy as np
import pytest

from chemotools.utils import models
from tests.test_for_utils.utils import ExpectedWhittakerSmoothLambda

### Type aliases ###

_Numeric = Union[float, int]
_LambdaValueNumeric = Union[_Numeric, Tuple[_Numeric, _Numeric]]
_LambdaValueNumericOrFlawed = Union[_LambdaValueNumeric, str]
_WhittakerMethod = Union[str, models.WhittakerSmoothMethods]
_WhittakerMethodSequence = List[_WhittakerMethod]
_ExpectedLambdaResult = Union[
    ExpectedWhittakerSmoothLambda,
    Type[ValueError],
    Type[TypeError],
]
_LambdaTestCombination = Tuple[
    _LambdaValueNumericOrFlawed,
    _WhittakerMethodSequence,
    _ExpectedLambdaResult,
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
        (  # Number 0
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
        (  # Number 1
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
        (  # Number 2
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
        (  # Number 3
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
        (  # Number 4
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
        (  # Number 5
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
        (  # Number 6
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
        (  # Number 7
            (100.0, 10_000.0),
            _LOGML_WHITTAKER_METHODS,
            ExpectedWhittakerSmoothLambda(
                fixed_lambda=_NAN,
                auto_bounds=(100.0, 10_000.0),
                fit_auto=True,
                method_used=models.WhittakerSmoothMethods.LOGML,
                log_auto_bounds=(np.log(100.0), np.log(10_000.0)),
            ),
        ),
        (  # Number 8
            (10_000.0, 100.0),
            _LOGML_WHITTAKER_METHODS,
            ExpectedWhittakerSmoothLambda(
                fixed_lambda=_NAN,
                auto_bounds=(100.0, 10_000.0),
                fit_auto=True,
                method_used=models.WhittakerSmoothMethods.LOGML,
                log_auto_bounds=(np.log(100.0), np.log(10_000.0)),
            ),
        ),
        (  # Number 9
            0.0,
            _FIXED_WHITTAKER_METHODS,
            ValueError,
        ),
        (  # Number 10
            (100.0, 10_000.0),
            _FIXED_WHITTAKER_METHODS,
            ValueError,
        ),
        (  # Number 11
            100.0,
            _aauto_whittaker_methods,
            ValueError,
        ),
        (  # Number 12
            (0.0, 100.0),
            _all_whittaker_methods,
            ValueError,
        ),
        (  # Number 13
            (100.0, 0.0),
            _all_whittaker_methods,
            ValueError,
        ),
        (  # Number 14
            (0.0, 0.0),
            _all_whittaker_methods,
            ValueError,
        ),
        (  # Number 15
            "error",
            _all_whittaker_methods,
            TypeError,
        ),
        (  # Number 16
            100.0,
            "error",
            ValueError,
        ),
    ],
)
def test_whittaker_smooth_lambda_model(combination: _LambdaTestCombination) -> None:
    """
    Tests the class :class:`WhittakerSmoothLambda` for the correct behavior of its
    ``__post_init__`` method.

    """

    # the combination is unpacked
    lambda_value, methods, expected_result = combination

    # if the expected result is an exception, it is tested whether the correct exception
    # is raised
    if not isinstance(expected_result, ExpectedWhittakerSmoothLambda):
        for meth in methods:
            with pytest.raises(expected_result):  # type: ignore
                models.WhittakerSmoothLambda(
                    bounds=lambda_value,  # type: ignore
                    method=meth,  # type: ignore
                )

        return

    # if the expected result is a valid result, the class is instantiated and the
    # attributes are tested
    for meth in methods:
        lambda_model = models.WhittakerSmoothLambda(
            bounds=lambda_value,  # type: ignore
            method=meth,  # type: ignore
        )

        expected_result.assert_is_equal_to(other=lambda_model)
