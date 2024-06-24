"""
Test suite for the utility models in the :mod:`chemotools.utils.models` module.

"""

### Imports ###

from math import log
from typing import List, Tuple, Union

import pytest

from chemotools.utils import _models
from tests.tests_for_utils.utils_models import ExpectedWhittakerSmoothLambda

### Type aliases ###

_RealNumeric = Union[float, int]
_LambdaValueNumeric = Union[_RealNumeric, Tuple[_RealNumeric, _RealNumeric]]
_LambdaValueNumericOrFlawed = Union[_LambdaValueNumeric, str]
_WhittakerMethod = Union[str, _models.WhittakerSmoothMethods]
_WhittakerMethodSequence = List[_WhittakerMethod]

### Constants ###

_NAN: float = float("nan")
_FIXED_WHITTAKER_METHODS: _WhittakerMethodSequence = [
    "fixed",
    _models.WhittakerSmoothMethods.FIXED,
]
_LOGML_WHITTAKER_METHODS: _WhittakerMethodSequence = [
    "logml",
    _models.WhittakerSmoothMethods.LOGML,
]
# NOTE: "aauto" is not a typo, but helps to not confuse it with "all"
_aauto_whittaker_methods: _WhittakerMethodSequence = _LOGML_WHITTAKER_METHODS + []
_all_whittaker_methods: _WhittakerMethodSequence = (
    _FIXED_WHITTAKER_METHODS + _aauto_whittaker_methods
)


### Test Suite ###


@pytest.mark.parametrize(
    "lam, methods, expected",
    [
        (  # Number 0 (fixed float; fixed method)
            100.0,
            _FIXED_WHITTAKER_METHODS,
            ExpectedWhittakerSmoothLambda(
                fixed_lambda=100.0,
                auto_bounds=(_NAN, _NAN),
                fit_auto=False,
                method_used=_models.WhittakerSmoothMethods.FIXED,
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
                method_used=_models.WhittakerSmoothMethods.FIXED,
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
                method_used=_models.WhittakerSmoothMethods.FIXED,
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
                method_used=_models.WhittakerSmoothMethods.FIXED,
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
                method_used=_models.WhittakerSmoothMethods.FIXED,
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
                method_used=_models.WhittakerSmoothMethods.FIXED,
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
                method_used=_models.WhittakerSmoothMethods.FIXED,
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
                method_used=_models.WhittakerSmoothMethods.FIXED,
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
                method_used=_models.WhittakerSmoothMethods.LOGML,
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
                method_used=_models.WhittakerSmoothMethods.LOGML,
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
                method_used=_models.WhittakerSmoothMethods.LOGML,
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
                method_used=_models.WhittakerSmoothMethods.LOGML,
                log_auto_bounds=(log(100.0), log(10_000.0)),
            ),
        ),
        (  # Number 12 (fixed zero float; fixed method)
            0.0,
            _FIXED_WHITTAKER_METHODS,
            ValueError("has to be greater than or equal to the zero tolerance"),
        ),
        (  # Number 13 (fixed zero integer; fixed method)
            0,
            _FIXED_WHITTAKER_METHODS,
            ValueError("has to be greater than or equal to the zero tolerance"),
        ),
        (  # Number 14 (search space floats; fixed method)
            (100.0, 10_000.0),
            _FIXED_WHITTAKER_METHODS,
            ValueError("for the penalty weight lambda are a search space"),
        ),
        (
            # Number 15 (search space integers; fixed method)
            (100, 10_000),
            _FIXED_WHITTAKER_METHODS,
            ValueError("for the penalty weight lambda are a search space"),
        ),
        (  # Number 16 (fixed float; automated method)
            100.0,
            _aauto_whittaker_methods,
            ValueError("was selected for a fixed penalty weight"),
        ),
        (
            # Number 17 (fixed integer; automated method)
            100,
            _aauto_whittaker_methods,
            ValueError("was selected for a fixed penalty weight"),
        ),
        (  # Number 18 (search space floats with zero; all methods)
            (0.0, 100.0),
            _all_whittaker_methods,
            ValueError("have to be greater than or equal to the zero tolerance"),
        ),
        (  # Number 19 (search space integers with zero; all methods)
            (0, 100),
            _all_whittaker_methods,
            ValueError("have to be greater than or equal to the zero tolerance"),
        ),
        (  # Number 20 (flipped search space floats with zero; all methods)
            (100.0, 0.0),
            _all_whittaker_methods,
            ValueError("have to be greater than or equal to the zero tolerance"),
        ),
        (  # Number 21 (flipped search space integer with zero; all methods)
            (100, 0),
            _all_whittaker_methods,
            ValueError("have to be greater than or equal to the zero tolerance"),
        ),
        (  # Number 22 (all float zeros; all methods)
            (0.0, 0.0),
            _all_whittaker_methods,
            ValueError("have to be greater than or equal to the zero tolerance"),
        ),
        (  # Number 23 (all float integers; all methods)
            (0, 0),
            _all_whittaker_methods,
            ValueError("have to be greater than or equal to the zero tolerance"),
        ),
        (  # Number 24 (wrong type; all methods)
            "error",
            _all_whittaker_methods,
            TypeError("have to be either a scalar or a tuple of two values"),
        ),
        (  # Number 25 (fixed float; wrong method)
            100.0,
            "error",
            ValueError("is not valid. Please choose one of the following"),
        ),
        (  # Number 26 (fixed integer; wrong method)
            100,
            "error",
            ValueError("is not valid. Please choose one of the following"),
        ),
    ],
)
def test_whittaker_smooth_lambda_model(
    lam: _LambdaValueNumericOrFlawed,
    methods: _WhittakerMethodSequence,
    expected: Union[ExpectedWhittakerSmoothLambda, Exception],
) -> None:
    """
    Tests the class :class:`WhittakerSmoothLambda` for the correct behavior of its
    ``__post_init__`` method.

    """

    # if the expected result is an exception, it is tested whether the correct exception
    # is raised
    if isinstance(expected, Exception):
        error_catch_phrase = str(expected)
        for meth in methods:
            with pytest.raises(
                type(expected),
                match=error_catch_phrase,
            ):
                _models.WhittakerSmoothLambda(
                    bounds=lam,  # type: ignore
                    method=meth,  # type: ignore
                )

        return

    # if the expected result is a valid result, the class is instantiated and the
    # generated object is compared to the expected result
    for meth in methods:
        lambda_model = _models.WhittakerSmoothLambda(
            bounds=lam,  # type: ignore
            method=meth,  # type: ignore
        )

        expected.assert_is_equal_to(other=lambda_model)
