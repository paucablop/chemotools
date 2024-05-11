"""
This submodule contains the functions used for the optimization in the automated fitting
of the penalty weight lambda within the ``WhittakerLikeSolver`` class that would have
cluttered the class implementation.

"""

### Imports ###

from math import ceil, exp
from typing import Callable, Tuple

from scipy.optimize import OptimizeResult, brute, minimize_scalar

from chemotools.utils.models import WhittakerSmoothLambda

### Constants ###

_LN_TEN: float = 2.302585092994046  # ln(10)
_half_log_decade: float = 0.5 * _LN_TEN
_X_ABS_LOG_TOL: float = 0.05

### Optimization Functions ###


def finish_lambda_optimization(
    fun: Callable[..., float],
    xmin: float,
    args: Tuple,
) -> OptimizeResult:
    """
    This function is used to finish the optimization of the penalty weight lambda
    after the initial optimization has been performed with the ``brute`` method.

    It spans an interval of +- half a decade around the minimum found by the brute force
    method and then performs a scalar optimization with the ``minimize_scalar`` method.

    """

    # first, the bounds for the scalar optimization are set
    bounds = (xmin - _half_log_decade, xmin + _half_log_decade)

    # now, the scalar optimization is performed
    return minimize_scalar(
        fun=fun,
        bounds=bounds,
        args=args,
        method="bounded",
        options={"xatol": _X_ABS_LOG_TOL},
    )


def get_optimized_lambda(
    fun: Callable[..., float],
    lam: WhittakerSmoothLambda,
    args: Tuple,
) -> float:
    """
    This function optimizes the penalty weight lambda with the brute force method.

    """

    # first, the number of steps is computed in a way that the step size is roughly
    # half a decade
    # if the bounds are at max one decade apart, the finish optimization can be run
    # directly
    log_low_bound, log_upp_bound = lam.log_auto_bounds
    bound_log_diff = log_upp_bound - log_low_bound
    if bound_log_diff <= _LN_TEN:
        return minimize_scalar(
            fun=fun,
            bounds=(log_low_bound, log_upp_bound),
            args=args,
            method="bounded",
            options={"xatol": _X_ABS_LOG_TOL},
        ).x

    # otherwise, the number of steps is computed ...
    n_steps = 1 + ceil(bound_log_diff / _half_log_decade)

    # ...and the brute force optimization with final polish is performed
    # NOTE: ``brute`` can work with floats internally and this is exploited here
    # NOTE: since the optimization is carried out over the log of lambda, the
    #       exponential of the result is returned
    return exp(
        brute(  # type: ignore
            func=fun,
            ranges=(lam.log_auto_bounds,),
            Ns=n_steps,
            args=args,
            finish=finish_lambda_optimization,
            full_output=False,
        )
    )
