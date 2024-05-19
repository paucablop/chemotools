"""
This submodule contains the functions used for the optimization in the automated fitting
of the penalty weight lambda within the ``WhittakerLikeSolver`` class that would have
cluttered the class implementation.

"""

### Imports ###

from math import ceil, exp
from typing import Callable, Tuple

from scipy.optimize import minimize_scalar

from chemotools.utils.models import WhittakerSmoothLambda

### Constants ###

_LN_TEN: float = 2.302585092994046  # ln(10)
_half_log_decade: float = 0.5 * _LN_TEN
_X_ABS_LOG_TOL: float = 0.0049  # ~0.5% when converted from log to real

### Optimization Functions ###


def get_optimized_lambda(
    fun: Callable[..., float],
    lam: WhittakerSmoothLambda,
    args: Tuple,
) -> float:
    """
    This function optimises the penalty weight lambda with the brute force method.

    Since the number of optimisations carried out is so little, the function uses a
    custom from-scratch-implementation of a brute force search to tackle the problem
    directly without too much overhead.
    This will also allow for a more direct control in case this is taken to a lower
    level implementation in the future.

    """

    # unless the search space spans less than 1 decade, i.e., ln(10) ~= 2.3, a grid
    # search is carried out to shrink the search space for the final optimization;
    # the grid is spanned with an integer number of steps of half a decade
    log_low_bound, log_upp_bound = lam.log_auto_bounds
    bound_log_diff = log_upp_bound - log_low_bound
    if bound_log_diff > _LN_TEN:
        target_best = float("inf")
        n_steps = 1 + ceil(bound_log_diff / _half_log_decade)
        # NOTE: the following ensures that the upper bound is not exceeded
        step_size = bound_log_diff / (n_steps - 1)

        # all the trial values are evaluated and the best one is stored
        for trial in range(0, n_steps):
            log_lam_curr = log_low_bound + trial * step_size
            target_curr = fun(log_lam_curr, *args)

            if target_curr < target_best:
                log_lam_best = log_lam_curr
                target_best = target_curr

        # then, the bounds for the final optimization are shrunk to plus/minus half
        # a decade around the best trial value
        # NOTE: the following ensures that the bounds are not violated
        log_low_bound = max(log_lam_best - _half_log_decade, log_low_bound)
        log_upp_bound = min(log_lam_best + _half_log_decade, log_upp_bound)

    # finally, a scalar optimization is performed
    # NOTE: since the optimization is carried out over the log of lambda, the
    #       exponential of the result is returned
    return exp(
        minimize_scalar(
            fun=fun,
            bounds=(log_low_bound, log_upp_bound),
            args=args,
            method="bounded",
            options={"xatol": _X_ABS_LOG_TOL},
        ).x
    )
