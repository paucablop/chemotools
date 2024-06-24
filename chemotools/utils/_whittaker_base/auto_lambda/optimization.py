"""
This submodule contains the functions used for the optimization in the automated fitting
of the penalty weight lambda within the ``WhittakerLikeSolver`` class that would have
cluttered the class implementation.

"""

### Imports ###

from math import ceil, exp
from typing import Callable, Tuple

from scipy.optimize import minimize_scalar

from chemotools.utils._models import WhittakerSmoothLambda

### Constants ###

_LN_OF_A_DECADE: float = 2.302585092994046  # ln(10)
_half_ln_of_a_decade: float = 0.5 * _LN_OF_A_DECADE
_X_ABS_LN_TOL: float = 0.0049  # ~0.5% when converted from log to real

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
    log_lower_bound, log_upper_bound = lam.log_auto_bounds
    bound_log_difference = log_upper_bound - log_lower_bound
    if bound_log_difference > _LN_OF_A_DECADE:
        target_best_so_far = float("inf")
        num_steps = 1 + ceil(bound_log_difference / _half_ln_of_a_decade)
        # NOTE: the following ensures that the upper bound is not exceeded
        step_size = bound_log_difference / (num_steps - 1)

        # all the trial values are evaluated and the best one is stored
        for trial in range(0, num_steps):
            log_lam_current = log_lower_bound + trial * step_size
            target_current = fun(log_lam_current, *args)

            if target_current < target_best_so_far:
                log_lam_best_so_far = log_lam_current
                target_best_so_far = target_current

        # then, the bounds for the final optimization are shrunk to plus/minus half
        # a decade around the best trial value
        # NOTE: the following ensures that the bounds are not violated
        log_lower_bound = max(
            log_lam_best_so_far - _half_ln_of_a_decade,
            log_lower_bound,
        )
        log_upper_bound = min(
            log_lam_best_so_far + _half_ln_of_a_decade,
            log_upper_bound,
        )

    # finally, a scalar optimization is performed
    # NOTE: since the optimization is carried out over the log of lambda, the
    #       exponential of the result is returned
    return exp(
        minimize_scalar(
            fun=fun,
            bounds=(log_lower_bound, log_upper_bound),
            args=args,
            method="bounded",
            options={"xatol": _X_ABS_LN_TOL},
        ).x
    )
