"""
This submodule contains the functions used for the automated fitting of the penalty
weight lambda within the ``WhittakerLikeSolver`` class that would have cluttered the
class implementation.

"""

### Imports ###

from chemotools.utils.whittaker_base.auto_lambda.logml import (  # noqa: F401
    get_log_marginal_likelihood,
    get_log_marginal_likelihood_constant_term,
)
from chemotools.utils.whittaker_base.auto_lambda.optimization import (  # noqa: F401
    get_optimized_lambda,
)
from chemotools.utils.whittaker_base.auto_lambda.shared import (  # noqa: F401
    _Factorization,
)