"""
This submodule contains the functions used for the automated fitting of the penalty
weight lambda within the ``WhittakerLikeSolver`` class that would have cluttered the
class implementation.

"""

### Imports ###

from chemotools.utils.whittaker_base.auto_lambda.optimization import (
    get_optimized_lambda,
)  # noqa: F401
