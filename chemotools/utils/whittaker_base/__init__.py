"""
This submodule contains the base class ``WhittakerLikeSolver`` which is used to
efficiently solve the Penalized Least Squares problems that arise in the
Whittaker-Henderson smoothing algorithm and its variants, e.g., for baseline correction.

Since the class would be too big if all the methods were implemented in a single file,
the implementation is split into the class itself and a utility module that contains
utility functions used by the class.

"""

### Imports ###

from chemotools.utils.models import (  # noqa: F401
    WhittakerSmoothLambda,
    WhittakerSmoothMethods,
)
from chemotools.utils.whittaker_base.main import (  # noqa: F401
    WhittakerLikeSolver,
)