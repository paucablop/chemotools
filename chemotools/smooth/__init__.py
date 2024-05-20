"""
The ``chemotools`` module for smoothing data.
It defines the estimator classes for smoothing data with an Sklearn-like API:

- :class:`MeanFilter`
- :class:`MedianFilter`
- :class:`SavitzkyGolayFilter`
- :class:`WhittakerSmooth`

as well as auxiliary models to allow for convenient usage of the them:

- :class:`WhittakerSmoothMethods` and :class:`WhittakerSmoothLambda` for the
    :class:`WhittakerSmooth` class.

"""

### Imports ###

from chemotools.utils.models import (   # noqa: F401
    WhittakerSmoothLambda,
    WhittakerSmoothMethods,
)

from ._mean_filter import MeanFilter  # noqa: F401
from ._median_filter import MedianFilter  # noqa: F401
from ._savitzky_golay_filter import SavitzkyGolayFilter  # noqa: F401
from ._whittaker_smooth import WhittakerSmooth  # noqa: F401
