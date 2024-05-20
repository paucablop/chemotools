"""
This script implements utility models required for testing the
:mod:`chemotools.utils` module.

"""

### Imports ###

from dataclasses import dataclass, field
from typing import Dict, Literal, Optional, Tuple

import numpy as np

from chemotools.utils import _models
from tests.test_for_utils.utils_funcs import float_is_bit_equal

### Dataclasses ###


@dataclass
class RefDifferenceKernel:
    """
    Dataclass for storing the reference for the difference kernel validity check.

    """

    differences: int
    accuracy: int
    kernel: np.ndarray

    size: int = field(init=False)

    def __post_init__(self) -> None:
        self.size = self.kernel.size


@dataclass
class NoiseEstimationReference:
    """
    Dataclass for storing the reference for the noise estimation validity check.

    """

    window_size: Optional[int]
    min_noise_level: float
    differences: int
    accuracy: int
    noise_level: np.ndarray

    raised_noise_levels: Dict[Literal[-2, -1, 1, 2], np.ndarray] = field(init=False)

    def __post_init__(self) -> None:
        self.raised_noise_levels = {
            power: self.noise_level**power for power in (-2, -1, 1, 2)
        }


@dataclass
class ExpectedWhittakerSmoothLambda:
    """
    Dataclass for checking the expected results for the class :class:`WhittakerSmoothLambda`
    from the module :mod:`chemotools.utils.models`.

    """  # noqa: E501

    fixed_lambda: float
    auto_bounds: Tuple[float, float]
    fit_auto: bool
    method_used: _models.WhittakerSmoothMethods
    log_auto_bounds: Tuple[float, float] = (0.0, 0.0)

    def assert_is_equal_to(self, other: _models.WhittakerSmoothLambda) -> None:
        """
        Checks if the current instance is equal to another instance of the same class.

        """

        assert other.fit_auto is self.fit_auto
        assert other.method_used == self.method_used
        # NOTE: since NAN-values are used, the comparison is split into two parts for
        #       the fixed lambda value and each of the bounds
        assert float_is_bit_equal(
            value=other.fixed_lambda,
            reference=self.fixed_lambda,
        )
        assert float_is_bit_equal(
            value=other.auto_bounds[0], reference=self.auto_bounds[0]
        )
        assert float_is_bit_equal(
            value=other.auto_bounds[1],
            reference=self.auto_bounds[1],
        )
        assert float_is_bit_equal(
            value=other.log_auto_bounds[0],
            reference=self.log_auto_bounds[0],
        )
        assert float_is_bit_equal(
            value=other.log_auto_bounds[1],
            reference=self.log_auto_bounds[1],
        )
