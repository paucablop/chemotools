# Authors: Pau Cabaneros
# License: MIT

import numpy as np

from chemotools._deprecation import (
    DEPRECATED_PARAMETER,
    resolve_renamed_parameter,
)


class XAxisMixin:
    """Mixin providing x-axis resolution and index lookup for transformers
    that accept ``x_axis`` and the deprecated ``wavenumbers`` parameter."""

    def __setstate__(self, state: dict) -> None:
        """Restore old pickles that stored only one of the axis names."""
        self.__dict__.update(state)
        if "x_axis" not in self.__dict__ and "wavenumbers" in self.__dict__:
            self.x_axis = self.wavenumbers
        if "wavenumbers" not in self.__dict__ and "x_axis" in self.__dict__:
            self.wavenumbers = DEPRECATED_PARAMETER
        if "x_axis" not in self.__dict__:
            self.x_axis = None
        if "wavenumbers" not in self.__dict__:
            self.wavenumbers = DEPRECATED_PARAMETER

    @staticmethod
    def _resolve_x_axis(x_axis, wavenumbers):
        return resolve_renamed_parameter(
            new_name="x_axis",
            new_value=x_axis,
            new_default=None,
            old_name="wavenumbers",
            old_value=wavenumbers,
        )

    @staticmethod
    def _find_index(target: float, axis_values: np.ndarray) -> int:
        return int(np.argmin(np.abs(np.asarray(axis_values) - target)))

    def _find_indices(self, targets: np.ndarray, axis_values: np.ndarray) -> np.ndarray:
        return np.array([self._find_index(t, axis_values) for t in targets])
