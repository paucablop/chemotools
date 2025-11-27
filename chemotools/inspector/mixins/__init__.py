"""Inspector mixin utilities for composing inspector functionality."""

from ._latent import LatentVariableMixin
from ._regression import RegressionMixin
from ._spectra import SpectraMixin

__all__ = ["LatentVariableMixin", "RegressionMixin", "SpectraMixin"]
