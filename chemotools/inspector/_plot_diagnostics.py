"""Diagnostic plot creation functions for inspectors.

This module re-exports the diagnostic plotting helpers implemented in
``_plot_core``. It remains as a compatibility layer for client code that
imports from ``chemotools.inspector._plot_diagnostics``.
"""

from ._plot_utils_latent_space import create_model_distances_plot

__all__ = ["create_model_distances_plot"]
