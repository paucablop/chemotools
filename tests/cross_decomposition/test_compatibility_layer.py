"""Tests for the chemotools.cross_decomposition compatibility layer.

This module is a deprecated shim that re-exports classes from
chemotools.projection and chemotools.regression. These tests verify that
importing from the old path still works and emits the expected FutureWarning.
"""

import importlib

import pytest


def test_import_emits_future_warning():
    """Importing chemotools.cross_decomposition should emit a FutureWarning."""
    import chemotools.cross_decomposition

    with pytest.warns(
        FutureWarning,
        match=r"chemotools\.cross_decomposition has been split",
    ):
        importlib.reload(chemotools.cross_decomposition)


def test_epo_available_via_compatibility_layer():
    """ExternalParameterOrthogonalization should be importable from the old path."""
    import chemotools.cross_decomposition

    with pytest.warns(FutureWarning):
        module = importlib.reload(chemotools.cross_decomposition)

    assert hasattr(module, "ExternalParameterOrthogonalization")


def test_osc_available_via_compatibility_layer():
    """OrthogonalSignalCorrection should be importable from the old path."""
    import chemotools.cross_decomposition

    with pytest.warns(FutureWarning):
        module = importlib.reload(chemotools.cross_decomposition)

    assert hasattr(module, "OrthogonalSignalCorrection")


def test_pls_available_via_compatibility_layer():
    """PLSRegression should be importable from the old path."""
    import chemotools.cross_decomposition

    with pytest.warns(FutureWarning):
        module = importlib.reload(chemotools.cross_decomposition)

    assert hasattr(module, "PLSRegression")
