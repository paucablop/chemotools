"""Tests for the chemotools.models backward-compatibility layer."""

import importlib
import sys

import pytest


def _import_compat_models_module():
    """Import the compatibility module and force warnings to be re-emitted."""
    sys.modules.pop("chemotools.models", None)
    return importlib.import_module("chemotools.models")


def test_models_compat_import_emits_both_future_warnings():
    """Importing `chemotools.models` emits both compatibility warnings."""
    # Arrange
    from chemotools.cross_decomposition import PLSRegression as CrossDecompositionPLS

    # Act
    with pytest.warns(FutureWarning) as recorded_warnings:
        models_module = _import_compat_models_module()

    warning_messages = [str(warning.message) for warning in recorded_warnings]

    # Assert
    assert len(recorded_warnings) == 2
    assert (
        "has moved to chemotools.cross_decomposition.PLSRegression"
        in warning_messages[0]
    )
    assert "compatibility layer" in warning_messages[0]
    assert "extends sklearn's PLSRegression" in warning_messages[1]
    assert "scikit-learn (see PR #32722)" in warning_messages[1]
    assert models_module.PLSRegression is CrossDecompositionPLS
