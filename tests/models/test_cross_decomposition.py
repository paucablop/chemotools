"""Tests for the chemotools.models backward-compatibility layer."""

import importlib
import sys

import pytest


def _import_compat_models_module():
    """Import the compatibility module and force warnings to be re-emitted."""
    sys.modules.pop("chemotools.models", None)
    return importlib.import_module("chemotools.models")


def test_models_compat_import_emits_future_warning():
    """Importing `chemotools.models` emits a compatibility warning."""
    # Arrange
    from chemotools.regression import PLSRegression as RegressionPLS

    # Act
    with pytest.warns(FutureWarning) as recorded_warnings:
        models_module = _import_compat_models_module()

    warning_messages = [str(warning.message) for warning in recorded_warnings]

    # Assert
    assert len(recorded_warnings) == 1
    assert "has moved to chemotools.regression.PLSRegression" in warning_messages[0]
    assert "compatibility layer" in warning_messages[0]
    assert models_module.PLSRegression is RegressionPLS
