"""Tests for optional visualization dependencies."""

from __future__ import annotations

import subprocess
import sys

IMPORT_BLOCKER = """
import builtins

real_import = builtins.__import__


def blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "matplotlib" or name.startswith("matplotlib."):
        raise ModuleNotFoundError("No module named 'matplotlib'")
    return real_import(name, globals, locals, fromlist, level)


builtins.__import__ = blocked_import
"""


def run_python(code: str) -> subprocess.CompletedProcess[str]:
    """Run Python code in a subprocess and capture the result."""
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )


def test_core_imports_work_without_matplotlib() -> None:
    """Core modules should import without matplotlib being installed."""
    result = run_python(
        IMPORT_BLOCKER
        + """
from chemotools.augmentation import AddNoise
from chemotools.baseline import AirPls
from chemotools.derivative import SavitzkyGolay
from chemotools.outliers import HotellingT2

assert AddNoise is not None
assert AirPls is not None
assert SavitzkyGolay is not None
assert HotellingT2 is not None
print('core-ok')
"""
    )

    assert result.returncode == 0, result.stderr
    assert "core-ok" in result.stdout


def test_plotting_import_raises_helpful_error_without_matplotlib() -> None:
    """Plotting imports should explain how to install visualization support."""
    result = run_python(
        IMPORT_BLOCKER
        + """
try:
    import chemotools.plotting
except ImportError as exc:
    message = str(exc)
    assert "chemotools.plotting" in message
    assert "chemotools[viz]" in message
    print('plotting-guard-ok')
else:
    raise AssertionError('chemotools.plotting imported without matplotlib')
"""
    )

    assert result.returncode == 0, result.stderr
    assert "plotting-guard-ok" in result.stdout


def test_inspector_import_raises_helpful_error_without_matplotlib() -> None:
    """Inspector imports should explain how to install visualization support."""
    result = run_python(
        IMPORT_BLOCKER
        + """
try:
    import chemotools.inspector
except ImportError as exc:
    message = str(exc)
    assert "chemotools.inspector" in message
    assert "chemotools[viz]" in message
    print('inspector-guard-ok')
else:
    raise AssertionError('chemotools.inspector imported without matplotlib')
"""
    )

    assert result.returncode == 0, result.stderr
    assert "inspector-guard-ok" in result.stdout
