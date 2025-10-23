import pytest
from chemotools.utils._optional_dependencies import check_optional_dependency


def test_check_optional_dependency_existing_module():
    """Test that an existing dependency is successfully imported."""
    module = check_optional_dependency("math", "test_function")
    import math

    assert module is math


def test_check_optional_dependency_nonexistent_module():
    """Test that a missing dependency raises ImportError with a clear message."""
    fake_module = "nonexistent_package_12345"

    with pytest.raises(ImportError) as excinfo:
        check_optional_dependency(fake_module, "test_function")

    # Assert the message is informative
    msg = str(excinfo.value)
    assert fake_module in msg
    assert "test_function" in msg
    assert "pip install" in msg


def test_check_optional_dependency_real_optional(monkeypatch):
    """
    Test that if __import__ fails for a real optional package (e.g. pandas),
    the raised ImportError includes the correct package name and caller.
    """

    # Monkeypatch __import__ to simulate ImportError
    def fake_import(name, *args, **kwargs):
        if name == "pandas":
            raise ImportError("No module named 'pandas'")
        return __import__(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)

    with pytest.raises(ImportError) as excinfo:
        check_optional_dependency("pandas", "test_loader")

    msg = str(excinfo.value)
    assert "pandas" in msg
    assert "test_loader" in msg
    assert "pip install pandas" in msg
