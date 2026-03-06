import pytest

from chemotools._optional import import_optional_dependency


def test_import_optional_dependency_existing_module():
    """Test that an existing dependency is successfully imported."""
    module = import_optional_dependency("math", caller_name="test_function")
    import math

    assert module is math


def test_import_optional_dependency_nonexistent_module():
    """Test that a missing dependency raises ImportError with a clear message."""
    fake_module = "nonexistent_package_12345"

    with pytest.raises(ImportError) as excinfo:
        import_optional_dependency(fake_module, caller_name="test_function")

    # Assert the message is informative
    msg = str(excinfo.value)
    assert fake_module in msg
    assert "test_function" in msg
    assert "pip install nonexistent_package_12345" in msg


def test_import_optional_dependency_with_extra_name():
    """Test that the error message includes the extra install hint."""
    fake_module = "nonexistent_package_12345"

    with pytest.raises(ImportError) as excinfo:
        import_optional_dependency(
            fake_module, caller_name="test_function", extra_name="viz"
        )

    msg = str(excinfo.value)
    assert "pip install chemotools[viz]" in msg


def test_import_optional_dependency_real_optional(monkeypatch):
    """
    Test that if import fails for a real optional package (e.g. pandas),
    the raised ImportError includes the correct package name and caller.
    """
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "pandas":
            raise ImportError("No module named 'pandas'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)

    with pytest.raises(ImportError) as excinfo:
        import_optional_dependency("pandas", caller_name="test_loader")

    msg = str(excinfo.value)
    assert "pandas" in msg
    assert "test_loader" in msg
    assert "pip install pandas" in msg
