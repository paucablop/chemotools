import pytest
from unittest.mock import patch, MagicMock
from chemotools.utils.discovery import all_estimators, all_displays, all_functions


def test_all_estimators_returns_list_of_tuples():
    result = all_estimators()
    assert isinstance(result, list)
    for name, cls in result:
        assert isinstance(name, str)
        assert isinstance(cls, type)
    # Check no duplicate names
    names = [name for name, _ in result]
    assert len(names) == len(set(names))


@pytest.mark.parametrize(
    "type_filter", ["classifier", "regressor", "transformer", "cluster"]
)
def test_all_estimators_type_filter(type_filter):
    result = all_estimators(type_filter)
    assert isinstance(result, list)
    for name, cls in result:
        assert isinstance(name, str)
        assert isinstance(cls, type)


def test_all_estimators_type_filter_list():
    """Test type_filter as a list to cover the copy line."""
    type_filter = ["classifier", "regressor"]
    result = all_estimators(type_filter)
    assert isinstance(result, list)
    # Verify original list wasn't modified
    assert type_filter == ["classifier", "regressor"]


@patch("chemotools.utils.discovery.import_module")
def test_all_estimators_handles_import_errors(mock_import_module):
    """Test that import errors are handled gracefully."""

    # Mock import_module to raise an exception for certain modules
    def side_effect(module_name):
        if "failing_module" in module_name:
            raise ImportError("Module not found")
        # For other modules, return a mock that behaves normally
        mock_module = MagicMock()
        mock_module.__name__ = module_name
        return mock_module

    mock_import_module.side_effect = side_effect

    # This should not raise an exception despite the import error
    result = all_estimators()
    assert isinstance(result, list)


def test_all_estimators_invalid_type_filter():
    with pytest.raises(ValueError, match="Parameter type_filter must be"):
        all_estimators("invalid_type")


def test_all_displays_returns_list_of_tuples():
    result = all_displays()
    assert isinstance(result, list)
    for name, cls in result:
        assert isinstance(name, str)
        assert isinstance(cls, type)
        assert name.endswith("Display")
    # Check no duplicate names
    names = [name for name, _ in result]
    assert len(names) == len(set(names))


def test_all_functions_returns_list_of_tuples():
    result = all_functions()
    assert isinstance(result, list)
    for name, func in result:
        assert isinstance(name, str)
        assert callable(func)
        assert not name.startswith("_")
    # Check no duplicate names
    names = [name for name, _ in result]
    assert len(names) == len(set(names))
