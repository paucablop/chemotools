import pytest
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
    "type_filter",
    ["classifier", "regressor", "transformer", "cluster", "selector", "outlier"],
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
