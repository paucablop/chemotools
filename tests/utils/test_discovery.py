import pytest

from chemotools.utils import discovery as discovery_module
from chemotools.utils.discovery import all_displays, all_estimators, all_functions


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


@pytest.mark.parametrize(
    "unimportable_module", ["chemotools.plotting", "chemotools.inspector"]
)
@pytest.mark.parametrize(
    "discover", [all_estimators, all_displays, all_functions], ids=lambda f: f.__name__
)
def test_discovery_skips_module_with_missing_optional_dependency(
    monkeypatch, unimportable_module, discover
):
    """Reproduces #283: a submodule whose package raises ImportError at
    import time (e.g. because matplotlib/the `viz` extra isn't installed)
    must be skipped with a warning instead of crashing discovery."""
    real_import_module = discovery_module.import_module

    def fake_import_module(name, package=None):
        if name == unimportable_module:
            raise ImportError(
                f"'{name}' requires the optional dependency 'matplotlib'. "
                "Install it with: pip install chemotools[viz]"
            )
        return real_import_module(name, package)

    monkeypatch.setattr(discovery_module, "import_module", fake_import_module)

    with pytest.warns(UserWarning, match=unimportable_module):
        result = discover()

    assert isinstance(result, list)
