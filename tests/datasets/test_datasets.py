import pytest
from chemotools.utils._optional_dependencies import check_optional_dependency

from chemotools.datasets import (
    load_coffee,
    load_fermentation_test,
    load_fermentation_train,
)


@pytest.fixture(scope="module")
def pd():
    """Fixture for optional pandas dependency."""
    try:
        return check_optional_dependency("pandas", "tests (pandas-dependent)")
    except ImportError:
        pytest.skip("pandas is not installed, skipping pandas-dependent tests")


@pytest.fixture(scope="module")
def pl():
    """Fixture for optional polars dependency."""
    try:
        return check_optional_dependency("polars", "tests (polars-dependent)")
    except ImportError:
        pytest.skip("polars is not installed, skipping polars-dependent tests")


def test_load_coffee_pandas(pd):
    # Arrange & Act
    coffee_spectra, coffee_labels = load_coffee()

    # Assert
    assert coffee_spectra.shape == (60, 1841)
    assert coffee_labels.shape == (60, 1)
    assert isinstance(coffee_spectra, pd.DataFrame)
    assert isinstance(coffee_labels, pd.DataFrame)


def test_load_coffee_polars(pl):
    # Arrange & Act
    coffee_spectra, coffee_labels = load_coffee(set_output="polars")

    # Assert
    assert coffee_spectra.shape == (60, 1841)
    assert coffee_labels.shape == (60, 1)
    assert isinstance(coffee_spectra, pl.DataFrame)
    assert isinstance(coffee_labels, pl.DataFrame)


def test_load_coffee_exception():
    # Arrange, Act & Assert
    with pytest.raises(ValueError):
        coffee_spectra, coffee_labels = load_coffee(set_output="plars")


def test_load_fermentation_test_pandas(pd):
    # Arrange & Act
    test_spectra, test_hplc = load_fermentation_test()

    # Assert
    assert test_spectra.shape == (1629, 1047)
    assert test_hplc.shape == (34, 6)
    assert isinstance(test_spectra, pd.DataFrame)
    assert isinstance(test_hplc, pd.DataFrame)


def test_load_fermentation_test_polars(pl):
    # Arrange & Act
    test_spectra, test_hplc = load_fermentation_test(set_output="polars")

    # Assert
    assert test_spectra.shape == (1629, 1047)
    assert test_hplc.shape == (34, 6)
    assert isinstance(test_spectra, pl.DataFrame)
    assert isinstance(test_hplc, pl.DataFrame)


def test_load_fermentation_test_exception():
    # Arrange, Act & Assert
    with pytest.raises(ValueError):
        test_spectra, test_hplc = load_fermentation_test(set_output="plars")


def test_load_fermentation_train_pandas(pd):
    # Arrange & Act
    train_spectra, train_hplc = load_fermentation_train()

    # Assert
    assert train_spectra.shape == (21, 1047)
    assert train_hplc.shape == (21, 1)
    assert isinstance(train_spectra, pd.DataFrame)
    assert isinstance(train_hplc, pd.DataFrame)


def test_load_fermentation_train_polars(pl):
    # Arrange & Act
    train_spectra, train_hplc = load_fermentation_train(set_output="polars")

    # Assert
    assert train_spectra.shape == (21, 1047)
    assert train_hplc.shape == (21, 1)
    assert isinstance(train_spectra, pl.DataFrame)
    assert isinstance(train_hplc, pl.DataFrame)


def test_load_fermentation_train_exception():
    # Arrange, Act & Assert
    with pytest.raises(ValueError):
        train_spectra, train_hplc = load_fermentation_train(set_output="plars")
