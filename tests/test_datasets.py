import pandas as pd
import polars as pl
import pytest

from chemotools.datasets import (
    load_coffee,
    load_fermentation_test,
    load_fermentation_train,
)


def test_load_coffee_pandas():
    # Arrange

    # Act
    coffee_spectra, coffee_labels = load_coffee()

    # Assert
    assert coffee_spectra.shape == (60, 1841)
    assert coffee_labels.shape == (60, 1)
    assert isinstance(coffee_spectra, pd.DataFrame)
    assert isinstance(coffee_labels, pd.DataFrame)


def test_load_coffee_polars():
    # Arrange

    # Act
    coffee_spectra, coffee_labels = load_coffee(set_output="polars")

    # Assert
    assert coffee_spectra.shape == (60, 1841)
    assert coffee_labels.shape == (60, 1)
    assert isinstance(coffee_spectra, pl.DataFrame)
    assert isinstance(coffee_labels, pl.DataFrame)


def test_load_coffee_exception():
    # Arrange

    # Act and Assert
    with pytest.raises(ValueError):
        coffee_spectra, coffee_labels = load_coffee(set_output="plars")


def test_load_fermentation_test_pandas():
    # Arrange

    # Act
    test_spectra, test_hplc = load_fermentation_test()

    # Assert
    assert test_spectra.shape == (1629, 1047)
    assert test_hplc.shape == (34, 6)
    assert isinstance(test_spectra, pd.DataFrame)
    assert isinstance(test_hplc, pd.DataFrame)


def test_load_fermentation_test_polars():
    # Arrange

    # Act
    test_spectra, test_hplc = load_fermentation_test(set_output="polars")

    # Assert
    assert test_spectra.shape == (1629, 1047)
    assert test_hplc.shape == (34, 6)
    assert isinstance(test_spectra, pl.DataFrame)
    assert isinstance(test_hplc, pl.DataFrame)


def test_load_fermentation_test_exception():
    # Arrange

    # Act and Assert
    with pytest.raises(ValueError):
        test_spectra, test_hplc = load_fermentation_test(set_output="plars")


def test_load_fermentation_train_pandas():
    # Arrange

    # Act
    train_spectra, train_hplc = load_fermentation_train()

    # Assert
    assert train_spectra.shape == (21, 1047)
    assert train_hplc.shape == (21, 1)
    assert isinstance(train_spectra, pd.DataFrame)
    assert isinstance(train_hplc, pd.DataFrame)


def test_load_fermentation_train_polars():
    # Arrange

    # Act
    train_spectra, train_hplc = load_fermentation_train(set_output="polars")

    # Assert
    assert train_spectra.shape == (21, 1047)
    assert train_hplc.shape == (21, 1)
    assert isinstance(train_spectra, pl.DataFrame)
    assert isinstance(train_hplc, pl.DataFrame)


def test_load_fermentation_train_exception():
    # Arrange

    # Act and Assert
    with pytest.raises(ValueError):
        train_spectra, train_hplc = load_fermentation_train(set_output="plars")
