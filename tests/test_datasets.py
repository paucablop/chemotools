import pandas as pd

from chemotools.datasets import load_coffee, load_fermentation_test, load_fermentation_train


def test_load_coffee():
    # Arrange

    # Act
    coffee_spectra, coffee_labels = load_coffee()

    # Assert
    assert coffee_spectra.shape == (60, 1841)
    assert coffee_labels.shape == (60, 1)
    assert isinstance(coffee_spectra, pd.DataFrame)
    assert isinstance(coffee_labels, pd.DataFrame)


def test_load_fermentation_test():
    # Arrange

    # Act
    test_spectra, test_hplc = load_fermentation_test()

    # Assert
    assert test_spectra.shape == (1629, 1047)
    assert test_hplc.shape == (34, 6)
    assert isinstance(test_spectra, pd.DataFrame)
    assert isinstance(test_hplc, pd.DataFrame)

def test_load_fermentation_train():
    # Arrange

    # Act
    train_spectra, train_hplc = load_fermentation_train()

    # Assert
    assert train_spectra.shape == (21, 1047)
    assert train_hplc.shape == (21, 1)
    assert isinstance(train_spectra, pd.DataFrame)
    assert isinstance(train_hplc, pd.DataFrame)

    
