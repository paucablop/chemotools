import pandas as pd

from chemotools.datasets import load_fermentation_test, load_fermentation_train


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

    
