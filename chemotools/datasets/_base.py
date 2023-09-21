import pandas as pd
import os

PACKAGE_DIRECTORY = os.path.dirname(os.path.abspath(__file__))


def load_fermentation_train():
    """
    Loads the training data of the fermentation dataset. This data corresponds to a synthetic dataset measured 
    off-line. This dataset is designed to represent the variability of real fermentation data.

    Returns
    -------
    train_spectra: pd.DataFrame A pandas DataFrame containing the synthetic spectra measured to train the model.
    train_hplc: pd.DataFrame A pandas DataFrame containing the corresponding reference measurements analyzed with HPLC.

    References
    -------
    - Cabaneros Lopez Pau, Udugama Isuru A., Thomsen Sune Tjalfe, Roslander Christian, Junicke Helena,
    Mauricio Iglesias Miguel, Gernaey Krist V. Transforming data into information:
    A parallel hybrid model for real-time state estimation in lignocellulose ethanol fermentations.
    """
    train_spectra = pd.read_csv(PACKAGE_DIRECTORY + "/data/train_spectra.csv")
    train_spectra.columns = train_spectra.columns.astype(float)
    train_hplc = pd.read_csv(PACKAGE_DIRECTORY + "/data/train_hplc.csv")

    return train_spectra, train_hplc


def load_fermentation_test():
    """
    Loads the testing data of the fermentation dataset. This data corresponds to real fermentation data measured
    on-line during a fermentation process. 

    Returns
    -------
    test_spectra: pd.DataFrame A pandas DataFrame containing the on-line spectra measured to train the model.
    test_hplc: pd.DataFrame A pandas DataFrame containing the corresponding HPLC measurements.

    References
    -------
    - Cabaneros Lopez Pau, Udugama Isuru A., Thomsen Sune Tjalfe, Roslander Christian, Junicke Helena,
    Mauricio Iglesias Miguel, Gernaey Krist V. Transforming data into information:
    A parallel hybrid model for real-time state estimation in lignocellulose ethanol fermentations.
    """
    fermentation_spectra = pd.read_csv(
        PACKAGE_DIRECTORY + "/data/fermentation_spectra.csv"
    )
    fermentation_spectra.columns = fermentation_spectra.columns.astype(float)
    fermentation_hplc = pd.read_csv(PACKAGE_DIRECTORY + "/data/fermentation_hplc.csv")

    return fermentation_spectra, fermentation_hplc


def load_coffee():
    """
    Loads the coffee dataset. This data corresponds to a coffee spectra from three different origins 
    measured off-line using attenuated total reflectance Fourier transform infrared spectroscopy (ATR-FTIR).

    Returns
    -------
    coffee_spectra: pd.DataFrame A pandas DataFrame containing the coffee spectra.
    coffee_labels: pd.DataFrame A pandas DataFrame containing the corresponding labels.
    """

    coffee_spectra = pd.read_csv(PACKAGE_DIRECTORY + "/data/coffee_spectra.csv")
    coffee_labels = pd.read_csv(PACKAGE_DIRECTORY + "/data/coffee_labels.csv")

    return coffee_spectra, coffee_labels