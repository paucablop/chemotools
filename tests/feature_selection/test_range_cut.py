import numpy as np
import pandas as pd
import polars as pl
from sklearn.utils.estimator_checks import check_estimator

from chemotools.feature_selection import RangeCut


# Test compliance with scikit-learn
def test_compliance_range_cut():
    # Arrange
    transformer = RangeCut()
    # Act & Assert
    check_estimator(transformer)


# Test functionality
def test_range_cut_by_index(spectrum):
    # Arrange
    range_cut = RangeCut(start=0, end=10)

    # Act
    spectrum_corrected = range_cut.fit_transform(spectrum)

    # Assert
    assert np.allclose(spectrum_corrected[0], spectrum[0][:10], atol=1e-8)


def test_range_cut_by_wavenumber():
    # Arrange
    wavenumbers = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    spectrum = np.array([[10, 12, 14, 16, 14, 12, 10, 12, 14, 16]])
    range_cut = RangeCut(start=2.5, end=7.9, wavenumbers=wavenumbers)

    # Act
    spectrum_corrected = range_cut.fit_transform(spectrum)

    # Assert
    assert np.allclose(spectrum_corrected[0], spectrum[0][1:7], atol=1e-8)


def test_range_cut_by_wavenumber_with_list():
    # Arrange
    wavenumbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    spectrum = np.array([[10, 12, 14, 16, 14, 12, 10, 12, 14, 16]])
    range_cut = RangeCut(start=2.5, end=7.9, wavenumbers=wavenumbers)

    # Act
    spectrum_corrected = range_cut.fit_transform(spectrum)

    # Assert
    assert np.allclose(spectrum_corrected[0], spectrum[0][1:7], atol=1e-8)
    assert range_cut.wavenumbers_ == [2, 3, 4, 5, 6, 7]


def test_range_cut_by_wavenumber_with_pandas_dataframe():
    # Arrange
    wavenumbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    spectrum = pd.DataFrame(np.array([[10, 12, 14, 16, 14, 12, 10, 12, 14, 16]]))
    range_cut = RangeCut(start=2.5, end=7.9, wavenumbers=wavenumbers).set_output(
        transform="pandas"
    )

    # Act
    spectrum_corrected = range_cut.fit_transform(spectrum)

    # Assert
    assert isinstance(spectrum_corrected, pd.DataFrame)


def test_range_cut_by_wavenumber_with_polars_dataframe():
    # Arrange
    wavenumbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    spectrum = pl.DataFrame(np.array([[10, 12, 14, 16, 14, 12, 10, 12, 14, 16]]))
    range_cut = RangeCut(start=2.5, end=7.9, wavenumbers=wavenumbers).set_output(
        transform="polars"
    )

    # Act
    spectrum_corrected = range_cut.fit_transform(spectrum)

    # Assert
    assert isinstance(spectrum_corrected, pl.DataFrame)
