import numpy as np
import pytest
from sklearn.utils.estimator_checks import check_estimator

from chemotools._optional import import_optional_dependency
from chemotools.feature_selection import RangeCut


@pytest.fixture(scope="module")
def pd():
    """Fixture for optional pandas dependency."""
    try:
        return import_optional_dependency(
            "pandas", caller_name="tests (pandas-dependent)"
        )
    except ImportError:
        pytest.skip("pandas is not installed, skipping pandas-dependent tests")


@pytest.fixture(scope="module")
def pl():
    """Fixture for optional polars dependency."""
    try:
        return import_optional_dependency(
            "polars", caller_name="tests (polars-dependent)"
        )
    except ImportError:
        pytest.skip("polars is not installed, skipping polars-dependent tests")


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
    range_cut = RangeCut(start=2.5, end=7.9, x_axis=wavenumbers)

    # Act
    spectrum_corrected = range_cut.fit_transform(spectrum)

    # Assert
    assert np.allclose(spectrum_corrected[0], spectrum[0][1:7], atol=1e-8)


def test_range_cut_by_deprecated_wavenumber_warns():
    # Arrange
    wavenumbers = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    spectrum = np.array([[10, 12, 14, 16, 14, 12, 10, 12, 14, 16]])
    range_cut = RangeCut(start=2.5, end=7.9, wavenumbers=wavenumbers)

    # Act
    with pytest.warns(
        FutureWarning,
        match="`wavenumbers` is deprecated and will be removed in a future release",
    ):
        spectrum_corrected = range_cut.fit_transform(spectrum)

    # Assert
    assert np.allclose(spectrum_corrected[0], spectrum[0][1:7], atol=1e-8)


def test_range_cut_by_wavenumber_with_list():
    # Arrange
    wavenumbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    spectrum = np.array([[10, 12, 14, 16, 14, 12, 10, 12, 14, 16]])
    range_cut = RangeCut(start=2.5, end=7.9, x_axis=wavenumbers)

    # Act
    spectrum_corrected = range_cut.fit_transform(spectrum)

    # Assert
    assert np.allclose(spectrum_corrected[0], spectrum[0][1:7], atol=1e-8)
    assert range_cut.wavenumbers_ == [2, 3, 4, 5, 6, 7]
    assert range_cut.x_axis_ == [2, 3, 4, 5, 6, 7]


def test_range_cut_by_wavenumber_with_pandas_dataframe(pd):
    # Arrange
    wavenumbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    spectrum = pd.DataFrame(np.array([[10, 12, 14, 16, 14, 12, 10, 12, 14, 16]]))
    range_cut = RangeCut(start=2.5, end=7.9, x_axis=wavenumbers).set_output(
        transform="pandas"
    )

    # Act
    spectrum_corrected = range_cut.fit_transform(spectrum)

    # Assert
    assert isinstance(spectrum_corrected, pd.DataFrame)


def test_range_cut_by_wavenumber_with_polars_dataframe(pl):
    # Arrange
    wavenumbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    spectrum = pl.DataFrame(np.array([[10, 12, 14, 16, 14, 12, 10, 12, 14, 16]]))
    range_cut = RangeCut(start=2.5, end=7.9, x_axis=wavenumbers).set_output(
        transform="polars"
    )

    # Act
    spectrum_corrected = range_cut.fit_transform(spectrum)

    # Assert
    assert isinstance(spectrum_corrected, pl.DataFrame)


def test_range_cut_with_x_axis_and_wavenumbers_raises_error(spectrum):
    # Arrange
    axis = np.arange(spectrum.shape[1])
    range_cut = RangeCut(start=0, end=10, x_axis=axis, wavenumbers=axis)

    # Act
    with pytest.raises(ValueError) as exc_info:
        range_cut.fit(spectrum)

    # Assert
    assert "Only one of `x_axis` or deprecated `wavenumbers` can be provided" in str(
        exc_info.value
    )
