import pytest
from chemotools.utils._optional_dependencies import check_optional_dependency
import numpy as np
from sklearn.utils.estimator_checks import check_estimator

from chemotools.feature_selection import IndexSelector


@pytest.fixture(scope="module")
def pd():
    """Fixture for optional pandas dependency."""
    try:
        return check_optional_dependency("pandas", "tests (pandas-dependent)")
    except ImportError:
        pytest.skip("pandas is not installed, skipping pandas-dependent tests")


# Test compliance with scikit-learn
def test_compliance_index_selector():
    # Arrange
    transformer = IndexSelector()
    # Act & Assert
    check_estimator(transformer)


# Test functionality
def test_index_selector():
    # Arrange
    spectrum = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])

    # Act
    select_features = IndexSelector()
    spectrum_corrected = select_features.fit_transform(spectrum)

    # Assert
    assert np.allclose(spectrum_corrected[0], spectrum[0], atol=1e-8)


def test_index_selector_with_index():
    # Arrange
    spectrum = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    expected = np.array([[1, 2, 3, 8, 9, 10]])

    # Act
    select_features = IndexSelector(features=np.array([0, 1, 2, 7, 8, 9]))
    spectrum_corrected = select_features.fit_transform(spectrum)

    # Assert
    assert np.allclose(spectrum_corrected[0], expected, atol=1e-8)


def test_index_selector_with_wavenumbers():
    # Arrange
    wavenumbers = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    spectrum = np.array([[1.0, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0, 34.0, 55.0, 89.0]])
    expected = np.array([[1.0, 2.0, 3.0, 34.0, 55.0, 89.0]])

    # Act
    select_features = IndexSelector(
        features=np.array([1, 2, 3, 8, 9, 10]), wavenumbers=wavenumbers
    )
    spectrum_corrected = select_features.fit_transform(spectrum)

    # Assert
    assert np.allclose(spectrum_corrected[0], expected, atol=1e-8)


def test_index_selector_with_wavenumbers_and_dataframe(pd):
    # Arrange
    wavenumbers = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    spectrum = pd.DataFrame(
        np.array([[1.0, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0, 34.0, 55.0, 89.0]])
    )
    expected = np.array([[1.0, 2.0, 3.0, 34.0, 55.0, 89.0]])

    # Act
    select_features = IndexSelector(
        features=np.array([1, 2, 3, 8, 9, 10]), wavenumbers=wavenumbers
    ).set_output(transform="pandas")

    spectrum_corrected = select_features.fit_transform(spectrum)

    # Assert
    assert isinstance(spectrum_corrected, pd.DataFrame)
    assert np.allclose(spectrum_corrected.values[0], expected, atol=1e-8)
