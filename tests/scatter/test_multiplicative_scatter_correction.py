import numpy as np
import pytest
from sklearn.utils.estimator_checks import check_estimator

from chemotools.scatter import MultiplicativeScatterCorrection


# Test compliance with scikit-learn
def test_compliance_multiplicative_scatter_correction():
    # Arrange
    transformer = MultiplicativeScatterCorrection()
    # Act & Assert
    check_estimator(transformer)


# Test functionality
def test_multiplicative_scatter_correction_mean(spectrum, reference_msc_mean):
    # Arrange
    msc = MultiplicativeScatterCorrection()

    # Act
    spectrum_corrected = msc.fit_transform(spectrum)

    # Assert
    assert np.allclose(spectrum_corrected[0], reference_msc_mean[0], atol=1e-8)


def test_multiplicative_scatter_correction_with_reference(spectrum, reference_msc_mean):
    # Arrange
    msc = MultiplicativeScatterCorrection(reference=reference_msc_mean[0])

    # Act
    spectrum_corrected = msc.fit_transform(spectrum)

    # Assert
    assert np.allclose(spectrum_corrected[0], reference_msc_mean[0], atol=1e-8)


def test_multiplicative_scatter_correction_median(spectrum, reference_msc_median):
    # Arrange
    msc = MultiplicativeScatterCorrection(method="median")

    # Act
    spectrum_corrected = msc.fit_transform(spectrum)

    # Assert
    assert np.allclose(spectrum_corrected[0], reference_msc_median[0], atol=1e-8)


def test_multiplicative_scatter_correction_with_reference_median(
    spectrum, reference_msc_median
):
    # Arrange
    msc = MultiplicativeScatterCorrection(
        method="median",
        reference=reference_msc_median[0],
    )

    # Act
    spectrum_corrected = msc.fit_transform(spectrum)

    # Assert
    assert np.allclose(spectrum_corrected[0], reference_msc_median[0], atol=1e-8)


def test_multiplicative_scatter_correction_with_weights(spectrum, reference_msc_mean):
    # Arrange
    weights = np.ones(len(spectrum[0]))

    msc = MultiplicativeScatterCorrection(weights=weights)

    # Act
    spectrum_corrected = msc.fit_transform(spectrum)

    # Assert
    assert np.allclose(spectrum_corrected[0], reference_msc_mean[0], atol=1e-8)


def test_multiplicative_scatter_correction_with_wrong_weights(spectrum):
    # Arrange
    weights = np.ones(10)
    msc = MultiplicativeScatterCorrection(weights=weights)

    # Act & Assert
    with pytest.raises(ValueError):
        msc.fit_transform(spectrum)


def test_multiplicative_scatter_correction_with_wrong_reference(spectrum):
    # Arrange
    reference = np.ones(10)
    msc = MultiplicativeScatterCorrection(reference=reference)

    # Act & Assert
    with pytest.raises(ValueError):
        msc.fit_transform(spectrum)


def test_multiplicative_scatter_correction_with_wrong_method(spectrum):
    # Arrange
    msc = MultiplicativeScatterCorrection(method="meant")
    # Act & Assert
    with pytest.raises(ValueError):
        msc.fit_transform(spectrum)
