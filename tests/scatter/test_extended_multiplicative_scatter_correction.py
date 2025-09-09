import numpy as np
import pytest
from sklearn.utils.estimator_checks import check_estimator

from chemotools.scatter import (
    ExtendedMultiplicativeScatterCorrection,
    MultiplicativeScatterCorrection,
)


# Test compliance with scikit-learn
def test_compliance_extended_multiplicative_scatter_correction():
    # Arrange
    transformer = ExtendedMultiplicativeScatterCorrection()
    # Act & Assert
    check_estimator(transformer)


# Test functionality
def test_extended_baseline_correction():
    # Arrange
    spectrum = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]).reshape(
        1, -1
    )
    reference = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    emsc = ExtendedMultiplicativeScatterCorrection(reference=reference)

    # Act
    spectrum_emsc = emsc.fit_transform(spectrum)

    # Assert
    assert np.allclose(spectrum_emsc[0], reference, atol=1e-8)


def test_extended_baseline_correction_with_weights():
    # Arrange
    spectrum = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]).reshape(
        1, -1
    )
    reference = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    emsc = ExtendedMultiplicativeScatterCorrection(reference=reference, weights=weights)

    # Act
    spectrum_emsc = emsc.fit_transform(spectrum)

    # Assert
    assert np.allclose(spectrum_emsc[0], reference, atol=1e-8)


def test_extended_baseline_correction_with_no_reference():
    # Arrange
    spectrum = np.array([1.0, 2.0, 3.0, 4.0, 5.0]).reshape(1, -1)

    # Act
    emsc = ExtendedMultiplicativeScatterCorrection(method="meant")

    # Assert
    with pytest.raises(ValueError):
        emsc.fit_transform(spectrum)


def test_extended_baseline_correction_with_wrong_reference():
    # Arrange
    spectrum = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]).reshape(
        1, -1
    )
    reference = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    # Act
    emsc = ExtendedMultiplicativeScatterCorrection(reference=reference)

    # Assert
    with pytest.raises(ValueError):
        emsc.fit_transform(spectrum)


def test_extended_baseline_correction_with_wrong_weights():
    # Arrange
    spectrum = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]).reshape(
        1, -1
    )
    weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    # Act
    emsc = ExtendedMultiplicativeScatterCorrection(weights=weights)

    # Assert
    with pytest.raises(ValueError):
        emsc.fit_transform(spectrum)


def test_extended_baseline_correction_with_noreference_no_median_no_mean(spectrum):
    # Arrange
    emsc = ExtendedMultiplicativeScatterCorrection(method="meant")

    # Act & Assert
    with pytest.raises(ValueError):
        emsc.fit_transform(spectrum)


def test_extended_baseline_correction_through_msc(spectrum):
    # EMSC of 0 order should be equivalient to MSC
    # Arrange
    msc = MultiplicativeScatterCorrection()
    emsc = ExtendedMultiplicativeScatterCorrection(order=0)

    # Act
    spectrum_msc = msc.fit_transform(spectrum)
    spectrum_emsc = emsc.fit_transform(spectrum)

    # Assert
    assert np.allclose(spectrum_emsc[0], spectrum_msc, atol=1e-8)


def test_extended_baseline_correction_through_msc_median(spectrum):
    # EMSC of 0 order should be equivalient to MSC
    # Arrange
    msc = MultiplicativeScatterCorrection(method="median")
    emsc = ExtendedMultiplicativeScatterCorrection(order=0, method="median")

    # Act
    spectrum_msc = msc.fit_transform(spectrum)
    spectrum_emsc = emsc.fit_transform(spectrum)

    # Assert
    assert np.allclose(spectrum_emsc[0], spectrum_msc, atol=1e-8)
