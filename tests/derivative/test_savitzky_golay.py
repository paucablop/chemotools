import numpy as np
import pytest
from sklearn.utils.estimator_checks import check_estimator

from chemotools.derivative import SavitzkyGolay


# Test compliance with scikit-learn
def test_compliance_savitzky_golay():
    # Arrange
    transformer = SavitzkyGolay()
    # Act & Assert
    check_estimator(transformer)


# Test functionality
def test_savizky_golay_filter_1(spectrum, reference_sg_15_2):
    # Arrange
    savitzky_golay_filter = SavitzkyGolay(
        window_length=15, polyorder=2, deriv=1, mode="interp"
    )

    # Act
    spectrum_corrected = savitzky_golay_filter.fit_transform(spectrum)

    # Assert
    assert np.allclose(spectrum_corrected[0], reference_sg_15_2[0], atol=1e-2)


def test_saviszky_golay_filter_2():
    # Arrange
    savitzky_golay_filter = SavitzkyGolay(
        window_length=3, polyorder=2, deriv=1, mode="interp"
    )
    array = np.ones((1, 10)).reshape(1, -1)

    # Act
    spectrum_corrected = savitzky_golay_filter.fit_transform(array)

    # Assert
    assert np.allclose(spectrum_corrected[0], np.zeros((1, 10)), atol=1e-2)


def test_saviszky_golay_filter_3():
    # Arrange
    savitzky_golay_filter = SavitzkyGolay(
        window_length=3, polyorder=2, deriv=1, mode="interp"
    )
    array = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]).reshape(1, -1)

    # Act
    spectrum_corrected = savitzky_golay_filter.fit_transform(array)

    # Assert
    assert np.allclose(spectrum_corrected[0], np.ones((1, 10)), atol=1e-2)


def test_savitzky_golay_deprecated_parameters_warn():
    # Arrange
    savitzky_golay_filter = SavitzkyGolay(
        window_size=3, polynomial_order=2, derivate_order=1, mode="interp"
    )
    array = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]).reshape(1, -1)

    # Act
    with pytest.warns(FutureWarning):
        spectrum_corrected = savitzky_golay_filter.fit_transform(array)

    # Assert
    assert np.allclose(spectrum_corrected[0], np.ones((1, 10)), atol=1e-2)


def test_savitzky_golay_old_and_new_parameter_names_raise_error(spectrum):
    # Arrange
    savitzky_golay_filter = SavitzkyGolay(window_length=15, window_size=11)

    # Act
    with pytest.raises(ValueError) as exc_info:
        savitzky_golay_filter.fit(spectrum)

    # Assert
    assert (
        "Only one of `window_length` or deprecated `window_size` can be provided"
        in str(exc_info.value)
    )
