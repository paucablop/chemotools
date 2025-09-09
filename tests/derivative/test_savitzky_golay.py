import numpy as np
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
        window_size=15, polynomial_order=2, derivate_order=1, mode="interp"
    )

    # Act
    spectrum_corrected = savitzky_golay_filter.fit_transform(spectrum)

    # Assert
    assert np.allclose(spectrum_corrected[0], reference_sg_15_2[0], atol=1e-2)


def test_saviszky_golay_filter_2():
    # Arrange
    savitzky_golay_filter = SavitzkyGolay(
        window_size=3, polynomial_order=2, derivate_order=1, mode="interp"
    )
    array = np.ones((1, 10)).reshape(1, -1)

    # Act
    spectrum_corrected = savitzky_golay_filter.fit_transform(array)

    # Assert
    assert np.allclose(spectrum_corrected[0], np.zeros((1, 10)), atol=1e-2)


def test_saviszky_golay_filter_3():
    # Arrange
    savitzky_golay_filter = SavitzkyGolay(
        window_size=3, polynomial_order=2, derivate_order=1, mode="interp"
    )
    array = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]).reshape(1, -1)

    # Act
    spectrum_corrected = savitzky_golay_filter.fit_transform(array)

    # Assert
    assert np.allclose(spectrum_corrected[0], np.ones((1, 10)), atol=1e-2)
