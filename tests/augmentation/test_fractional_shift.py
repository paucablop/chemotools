import numpy as np
from sklearn.utils.estimator_checks import check_estimator

from chemotools.augmentation import FractionalShift


# Test compliance with scikit-learn
def test_compliance_fractional_shift():
    # Arrange
    transformer = FractionalShift()
    # Act & Assert
    check_estimator(transformer)


# Test functionality
def test_fractional_shift_constant():
    # Arrange
    spectrum = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9]])
    spectrum_right_shift = FractionalShift(
        shift=1, padding_mode="constant", pad_value=30, random_state=44
    )
    spectrum_left_shift = FractionalShift(
        shift=1, padding_mode="constant", pad_value=30, random_state=42
    )

    # Act
    spectrum_right_shifted = spectrum_right_shift.fit_transform(spectrum)
    spectrum_left_shifted = spectrum_left_shift.fit_transform(spectrum)

    # Assert
    assert spectrum_right_shifted[0][6] == 6.245131019347744
    assert spectrum_left_shifted[0][4] == 5.547912097111927
    assert spectrum_right_shifted[0][0] == 30
    assert spectrum_left_shifted[0][-1] == 30


def test_fractional_shift_zeros():
    # Arrange
    spectrum = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9]])
    spectrum_right_shift = FractionalShift(
        shift=1, padding_mode="zeros", random_state=44
    )
    spectrum_left_shift = FractionalShift(
        shift=1, padding_mode="zeros", random_state=42
    )

    # Act
    spectrum_right_shifted = spectrum_right_shift.fit_transform(spectrum)
    spectrum_left_shifted = spectrum_left_shift.fit_transform(spectrum)

    # Assert
    assert spectrum_right_shifted[0][6] == 6.245131019347744
    assert spectrum_left_shifted[0][4] == 5.547912097111927
    assert spectrum_right_shifted[0][0] == 0
    assert spectrum_left_shifted[0][-1] == 0


def test_fractional_shift_extend():
    # Arrange
    spectrum = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9]])
    spectrum_right_shift = FractionalShift(
        shift=1, padding_mode="extend", random_state=44
    )
    spectrum_left_shift = FractionalShift(
        shift=1, padding_mode="extend", random_state=42
    )

    # Act
    spectrum_right_shifted = spectrum_right_shift.fit_transform(spectrum)
    spectrum_left_shifted = spectrum_left_shift.fit_transform(spectrum)

    # Assert
    assert spectrum_right_shifted[0][6] == 6.245131019347744
    assert spectrum_left_shifted[0][4] == 5.547912097111927
    assert spectrum_right_shifted[0][0] == 1
    assert spectrum_left_shifted[0][-1] == 9


def test_fractional_shift_mirror():
    # Arrange
    spectrum = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9]])
    spectrum_right_shift = FractionalShift(
        shift=1, padding_mode="mirror", random_state=44
    )
    spectrum_left_shift = FractionalShift(
        shift=1, padding_mode="mirror", random_state=42
    )

    # Act
    spectrum_right_shifted = spectrum_right_shift.fit_transform(spectrum)
    spectrum_left_shifted = spectrum_left_shift.fit_transform(spectrum)

    # Assert
    assert spectrum_right_shifted[0][6] == 6.245131019347744
    assert spectrum_left_shifted[0][4] == 5.547912097111927
    assert spectrum_right_shifted[0][0] == 1
    assert spectrum_left_shifted[0][-1] == 8


def test_fractional_shift_linear():
    # Arrange
    spectrum = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9]])
    spectrum_right_shift = FractionalShift(
        shift=1, padding_mode="linear", random_state=44
    )
    spectrum_left_shift = FractionalShift(
        shift=1, padding_mode="linear", random_state=42
    )

    # Act
    spectrum_right_shifted = spectrum_right_shift.fit_transform(spectrum)
    spectrum_left_shifted = spectrum_left_shift.fit_transform(spectrum)

    # Assert
    assert spectrum_right_shifted[0][6] == 6.245131019347744
    assert spectrum_left_shifted[0][4] == 5.547912097111927
    assert spectrum_right_shifted[0][0] == 0.0
    assert spectrum_left_shifted[0][-1] == 9.547912097111926
