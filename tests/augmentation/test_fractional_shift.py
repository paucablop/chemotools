import numpy as np
import pytest
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
    assert spectrum_right_shifted[0][6] == 7.669684297331299
    assert spectrum_left_shifted[0][4] == 4.749080237694725
    assert spectrum_right_shifted[0][-1] == 30
    assert spectrum_left_shifted[0][0] == 30


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
    assert spectrum_right_shifted[0][6] == 7.669684297331299
    assert spectrum_left_shifted[0][4] == 4.749080237694725
    assert spectrum_right_shifted[0][-1] == 0
    assert spectrum_left_shifted[0][0] == 0


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
    assert spectrum_right_shifted[0][6] == 7.669684297331299
    assert spectrum_left_shifted[0][4] == 4.749080237694725
    assert spectrum_right_shifted[0][-1] == 9
    assert spectrum_left_shifted[0][0] == 1


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
    assert spectrum_right_shifted[0][6] == 7.669684297331299
    assert spectrum_left_shifted[0][4] == 4.749080237694725
    assert spectrum_right_shifted[0][-1] == 8
    assert spectrum_left_shifted[0][0] == 9


def test_fractional_shift_linear():
    # Arrange
    spectrum = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9]])
    spectrum_right_shift = FractionalShift(
        shift=1.5, padding_mode="linear", random_state=44
    )
    spectrum_left_shift = FractionalShift(
        shift=1.5, padding_mode="linear", random_state=42
    )

    # Act
    spectrum_right_shifted = spectrum_right_shift.fit_transform(spectrum)
    spectrum_left_shifted = spectrum_left_shift.fit_transform(spectrum)

    # Assert
    assert spectrum_right_shifted[0][6] == 8.004526445996948
    assert spectrum_left_shifted[0][4] == 4.623620356542087
    assert spectrum_right_shifted[0][-1] == 11.0
    assert spectrum_left_shifted[0][0] == 0


def test_fractional_shift_min_shift():
    # Arrange
    # A linear ramp combined with 'linear' padding guarantees that the
    # value-delta at every point equals exactly the applied index shift,
    # which lets us assert directly on the shift magnitude.
    spectrum = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9]])
    min_shift = 0.5
    shift = 1.0

    seen_signs = set()
    for random_state in range(50):
        transformer = FractionalShift(
            shift=shift,
            min_shift=min_shift,
            padding_mode="linear",
            random_state=random_state,
        )

        # Act
        shifted_spectrum = transformer.fit_transform(spectrum)
        delta = shifted_spectrum[0] - spectrum[0]
        delta = delta[1:-1]  # Ignore edge points affected by padding

        # Assert: on the interior (unaffected by edge padding for |shift| <= 1)
        # the applied shift is uniform and its magnitude stays within
        # [min_shift, shift].
        np.testing.assert_allclose(delta, delta[0])  # All deltas should be the same
        assert min_shift - 1e-9 <= np.abs(delta[0]) <= shift + 1e-9, (
            f"Shift magnitude {np.abs(delta[0])} not in [{min_shift}, {shift}]"
        )
        seen_signs.add(np.sign(delta[0]))

    # Both shift directions should be sampled.
    assert seen_signs == {-1.0, 1.0}


def test_fractional_shift_min_shift_none_matches_full_range():
    # Arrange
    # With min_shift=None the behaviour must match drawing from [-shift, shift].
    spectrum = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9]])
    transformer = FractionalShift(
        shift=1.0,
        min_shift=None,
        padding_mode="linear",
        random_state=42,
    )
    reference = FractionalShift(
        shift=1.0,
        padding_mode="linear",
        random_state=42,
    )

    # Act
    shifted = transformer.fit_transform(spectrum)
    expected = reference.fit_transform(spectrum)

    # Assert
    np.testing.assert_allclose(shifted, expected)


def test_fractional_shift_min_shift_greater_than_shift_raises():
    # Arrange
    spectrum = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9]])
    transformer = FractionalShift(shift=1.0, min_shift=2.0)

    # Act & Assert
    with pytest.raises(ValueError):
        transformer.fit(spectrum)
