import numpy as np
import pytest
from sklearn.utils.estimator_checks import check_estimator

from chemotools.baseline import SubtractReference


# Test compliance with scikit-learn
def test_compliance_subtract_reference():
    # Arrange
    transformer = SubtractReference()
    # Act & Assert
    check_estimator(transformer)


# Test functionality
def test_subtract_reference(spectrum):
    # Arrange
    baseline = SubtractReference(reference=spectrum[0])

    # Act
    spectrum_corrected = baseline.fit_transform(spectrum)

    # Assert
    assert np.allclose(spectrum_corrected[0], np.zeros(spectrum.shape[1]), atol=1e-8)


def test_subtract_reference_raise_error_with_2D_reference(spectrum):
    # Arrange
    baseline = SubtractReference(reference=spectrum)

    # Act & Assert
    with pytest.raises(
        ValueError, match="Reference spectrum must be a 1D array. Got 2D array instead."
    ):
        baseline.fit(spectrum)


def test_subtract_reference_raise_error_unequal_length(spectrum):
    # Arrange
    baseline = SubtractReference(reference=spectrum[0][0:-2])

    # Act & Assert
    with pytest.raises(
        ValueError,
        match="Reference spectrum must have the same number of features as X. "
        "Got .* features in reference and .* features in X.",
    ):
        baseline.fit(spectrum)


def test_subtract_reference_without_reference(spectrum):
    # Arrange
    baseline = SubtractReference()

    # Act
    spectrum_corrected = baseline.fit_transform(spectrum)

    # Assert
    assert np.allclose(spectrum_corrected[0], spectrum, atol=1e-8)


# --- Tests for scale_reference with x_axis (lines 144-157) ---


def test_scale_reference_with_x_axis_resolves_indices():
    # Arrange
    n_features = 100
    x_axis = np.linspace(0, 99, n_features)
    reference = np.ones(n_features)
    X = 2.0 * np.ones((3, n_features))

    baseline = SubtractReference(
        reference=reference, scale_reference=True, start=20, end=80, x_axis=x_axis
    )

    # Act
    baseline.fit(X)

    # Assert — indices resolved from x_axis values
    assert baseline.start_index_ == 20
    assert baseline.end_index_ == 80
    assert baseline.x_axis_ is not None
    assert len(baseline.x_axis_) == 60


def test_scale_reference_with_x_axis_closest_indices():
    # Arrange: x_axis values don't exactly match start/end
    n_features = 50
    x_axis = np.linspace(0, 100, n_features)  # step ~2.04
    reference = np.ones(n_features)
    X = np.ones((2, n_features))

    baseline = SubtractReference(
        reference=reference, scale_reference=True, start=10, end=90, x_axis=x_axis
    )

    # Act
    baseline.fit(X)

    # Assert — closest indices are found
    expected_start = int(np.argmin(np.abs(x_axis - 10)))
    expected_end = int(np.argmin(np.abs(x_axis - 90)))
    assert baseline.start_index_ == expected_start
    assert baseline.end_index_ == expected_end


def test_scale_reference_with_x_axis_start_ge_end_raises():
    # Arrange: start > end in x_axis space
    n_features = 100
    x_axis = np.linspace(0, 99, n_features)
    reference = np.ones(n_features)
    X = np.ones((2, n_features))

    baseline = SubtractReference(
        reference=reference, scale_reference=True, start=80, end=20, x_axis=x_axis
    )

    # Act & Assert
    with pytest.raises(ValueError, match="start_index .* must be less than end_index"):
        baseline.fit(X)


def test_scale_reference_without_x_axis_start_ge_end_raises():
    # Arrange: start >= end with index-based range
    n_features = 100
    reference = np.ones(n_features)
    X = np.ones((2, n_features))

    baseline = SubtractReference(
        reference=reference, scale_reference=True, start=50, end=50
    )

    # Act & Assert
    with pytest.raises(ValueError, match="start_index .* must be less than end_index"):
        baseline.fit(X)


def test_scale_reference_with_x_axis_transform_uses_full_spectrum():
    # Arrange: reference = [1, 1, 1, ...], X = 3 * reference
    # scaling factor should be 3.0, result should be zeros
    n_features = 100
    x_axis = np.linspace(0, 99, n_features)
    reference = np.ones(n_features)
    X = 3.0 * np.ones((2, n_features))

    baseline = SubtractReference(
        reference=reference, scale_reference=True, start=20, end=80, x_axis=x_axis
    )

    # Act
    result = baseline.fit_transform(X)

    # Assert — full spectrum subtracted with a=3.0
    assert result.shape == X.shape
    assert np.allclose(result, 0.0, atol=1e-10)


def test_scale_reference_denominator_zero_raises_error():
    # Arrange: reference = [0, 0, 0, ...], X = [1, 1, 1, ...]
    n_features = 50
    x_axis = np.linspace(0, 100, n_features)
    reference = np.zeros(n_features)
    X = np.ones((2, n_features))

    baseline = SubtractReference(
        reference=reference, scale_reference=True, start=10, end=40, x_axis=x_axis
    )

    # Act & Assert — should raise an error due to zero denominator
    with pytest.raises(
        ValueError, match="Reference spectrum has zero or near-zero norm in the"
    ):
        baseline.fit_transform(X)
