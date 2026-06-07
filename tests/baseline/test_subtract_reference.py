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


def test_subtract_reference_snapshot_simple():
    # Snapshot of the exact output for plain subtraction (x - r).
    # Arrange
    rng = np.random.default_rng(0)
    X = rng.normal(size=(3, 10))
    reference = rng.normal(size=10)
    transformer = SubtractReference(reference=reference)

    # Act
    result = transformer.fit_transform(X)

    # Assert
    expected = np.array(
        [
            [
                1.1353484046321292,
                0.07707071158041118,
                0.7996476603577598,
                -0.43594546753276797,
                -0.7503284956674519,
                0.00622234586956333,
                1.9578286545484767,
                1.0766945968220116,
                -1.487710705868322,
                -2.758852616266813,
            ],
            [
                0.3863437210013837,
                0.25050155421895665,
                -2.1658057647243565,
                -0.7596372486183534,
                -1.460570069759406,
                -1.087640063743373,
                0.10956962656102953,
                -0.18668652267638508,
                -0.3723449336871967,
                -0.4509177757780831,
            ],
            [
                0.8810835205947016,
                1.5756390454213989,
                -0.5059696635721358,
                -0.18933551459278797,
                0.6888110591454677,
                -0.26136041127904686,
                -0.08967063993546898,
                -0.79211174256565,
                -1.2417012957286686,
                -1.2732360217507113,
            ],
        ]
    )
    assert np.allclose(result, expected, atol=1e-12)


def test_subtract_reference_snapshot_scaled():
    # Snapshot of the exact output for scaled subtraction (x - a*r).
    # Arrange
    rng = np.random.default_rng(0)
    X = rng.normal(size=(3, 10))
    reference = rng.normal(size=10)
    transformer = SubtractReference(reference=reference, scale_reference=True)

    # Act
    result = transformer.fit_transform(X)

    # Assert
    expected = np.array(
        [
            [
                -0.6139669596271106,
                -0.2853574349376211,
                0.5237663809205624,
                0.5011508589879359,
                -0.3783992780670029,
                0.6219590175121377,
                0.8249722433863134,
                0.8521194806711528,
                -0.12935528265894392,
                -0.17125852337801795,
            ],
            [
                -0.07676382518630498,
                0.15455361193675277,
                -2.238841596324328,
                -0.5115536902256272,
                -1.3621068476111295,
                -0.9246321207914073,
                -0.19033876244375175,
                -0.24613974254762308,
                -0.01273873588785562,
                0.23411270835931053,
            ],
            [
                0.06805019118113773,
                1.4071924813214367,
                -0.6341916409962466,
                0.24620090262434208,
                0.8616734579273617,
                0.02481693925235796,
                -0.6161909224030617,
                -0.8964880363535276,
                -0.6103753182503009,
                -0.07059395760630963,
            ],
        ]
    )
    assert np.allclose(result, expected, atol=1e-12)
