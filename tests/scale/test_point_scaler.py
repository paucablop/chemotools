import numpy as np
import pytest
from sklearn.utils.estimator_checks import check_estimator

from chemotools.scale import PointScaler


# Test compliance with scikit-learn
def test_compliance_point_scaler():
    # Arrange
    transformer = PointScaler()
    # Act & Assert
    check_estimator(transformer)


# Test functionality
def test_point_scaler(spectrum):
    # Arrange
    index_scaler = PointScaler(point=0)
    reference_spectrum = [value / spectrum[0][0] for value in spectrum[0]]

    # Act
    spectrum_corrected = index_scaler.fit_transform(spectrum)

    # Assert
    assert np.allclose(spectrum_corrected[0], reference_spectrum, atol=1e-8)


def test_point_scaler_with_wavenumbers():
    # Arrange
    wavenumbers = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    spectrum = np.array([[10.0, 12.0, 14.0, 16.0, 14.0, 12.0, 10.0, 12.0, 14.0, 16.0]])

    # Act
    index_scaler = PointScaler(point=4, x_axis=wavenumbers)
    spectrum_corrected = index_scaler.fit_transform(spectrum)

    # Assert
    assert np.allclose(spectrum_corrected[0], spectrum[0] / spectrum[0][3], atol=1e-8)


# --- Deprecation tests ---
def test_point_scaler_wavenumbers_deprecated():
    """Using the old `wavenumbers` parameter emits a FutureWarning."""
    # Arrange
    wavenumbers = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    spectrum = np.array([[10.0, 12.0, 14.0, 16.0, 14.0]])
    scaler = PointScaler(point=2, wavenumbers=wavenumbers)

    # Act
    with pytest.warns(FutureWarning, match="wavenumbers"):
        scaler.fit(spectrum)


def test_point_scaler_wavenumbers_conflict():
    """Passing both `x_axis` and `wavenumbers` raises ValueError."""
    # Arrange
    wavenumbers = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    spectrum = np.array([[10.0, 12.0, 14.0, 16.0, 14.0]])
    scaler = PointScaler(point=2, x_axis=wavenumbers, wavenumbers=wavenumbers)

    # Act
    with pytest.raises(ValueError) as exc_info:
        scaler.fit(spectrum)

    # Assert
    assert "Only one of" in str(exc_info.value)


def test_point_scaler_scales_to_one_at_point():
    # Scaling by index 3: the value at column 3 must equal 1.0 in every row.
    # Arrange
    rng = np.random.default_rng(1)
    X = rng.normal(size=(5, 8)) + 5.0  # positive so division is well-defined

    # Act
    result = PointScaler(point=3).fit_transform(X)

    # Assert
    assert np.allclose(result[:, 3], np.ones(5), atol=1e-12)


def test_point_scaler_multi_row_ratios_preserved():
    # Each row scaled by its own point value, so relative ratios are preserved.
    # Arrange
    X = np.array([[2.0, 4.0, 8.0], [3.0, 6.0, 9.0]])

    # Act
    result = PointScaler(point=0).fit_transform(X)

    # Assert — row 0 scaled by 2, row 1 scaled by 3
    expected = np.array([[1.0, 2.0, 4.0], [1.0, 2.0, 3.0]])
    assert np.allclose(result, expected, atol=1e-12)


def test_point_scaler_snapshot():
    # Snapshot of the exact floating-point output of the current implementation.
    # Arrange
    rng = np.random.default_rng(0)
    X = rng.normal(size=(3, 10)) + 3.0
    transformer = PointScaler(point=2)

    # Act
    result = transformer.fit_transform(X)

    # Assert
    expected = np.array(
        [
            [
                0.8586173972719304,
                0.7877918066352774,
                1.0,
                0.852895505628987,
                0.6769353076458899,
                0.9234079055354066,
                1.1822803169862857,
                1.0842370082079946,
                0.6307687278875157,
                0.4764772378126847,
            ],
            [
                3.5212354106824626,
                4.505873549597579,
                1.0,
                4.12049650793971,
                2.5987689317366267,
                3.3597570971967192,
                3.63830071782704,
                3.976032895714377,
                5.054497906254351,
                5.989181754589761,
            ],
            [
                1.229852144180254,
                1.8701616879854464,
                1.0,
                1.4354558952021494,
                1.6718610915116177,
                1.3251692818352558,
                0.9664620536119263,
                0.8901275837181302,
                1.0888591633158093,
                1.3792135416612374,
            ],
        ]
    )
    assert np.allclose(result, expected, atol=1e-12)
