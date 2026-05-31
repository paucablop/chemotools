import numpy as np
import pytest
from sklearn.utils.estimator_checks import check_estimator

from chemotools.scale import BandScaler


# Test compliance with scikit-learn
def test_compliance_band_scaler():
    # Arrange
    transformer = BandScaler()
    # Act & Assert
    check_estimator(transformer)


# Test functionality
def test_band_scaler_with_mean():
    """Test that BandScaler correctly scales the spectrum using mean aggregation."""
    # Arrange
    spectra = np.array([[1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0]])

    # The band includes features 2:4
    scaling_factor = spectra[0, 2:4].mean()
    reference_spectra = spectra / scaling_factor

    # Act
    scaler = BandScaler(start=2, end=4)
    spectra_scaled = scaler.fit_transform(spectra)

    # Assert
    assert np.allclose(spectra_scaled, reference_spectra, atol=1e-8)


def test_band_scaler_with_mean_and_baseline_correction():
    """Test that BandScaler correctly scales the spectrum using mean aggregation with
    baseline correction."""
    # Arrange
    spectra = np.array([[1.0, 1.0, 2.0, 3.0, 2.0, 1.0, 1.0]])
    x_axis = np.array([100, 200, 300, 400, 500, 600, 700])

    # The band includes features 1:6
    baseline = np.ones_like(spectra)
    band_y = spectra[0, 1:6] - baseline[0, 1:6]
    scaling_factor = band_y.mean()
    reference_spectra = spectra / scaling_factor

    # Act
    scaler = BandScaler(
        start=200,
        end=700,
        x_axis=x_axis,
        aggregation="mean",
        baseline_correction=True,
    )
    spectra_scaled = scaler.fit_transform(spectra)

    # Assert
    assert np.allclose(spectra_scaled, reference_spectra, atol=1e-8)


def test_band_scaler_with_mean_and_baseline_correction_nonuniform_x_axis():
    """Test that BandScaler baseline correction uses actual x-axis spacing, not
    index-based spacing, so that non-uniform x-axis grids are handled correctly."""
    # Arrange: non-uniformly spaced x-axis with 6 points
    spectra = np.array([[0.0, 1.0, 3.0, 2.0, 4.0, 5.0]])
    x_axis = np.array([0.0, 1.0, 3.0, 4.0, 10.0, 11.0])

    # Band: start=1.0 (start_index_=1), end=10.0 (end_index_=4, exclusive).
    # BandScaler slices
    # X[:, start_index_:end_index_] = X[:, 1:4] → indices [1,2,3] → x=[1,3,4]
    band_y = spectra[0, 1:4]
    band_x = x_axis[1:4]  # [1.0, 3.0, 4.0] — non-uniform spacing

    # Expected baseline: linear in x (not linear in index)
    # t = (x - x[0]) / (x[-1] - x[0]) = [0, 2/3, 1] ≠ linspace(0, 1, 3) = [0, 0.5, 1]
    t = (band_x - band_x[0]) / (band_x[-1] - band_x[0])
    baseline_values = band_y[0] + t * (band_y[-1] - band_y[0])
    band_y_corrected = band_y - baseline_values
    scaling_factor = band_y_corrected.mean()
    reference_spectra = spectra / scaling_factor

    # Act
    scaler = BandScaler(
        start=1.0,
        end=10.0,
        x_axis=x_axis,
        aggregation="mean",
        baseline_correction=True,
    )
    spectra_scaled = scaler.fit_transform(spectra)

    # Assert
    assert np.allclose(spectra_scaled, reference_spectra, atol=1e-8)


def test_band_scaler_with_area_and_baseline_correction():
    """Test that BandScaler correctly scales the spectrum using area aggregation with
    baseline correction."""
    # Arrange
    spectra = np.array([[1.0, 1.0, 2.0, 3.0, 2.0, 1.0, 1.0]])
    x_axis = np.array([100, 200, 300, 400, 500, 600, 700])

    # The band includes features 1:6
    baseline = np.ones_like(spectra)
    band_y = spectra[0, 1:6] - baseline[0, 1:6]
    trapz_func = getattr(np, "trapezoid", getattr(np, "trapz", None))
    scaling_factor = trapz_func(band_y, x=x_axis[1:6], axis=0)
    reference_spectra = spectra / scaling_factor

    # Act
    scaler = BandScaler(
        start=200,
        end=700,
        x_axis=x_axis,
        aggregation="area",
        baseline_correction=True,
    )
    spectra_scaled = scaler.fit_transform(spectra)

    # Assert
    assert np.allclose(spectra_scaled, reference_spectra, atol=1e-8)


def test_band_scaler_with_area():
    """Test that BandScaler correctly scales the spectrum using area aggregation."""
    # Arrange
    spectra = np.array([[1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0]])
    x_axis = np.array([100, 200, 300, 400, 500, 600, 700])

    # The band includes features 2:4
    trapz_func = getattr(np, "trapezoid", getattr(np, "trapz", None))
    scaling_factor = trapz_func(spectra[0, 2:4], x=x_axis[2:4], axis=0)
    reference_spectra = spectra / scaling_factor

    # Act
    scaler = BandScaler(start=300, end=500, x_axis=x_axis, aggregation="area")
    spectra_scaled = scaler.fit_transform(spectra)

    # Assert
    assert np.allclose(spectra_scaled, reference_spectra, atol=1e-8)


def test_raises_error_start_larger_than_end():
    """Test that BandScaler raises an error when start is larger than end."""
    # Arrange
    spectra = np.array([[1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0]])

    # Act & Assert
    with pytest.raises(
        ValueError,
        match=r"start_index_ \(4\) must be less than end_index_ \(2\)\.",
    ):
        scaler = BandScaler(start=4, end=2)
        scaler.fit(spectra)


def test_raises_error_area_no_axis():
    """Test BandScaler raises an error when aggregation is area but no x_axis
    is provided."""
    # Arrange
    spectra = np.array([[1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0]])

    # Act & Assert
    with pytest.raises(
        ValueError,
        match=r"x_axis must be provided when aggregation='area'.",
    ):
        scaler = BandScaler(start=300, end=500, aggregation="area")
        scaler.fit(spectra)


def test_raises_warning_zero_band():
    """Test that BandScaler raises a warning when the band has zero mean or area."""
    # Arrange
    spectra = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    x_axis = np.array([100, 200, 300, 400, 500, 600, 700])

    # Act & Assert
    with pytest.warns(
        UserWarning,
        match=r"The scaling factor for sample\(s\) \[0\] is zero\. These samples will "
        "not be scaled\.",
    ):
        scaler = BandScaler(start=300, end=500, x_axis=x_axis, aggregation="area")
        scaler.fit_transform(spectra)


def test_band_scaler_snapshot_mean():
    # Snapshot of exact output for mean aggregation (default).
    # Arrange
    rng = np.random.default_rng(0)
    X = rng.normal(size=(3, 10)) + 3.0
    scaler = BandScaler(start=2, end=8)

    # Act
    result = scaler.fit_transform(X)

    # Assert
    expected = np.array(
        [
            [
                0.9006860334596942,
                0.8263902871812575,
                1.0489957882537995,
                0.894683793225402,
                0.7101022866408285,
                0.9686510037469038,
                1.2402070730539805,
                1.1373600550790866,
                0.6616737389162108,
                0.4998226156643102,
            ],
            [
                1.130209700927692,
                1.4462486607567278,
                0.32096965102047587,
                1.322554326184498,
                0.8341259571023598,
                1.078380063000798,
                1.1677841117084917,
                1.2761858909833756,
                1.6223404290541847,
                1.9223455776688767,
            ],
            [
                1.012352314664445,
                1.539422866877033,
                0.82314961148376,
                1.181594962437722,
                1.3761918079326028,
                1.090812579492904,
                0.7955428639444538,
                0.7327081747085569,
                0.8962939972439403,
                1.135299090971588,
            ],
        ]
    )
    assert np.allclose(result, expected, atol=1e-12)


def test_band_scaler_snapshot_mean_baseline_correction():
    # Snapshot of exact output for mean aggregation with baseline correction.
    # Arrange
    rng = np.random.default_rng(0)
    X = rng.normal(size=(3, 10)) + 3.0
    scaler = BandScaler(start=2, end=8, baseline_correction=True)

    # Act
    result = scaler.fit_transform(X)

    # Assert
    expected = np.array(
        [
            [
                -9.666303104333652,
                -8.868949558024692,
                -11.257986543303483,
                -9.601886125215154,
                -7.620928584164432,
                -10.395713774497661,
                -13.310095899044178,
                -12.20632564815723,
                -7.101185850494308,
                -5.364174331485617,
            ],
            [
                5.611146825996542,
                7.180183974483552,
                1.5935165280270274,
                6.566079289079577,
                4.141181245245456,
                5.35382846453908,
                5.797692327789987,
                6.335874135300022,
                8.054425954494313,
                9.543860115296695,
            ],
            [
                4.558685408375982,
                6.932116871663158,
                3.7066938736887995,
                5.320795572696278,
                6.197077265564783,
                4.91199685857933,
                3.582378973276023,
                3.299430461369407,
                4.036067589972622,
                5.11232238538434,
            ],
        ]
    )
    assert np.allclose(result, expected, atol=1e-12)


def test_band_scaler_snapshot_area():
    # Snapshot of exact output for area aggregation with x_axis.
    # Arrange
    rng = np.random.default_rng(0)
    X = rng.normal(size=(3, 10)) + 3.0
    x_axis = np.linspace(1000.0, 1900.0, 10)
    scaler = BandScaler(start=1100.0, end=1800.0, x_axis=x_axis, aggregation="area")

    # Act
    result = scaler.fit_transform(X)

    # Assert
    expected = np.array(
        [
            [
                0.00154107914099853,
                0.00141395867881614,
                0.00179483801038154,
                0.00153080927238649,
                0.00121498922073217,
                0.00165736760794176,
                0.00212200165185293,
                0.00194602979459408,
                0.00113212768857253,
                0.00085519945750781,
            ],
            [
                0.00185736051211631,
                0.00237673163748795,
                0.00052747410936559,
                0.00217345522566952,
                0.00137078332771474,
                0.00177218488252856,
                0.00191910943074,
                0.00209725441047523,
                0.00266611678139175,
                0.00315913831183087,
            ],
            [
                0.00158097113740422,
                0.00240408708069122,
                0.00128549691512552,
                0.0018452741250811,
                0.00214917227565656,
                0.00170350102381832,
                0.001242383988504,
                0.00114425626293779,
                0.00139972509544862,
                0.00177297475310486,
            ],
        ]
    )
    assert np.allclose(result, expected, atol=1e-12)
