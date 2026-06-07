import numpy as np
import pytest
from sklearn.utils.estimator_checks import check_estimator

from chemotools.scale import ParetoScaler


# Test compliance with scikit-learn
def test_compliance_pareto_scaler():
    # Arrange
    transformer = ParetoScaler()
    # Act & Assert
    check_estimator(transformer)


# Test functionality
def test_pareto_scaler_with_mean():
    """Test that ParetoScaler correctly scales the spectrum."""
    # Arrange
    # Create a simple spectrum with known mean and std
    spectra = np.array([[1.0, 2.0, 3.0, 4.0, 5.0], [2.0, 3.0, 4.0, 5.0, 6.0]])
    mean = np.mean(spectra, axis=0)
    scale = np.power(np.std(spectra, axis=0, ddof=0), 0.5)

    # reference spectrum after Pareto scaling: (x - mean) / np.power(std, 0.5)
    reference_spectra = (spectra - mean) / scale

    # Act
    scaler = ParetoScaler()
    spectra_corrected = scaler.fit_transform(spectra)

    # Assert
    assert np.allclose(spectra_corrected, reference_spectra, atol=1e-8)


def test_pareto_scaler_without_mean():
    """Test that ParetoScaler correctly scales the spectrum without centering."""
    # Arrange
    spectra = np.array([[1.0, 2.0, 3.0, 4.0, 5.0], [2.0, 3.0, 4.0, 5.0, 6.0]])
    scale = np.power(np.std(spectra, axis=0, ddof=0), 0.5)

    # reference spectrum after Pareto scaling without mean: x / np.power(std, 0.5)
    reference_spectra = spectra / scale

    # Act
    scaler = ParetoScaler(with_mean=False)
    spectra_corrected = scaler.fit_transform(spectra)

    # Assert
    assert np.allclose(spectra_corrected, reference_spectra, atol=1e-8)


def test_raises_warning_zero_std():
    """Test that ParetoScaler raises a warning when a feature has zero std."""
    # Arrange
    spectra = np.array([[1.0, 2.0, 3.0], [1.0, 3.0, 4.0]])  # First feature has zero std

    # Act & Assert
    with pytest.warns(
        UserWarning,
        match=r"The scale for 1 feature\(s\) is zero \(constant columns\)\.",
    ):
        ParetoScaler().fit(spectra)


def test_inverse_transform():
    """Test that inverse_transform correctly recovers the original data."""
    # Arrange
    spectra = np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
    scaler = ParetoScaler()

    # Act
    spectra_scaled = scaler.fit_transform(spectra)
    spectra_recovered = scaler.inverse_transform(spectra_scaled)

    # Assert
    assert np.allclose(spectra_recovered, spectra, atol=1e-8)


def test_pareto_scaler_snapshot_with_mean():
    # Snapshot of exact output for default settings (p=0.5, with_mean=True).
    # Arrange
    rng = np.random.default_rng(0)
    X = rng.normal(size=(3, 10))
    scaler = ParetoScaler()

    # Act
    result = scaler.fit_transform(X)

    # Assert
    expected = np.array(
        [
            [
                0.599686170011576,
                -0.6812419267541049,
                1.2923871167240137,
                0.05316882667527211,
                -0.25694626492996453,
                0.6650927994569555,
                1.3525217164418657,
                1.1832765752184269,
                -0.6560431076072302,
                -1.293903374591589,
            ],
            [
                -0.7434256850341884,
                -0.4692531356236503,
                -1.3995704932320623,
                -0.6166509980014265,
                -1.008055162832515,
                -0.93802724531299,
                -0.5725030921121661,
                -0.24856232704894893,
                0.956432358677627,
                1.0676657747472797,
            ],
            [
                0.14373951502261226,
                1.150495062377755,
                0.1071833765080484,
                0.5634821713261544,
                1.2650014277624793,
                0.2729344458560345,
                -0.7800186243296997,
                -0.934714248169478,
                -0.3003892510703968,
                0.22623759984430938,
            ],
        ]
    )
    assert np.allclose(result, expected, atol=1e-12)


def test_pareto_scaler_snapshot_without_mean():
    # Snapshot of exact output without mean-centering (p=0.5, with_mean=False).
    # Arrange
    rng = np.random.default_rng(0)
    X = rng.normal(size=(3, 10))
    scaler = ParetoScaler(with_mean=False)

    # Act
    result = scaler.fit_transform(X)

    # Assert
    expected = np.array(
        [
            [
                0.22545887119086863,
                -0.16147502857921847,
                0.5813581872058213,
                0.21707124550510618,
                -0.5664918067202594,
                0.5299389352086833,
                1.358160517124302,
                1.0733636474983146,
                -1.0173844588039682,
                -1.2948287163516907,
            ],
            [
                -1.1176529838548956,
                0.05051376255123627,
                -2.1105994227502545,
                -0.45274857917159245,
                -1.31760070462281,
                -1.0731811095612622,
                -0.5668642914297296,
                -0.3584752547690612,
                0.5950910074808891,
                1.0667404329871781,
            ],
            [
                -0.23048778379809515,
                1.6702619605526416,
                -0.6038455530101441,
                0.7273845901559884,
                0.9554558859721846,
                0.1377805816077623,
                -0.7743798236472632,
                -1.0446271758895904,
                -0.6617306022671348,
                0.22531225808420782,
            ],
        ]
    )
    assert np.allclose(result, expected, atol=1e-12)
