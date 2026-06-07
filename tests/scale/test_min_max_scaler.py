import numpy as np
from sklearn.utils.estimator_checks import check_estimator

from chemotools.scale import MinMaxScaler


# Test compliance with scikit-learn
def test_compliance_min_max_norm():
    # Arrange
    transformer = MinMaxScaler()
    # Act & Assert
    check_estimator(transformer)


# Test functionality
def test_max_scaler(spectrum):
    # Arrange
    max_norm = MinMaxScaler(use_min=False)

    # Act
    spectrum_corrected = max_norm.fit_transform(spectrum)

    # Assert
    assert np.allclose(
        spectrum_corrected[0], spectrum[0] / np.max(spectrum[0]), atol=1e-8
    )


def test_min_norm(spectrum):
    # Arrange
    min_norm = MinMaxScaler()

    # Act
    spectrum_corrected = min_norm.fit_transform(spectrum)

    # Assert
    assert np.allclose(
        spectrum_corrected[0],
        (spectrum[0] - np.min(spectrum[0]))
        / (np.max(spectrum[0]) - np.min(spectrum[0])),
        atol=1e-8,
    )


def test_min_max_scaler_snapshot_use_min():
    # Snapshot of exact output for use_min=True (default).
    # Arrange
    rng = np.random.default_rng(0)
    X = rng.normal(size=(3, 10)) + 3.0
    scaler = MinMaxScaler(use_min=True)

    # Act
    result = scaler.fit_transform(X)

    # Assert
    expected = np.array(
        [
            [
                0.5414260304824395,
                0.4410785076017231,
                0.7417405472363325,
                0.5333191068775681,
                0.2840141616666143,
                0.6332228930568237,
                1.0,
                0.8610897123131195,
                0.218604161171251,
                0.0,
            ],
            [
                0.5053404615622733,
                0.7026950955178927,
                0.0,
                0.6254525614483842,
                0.3204471214675334,
                0.4729747708681646,
                0.5288042904830947,
                0.5964971897399003,
                0.8126578877437057,
                1.0,
            ],
            [
                0.34664565139404546,
                1.0,
                0.11211080900496988,
                0.55643809650044,
                0.7976594940825243,
                0.4439046521165378,
                0.07788960563863696,
                0.0,
                0.20278026931139595,
                0.4990499369496462,
            ],
        ]
    )
    assert np.allclose(result, expected, atol=1e-12)


def test_min_max_scaler_snapshot_use_max():
    # Snapshot of exact output for use_min=False (max-only scaling).
    # Arrange
    rng = np.random.default_rng(0)
    X = rng.normal(size=(3, 10)) + 3.0
    scaler = MinMaxScaler(use_min=False)

    # Act
    result = scaler.fit_transform(X)

    # Assert
    expected = np.array(
        [
            [
                0.7262384266538461,
                0.6663325061888988,
                0.8458230976466472,
                0.7213987185400131,
                0.5725675188194327,
                0.7810397350513603,
                1.0,
                0.9170727048656192,
                0.5335187593204537,
                0.4030154532448431,
            ],
            [
                0.5879326350354942,
                0.7523354164606108,
                0.16696771628839918,
                0.687989891905017,
                0.4339105136933073,
                0.5609709698026775,
                0.6074787620260242,
                0.6638691324849804,
                0.8439379723917841,
                1.0,
            ],
            [
                0.6576180830145552,
                1.0,
                0.5347131247658101,
                0.7675571071870445,
                0.893966068416555,
                0.7085854075337942,
                0.516779944654415,
                0.4759629017301616,
                0.5822272856464819,
                0.737483582581,
            ],
        ]
    )
    assert np.allclose(result, expected, atol=1e-12)
