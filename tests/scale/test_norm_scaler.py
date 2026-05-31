import numpy as np
from sklearn.utils.estimator_checks import check_estimator

from chemotools.scale import NormScaler


# Test compliance with scikit-learn
def test_compliance_l_norm():
    # Arrange
    transformer = NormScaler()
    # Act & Assert
    check_estimator(transformer)


# Test functionality
def test_l1_norm(spectrum):
    # Arrange
    norm = 1
    l1_norm = NormScaler(l_norm=norm)
    spectrum_norm = np.linalg.norm(spectrum[0], ord=norm)

    # Act
    spectrum_corrected = l1_norm.fit_transform(spectrum)

    # Assert
    assert np.allclose(spectrum_corrected[0], spectrum[0] / spectrum_norm, atol=1e-8)


def test_l2_norm(spectrum):
    # Arrange
    norm = 2
    l1_norm = NormScaler(l_norm=norm)
    spectrum_norm = np.linalg.norm(spectrum[0], ord=norm)

    # Act
    spectrum_corrected = l1_norm.fit_transform(spectrum)

    # Assert
    assert np.allclose(spectrum_corrected[0], spectrum[0] / spectrum_norm, atol=1e-8)


def test_norm_scaler_snapshot_l2():
    # Snapshot of exact output for l_norm=2.
    # Arrange
    rng = np.random.default_rng(0)
    X = rng.normal(size=(3, 10)) + 3.0
    scaler = NormScaler(l_norm=2)

    # Act
    result = scaler.fit_transform(X)

    # Assert
    expected = np.array(
        [
            [
                0.3115670390553379,
                0.28586651209872105,
                0.362870633701663,
                0.30949073260889076,
                0.24563994406049428,
                0.3350776118467584,
                0.4290148078378164,
                0.39343777025123017,
                0.22888744800773467,
                0.17289959722950685,
            ],
            [
                0.27908861411956376,
                0.3571297734142291,
                0.07925872075263285,
                0.32658528208499227,
                0.20597510106113126,
                0.26629004956339114,
                0.28836706060835704,
                0.315135280984708,
                0.400613038096581,
                0.4746948842237935,
            ],
            [
                0.2945300565998865,
                0.44787402324727077,
                0.23948411847198323,
                0.34376888966789826,
                0.4003841797082674,
                0.3173569972864674,
                0.23145231294587476,
                0.21317141971433284,
                0.2607644768668278,
                0.33029973920936334,
            ],
        ]
    )
    assert np.allclose(result, expected, atol=1e-12)


def test_norm_scaler_snapshot_l1():
    # Snapshot of exact output for l_norm=1.
    # Arrange
    rng = np.random.default_rng(0)
    X = rng.normal(size=(3, 10)) + 3.0
    scaler = NormScaler(l_norm=1)

    # Act
    result = scaler.fit_transform(X)

    # Assert
    expected = np.array(
        [
            [
                0.10133078350931661,
                0.09297221470495168,
                0.11801622449215807,
                0.1006555074606632,
                0.07988934923380543,
                0.10897711467750004,
                0.1395282593021133,
                0.12795755816338053,
                0.07444094379300603,
                0.05623204466310518,
            ],
            [
                0.0932428215172049,
                0.11931618144291942,
                0.02648014422277243,
                0.10911134179967366,
                0.0688157761140461,
                0.08896685248725236,
                0.09634272773387646,
                0.10528592451300418,
                0.13384383353131646,
                0.15859439663793407,
            ],
            [
                0.095655021053868,
                0.14545679859559285,
                0.07777765929548054,
                0.11164639955072181,
                0.1300334423649608,
                0.10306856491141918,
                0.07516915632783885,
                0.06923203993593807,
                0.08468891702513902,
                0.10727200093904078,
            ],
        ]
    )
    assert np.allclose(result, expected, atol=1e-12)
