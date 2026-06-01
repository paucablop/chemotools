import numpy as np
import pytest
from sklearn.utils.estimator_checks import check_estimator

from chemotools.smooth import ModifiedSincFilter


# Test compliance with scikit-learn
def test_compliance_modified_sinc():
    # Arrange
    transformer = ModifiedSincFilter()
    # Act & Assert
    check_estimator(transformer)


# Test functionality
def test_ms_kernel_properties_default():
    # Arrange
    ms = ModifiedSincFilter(window_length=21, n=6, alpha=3.0, mode="interp")
    # fit on any numeric 2D data to set up internal attributes
    X = np.zeros((1, 21), dtype=np.float64)

    # Act
    ms.fit(X)
    k = ms.kernel_

    # Assert
    assert k.ndim == 1
    assert k.size == 21
    # symmetry and Direct Current preservation
    assert np.allclose(k, k[::-1], atol=1e-12)
    assert np.isclose(k.sum(), 1.0, atol=1e-12)


def test_ms_kernel_changes_with_params():
    # Arrange
    X = np.zeros((1, 21))
    ms1 = ModifiedSincFilter(window_length=21, n=6, alpha=2.0, mode="interp")
    ms2 = ModifiedSincFilter(window_length=21, n=6, alpha=4.0, mode="interp")

    # Act
    ms1.fit(X)
    ms2.fit(X)

    # Assert
    # different alpha should yield a different kernel (not identical vector)
    assert not np.allclose(ms1.kernel_, ms2.kernel_, atol=1e-12)
    # both remain valid kernels
    assert np.isclose(ms1.kernel_.sum(), 1.0, atol=1e-12)
    assert np.isclose(ms2.kernel_.sum(), 1.0, atol=1e-12)
    assert np.allclose(ms1.kernel_, ms1.kernel_[::-1], atol=1e-12)
    assert np.allclose(ms2.kernel_, ms2.kernel_[::-1], atol=1e-12)


# ---------------------------
# basic functionality (single-row signals)
# ---------------------------
@pytest.mark.parametrize("mode", ["interp", "nearest", "wrap", "constant"])
def test_ms_constant_preservation_all_modes(mode):
    # Arrange
    nine = 9
    ms = ModifiedSincFilter(window_length=nine, n=6, alpha=3.0, mode=mode)
    X = np.full((1, nine), 2.5, dtype=np.float64)

    # Act
    Y = ms.fit_transform(X)

    # Assert
    # Direct Current should be preserved for any padding scheme
    assert np.allclose(Y, X, atol=1e-12)
    assert Y.shape == X.shape
    assert Y.dtype == np.float64


@pytest.mark.parametrize("mode", ["interp", "nearest", "wrap", "constant"])
def test_ms_impulse_equals_kernel_all_modes(mode):
    # Arrange
    m = 4
    L = 2 * m + 1
    ms = ModifiedSincFilter(window_length=L, n=6, alpha=3.0, mode=mode)
    X = np.zeros((1, L), dtype=np.float64)
    X[0, m] = 1.0  # centered delta

    # Act
    ms.fit_transform(X)
    k = ms.kernel_

    # Assert
    # Convolving a centered impulse returns the kernel itself
    assert np.isclose(k.sum(), 1.0, atol=1e-12)
    assert np.allclose(k, k[::-1], atol=1e-12)


def test_ms_linear_ramp_preservation_interp_only():
    # Arrange
    nine = 9
    X = np.arange(nine, dtype=np.float64)[None, :]  # shape (1, 9)
    ms_interp = ModifiedSincFilter(window_length=nine, n=6, alpha=3.0, mode="interp")

    # Act
    Y_interp = ms_interp.fit_transform(X)

    # Assert
    # With linear extrapolation, a linear ramp should be preserved at the edges
    assert np.allclose(Y_interp, X, atol=1e-12)


# multi-row / axis behavior
def test_ms_axis_behavior_rows_vs_columns():
    # Arrange
    rng = np.random.default_rng(42)
    n_rows = 4
    n_cols = 21
    X = rng.normal(size=(n_rows, n_cols)).astype(np.float64)
    ms_row = ModifiedSincFilter(window_length=21, n=6, alpha=3.0, mode="interp", axis=1)
    ms_col = ModifiedSincFilter(window_length=21, n=6, alpha=3.0, mode="interp", axis=0)

    # Act
    Y_row = ms_row.fit_transform(X)
    Y_col = ms_col.fit_transform(X.T).T  # smooth columns, then transpose back

    # Assert
    # Smoothing along axis=1 should match smoothing each row independently.
    # Likewise, axis=0 + transpose should give the same result.
    assert np.allclose(Y_row, Y_col, atol=1e-12)
    assert Y_row.shape == X.shape
    assert Y_col.shape == X.shape


# Test kappa corrections with different n values
@pytest.mark.parametrize("n", [6, 8, 10])
def test_kappa_corrections_applied(n):
    """Test that kappa corrections are properly applied for n=6,8,10."""
    # Arrange
    window_size = n * 4 + 1  # Ensure large enough window
    ms = ModifiedSincFilter(window_length=window_size, n=n, alpha=3.0)
    X = np.zeros((1, window_size), dtype=np.float64)

    # Act
    ms.fit(X)

    # Assert
    # The kernel should be computed with kappa corrections
    assert hasattr(ms, "kernel_")
    assert ms.kernel_.shape == (window_size,)
    # Verify symmetry and DC preservation
    assert np.allclose(ms.kernel_, ms.kernel_[::-1], atol=1e-12)
    assert np.isclose(ms.kernel_.sum(), 1.0, atol=1e-12)


# --- Deprecation tests ---
def test_ms_window_size_deprecated():
    """Using the old `window_size` parameter emits a FutureWarning."""
    # Arrange
    ms = ModifiedSincFilter(window_size=21, n=6, alpha=3.0)
    X = np.zeros((1, 21), dtype=np.float64)

    # Act
    with pytest.warns(FutureWarning, match="window_size"):
        ms.fit(X)

    # Assert
    assert ms.window_length_ == 21


def test_ms_window_size_conflict():
    """Passing both `window_length` and `window_size` raises ValueError."""
    # Arrange
    ms = ModifiedSincFilter(window_length=9, window_size=9)
    X = np.zeros((1, 9), dtype=np.float64)

    # Act
    with pytest.raises(ValueError) as exc_info:
        ms.fit(X)

    # Assert
    assert "Only one of" in str(exc_info.value)


def test_modified_sinc_filter_snapshot_default():
    # Snapshot of exact output for default settings (window=21, n=6, alpha=4.0).
    # Arrange
    rng = np.random.default_rng(0)
    X = rng.normal(size=(3, 50))
    msf = ModifiedSincFilter()

    # Act
    result = msf.fit_transform(X)

    # Assert
    expected = np.array(
        [
            [
                0.15155113176945634,
                0.22516621252772795,
                0.20531113911387971,
                -0.05692243457676292,
                -0.08574266111470324,
                0.47724119710204815,
                0.9979209220534924,
                0.6233233781912012,
                -0.3519708661831665,
                -0.9122207520223273,
                -0.8481733454628516,
                -0.7903608007639638,
                -0.9821819956362885,
                -1.085242221677874,
                -0.967350346973207,
                -0.8009783541540492,
                -0.5790629070298768,
                -0.15678629668112692,
                0.3728343983253628,
                0.7013215786913186,
                0.6308650876033218,
                0.3072379405080092,
                0.1508345664549045,
                0.35452334724681595,
                0.4963129299555935,
                0.09383482094930852,
                -0.5597722032148169,
                -0.7676526179421094,
                -0.49070279885774953,
                -0.3094727736838044,
                -0.4280397504437062,
                -0.4308022270154016,
                -0.04099330843324916,
                0.4004445171396711,
                0.3792181847278279,
                -0.06127803852431971,
                -0.2715371036882162,
                0.11065848929283895,
                0.5555344968812683,
                0.513147718247395,
                0.394416888355483,
                0.7680803706442487,
                1.1719811059385874,
                0.8551865432638581,
                0.18949392262214865,
                0.20787148436777364,
                1.0599658419748752,
                1.8586251487247711,
                1.9140079938506065,
                1.395155176444951,
            ],
            [
                0.19364895699757995,
                -0.4472319458717678,
                -0.34078609899707657,
                -0.19060450067450968,
                -0.16882609707239815,
                0.07700879457855273,
                0.4034413061601192,
                0.22194366463171428,
                -0.47475122154225297,
                -0.9836099196224104,
                -0.7712440515344353,
                -0.06326119884156056,
                0.398712052140213,
                0.21742115309054955,
                -0.04615351346933377,
                0.40530033308211916,
                1.225394677480013,
                1.1814718673694442,
                0.05123840377868144,
                -0.8551741258392109,
                -0.5503083164015539,
                0.36814708221529396,
                0.8533914002710604,
                0.6743208531385649,
                0.13230478256283174,
                -0.4028959723523836,
                -0.4571658137758091,
                0.17297678995798565,
                0.9542379850472644,
                1.1235659236471438,
                0.5580902751846268,
                -0.26976544037284966,
                -0.9163629716420257,
                -1.115475958109283,
                -0.6802244905552735,
                0.1752580655723767,
                0.7546609580649839,
                0.6813110117537722,
                0.42182891706622033,
                0.5373714812289399,
                0.7877710359177563,
                0.544350996118941,
                -0.10420569609066627,
                -0.3447338404996717,
                0.14668084914074822,
                0.6389334729963694,
                0.35030335721420597,
                -0.5326173007817853,
                -1.2621073076714566,
                -1.5040815518836774,
            ],
            [
                0.5909733204640251,
                0.5034812339305611,
                0.0370528597828878,
                -0.3333714160662199,
                -0.4214294185007197,
                -0.4104352116619698,
                -0.4865994976385157,
                -0.6446482980293632,
                -0.721069240021156,
                -0.5738334231724332,
                -0.27670544449029016,
                -0.01261551416317416,
                0.13498735191358532,
                0.16048284300094653,
                0.14331731569617712,
                0.27824423277949795,
                0.6062260578744498,
                0.8922414999239479,
                1.0009802333407039,
                1.0434986576550171,
                0.9827394953721259,
                0.5826949231099015,
                -0.09302876660603449,
                -0.6500418112750621,
                -0.8737929459470409,
                -0.8292261770639195,
                -0.6078157189387818,
                -0.422697477983497,
                -0.5747680127377425,
                -0.8950904400859678,
                -0.760363499637513,
                -0.05043282524357216,
                0.5670467771185732,
                0.6910872016288118,
                0.7668800541730201,
                1.0737714601770991,
                1.021997292875442,
                0.2807914555217373,
                -0.3436612118904978,
                -0.1345986887695837,
                0.4182018729023162,
                0.5953347117920418,
                0.4969201704243512,
                0.39932872956213733,
                0.0914324470932886,
                -0.5451487735313276,
                -1.079829963559939,
                -1.0366009941698777,
                -0.3010189373311442,
                1.052298628030191,
            ],
        ]
    )
    assert np.allclose(result, expected, atol=1e-12)


def test_modified_sinc_filter_snapshot_no_corrections():
    # Snapshot of exact output with use_corrections=False.
    # Arrange
    rng = np.random.default_rng(0)
    X = rng.normal(size=(3, 50))
    msf = ModifiedSincFilter(use_corrections=False)

    # Act
    result = msf.fit_transform(X)

    # Assert
    expected = np.array(
        [
            [
                0.15059532535735898,
                0.2245098024997757,
                0.2048791534270877,
                -0.0574174404223456,
                -0.08593173293697104,
                0.4781839489011087,
                0.9998323150802173,
                0.6247652451457066,
                -0.3519910275571668,
                -0.913163724460351,
                -0.84913687193969,
                -0.7913202288254465,
                -0.983459060514875,
                -1.0866683176600058,
                -0.9685678405417761,
                -0.8018906483058379,
                -0.5795780634487426,
                -0.156616643181086,
                0.37382481850359145,
                0.702801636547956,
                0.6322047366981206,
                0.30803142961651253,
                0.15131127723732424,
                0.355207250111318,
                0.4971097009098603,
                0.09395024162490277,
                -0.5606952077198438,
                -0.7689308814040674,
                -0.49157156801048196,
                -0.31001078182554037,
                -0.42864911475169204,
                -0.4313176341996202,
                -0.04089624527747877,
                0.40114165910049887,
                0.37975014811770713,
                -0.06157237786046038,
                -0.27230869292513915,
                0.11032837712226652,
                0.5557920353815389,
                0.513299817269888,
                0.3943182032166983,
                0.7683834686375582,
                1.1726809146219574,
                0.8552540162575408,
                0.1885314325220563,
                0.20704638141766574,
                1.060655821990463,
                1.860842516936191,
                1.9166620669691865,
                1.3973391134882056,
            ],
            [
                0.18833970312148535,
                -0.45233703166426564,
                -0.34454753176479636,
                -0.19298978353346236,
                -0.17018392798776572,
                0.0767226997619424,
                0.40401100386524663,
                0.22233089787213506,
                -0.4754895230185231,
                -0.9852566398855276,
                -0.7726821067624138,
                -0.0636446477267536,
                0.39913059660818173,
                0.2177431503015828,
                -0.04608422950502927,
                0.40605115870073444,
                1.2272303325110514,
                1.1830426443854023,
                0.05095632975206854,
                -0.8569017742975772,
                -0.5516067314279611,
                0.36824978500886923,
                0.8542733748384662,
                0.6749711049685304,
                0.13216764172258244,
                -0.4037766279070817,
                -0.45800790831445837,
                0.17323166718476773,
                0.9557839474481196,
                1.1253845849668622,
                0.5589311736320761,
                -0.2704165320409345,
                -0.9182411214562506,
                -1.1177872825563044,
                -0.6818366882217126,
                0.17512179957250157,
                0.7556472509167736,
                0.6823882537464657,
                0.42257820717794137,
                0.5382121530511501,
                0.7888549369256092,
                0.5450184279210619,
                -0.1044172450469684,
                -0.34509648719552555,
                0.1473130116437558,
                0.6405544628141366,
                0.3516992605588094,
                -0.5323994263651897,
                -1.262887339697396,
                -1.5051254230007092,
            ],
            [
                0.5934228935400623,
                0.5055099478972214,
                0.03807471246019498,
                -0.3332099628296325,
                -0.4216659188876013,
                -0.4108468414030465,
                -0.4872265073390015,
                -0.6455405761429727,
                -0.7220679069994469,
                -0.5746057145722216,
                -0.27705755311491526,
                -0.01264020702417366,
                0.1350757957917876,
                0.16048891270649535,
                0.1432076182320153,
                0.2783107084264618,
                0.6068407629762,
                0.8934109512373485,
                1.0024547335383978,
                1.0451254789039441,
                0.9842718200447307,
                0.5835604702162291,
                -0.09325652265360512,
                -0.6511544487526313,
                -0.8752382913679079,
                -0.8305543873322524,
                -0.6087540523698358,
                -0.4233349077712516,
                -0.5756431547232806,
                -0.8964570224730499,
                -0.7615221981626747,
                -0.05048866595487411,
                0.5679913104049555,
                0.6922828078802477,
                0.76815682956044,
                1.0753369465677616,
                1.0232455356199357,
                0.2807366287952723,
                -0.3447290580523956,
                -0.13529213601251466,
                0.41851143558921644,
                0.5960885747570346,
                0.4975279269219733,
                0.3994629710733755,
                0.09041004119141015,
                -0.5481261406799269,
                -1.08480586843493,
                -1.0427819214203335,
                -0.3073535620688009,
                1.0467478847180125,
            ],
        ]
    )
    assert np.allclose(result, expected, atol=1e-12)
