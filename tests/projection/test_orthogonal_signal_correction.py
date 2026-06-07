"""Tests for OrthogonalSignalCorrection."""

import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning, NotFittedError
from sklearn.utils.estimator_checks import check_estimator

import chemotools.projection._orthogonal_signal_correction as osc_module
from chemotools.projection import OrthogonalSignalCorrection


def _make_osc_dataset(
    *,
    n_samples: int = 40,
    n_features: int = 12,
    signal_scale: float = 3.0,
    nuisance_scale: float = 8.0,
    noise_scale: float = 1e-2,
    seed: int = 0,
):
    """Create a stable synthetic dataset with signal and nuisance variation."""
    rng = np.random.default_rng(seed)

    y = np.linspace(-1.0, 1.0, n_samples)
    y_centered = y - y.mean()

    nuisance_scores = rng.normal(size=n_samples)
    nuisance_scores -= y_centered * (
        (y_centered @ nuisance_scores) / (y_centered @ y_centered)
    )
    nuisance_scores /= np.linalg.norm(nuisance_scores)

    signal_vector = rng.normal(size=n_features)
    signal_vector /= np.linalg.norm(signal_vector)

    nuisance_vector = rng.normal(size=n_features)
    nuisance_vector -= signal_vector * (signal_vector @ nuisance_vector)
    nuisance_vector /= np.linalg.norm(nuisance_vector)

    X = (
        signal_scale * np.outer(y_centered, signal_vector)
        + nuisance_scale * np.outer(nuisance_scores, nuisance_vector)
        + noise_scale * rng.normal(size=(n_samples, n_features))
    )
    y_multi = np.column_stack([y, y**2])

    return X, y, y_multi, nuisance_vector


# Test compliance with scikit-learn
def test_compliance_osc():
    """Check sklearn estimator compliance for the OSC transformer."""
    # Arrange
    transformer = OrthogonalSignalCorrection()

    # Act & Assert
    # n_iter_ is per-component (array), but sklearn's check_transformer_n_iter
    # only handles this for its own hardcoded CROSS_DECOMPOSITION list.
    check_estimator(
        transformer,
        expected_failed_checks={
            "check_transformer_n_iter": "n_iter_ is stored per component (array) "
            "instead of scalar"
        },
    )


@pytest.mark.parametrize(
    ("method", "n_components"),
    [("wold", 2), ("sjoblom", 2), ("fearn", 2)],
)
@pytest.mark.parametrize(
    "y_factory",
    [
        lambda rng, n_samples: rng.normal(size=n_samples),
        lambda rng, n_samples: rng.normal(size=(n_samples, 2)),
    ],
)
def test_osc_methods_preserve_expected_shapes(method, n_components, y_factory):
    """Ensure each OSC method produces finite outputs with expected shapes."""
    # Arrange
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 6))
    y = y_factory(rng, X.shape[0])
    transformer = OrthogonalSignalCorrection(
        method=method,
        n_components=n_components,
    )

    # Act
    Xt = transformer.fit_transform(X, y)

    # Assert
    assert Xt.shape == X.shape
    assert transformer.scores_.shape == (X.shape[0], n_components)
    assert transformer.weights_.shape == (X.shape[1], n_components)
    assert transformer.loadings_.shape == (X.shape[1], n_components)
    assert transformer.n_iter_.shape == (n_components,)
    assert np.isfinite(transformer.scores_).all()
    assert np.isfinite(transformer.weights_).all()
    assert np.isfinite(transformer.loadings_).all()
    assert np.isfinite(transformer.n_iter_).all()


@pytest.mark.parametrize(
    ("method", "n_components"),
    [("wold", 2), ("sjoblom", 2), ("fearn", 2)],
)
def test_fit_transform_matches_fit_then_transform(method, n_components):
    """Verify `fit_transform()` matches `fit()` followed by `transform()`."""
    # Arrange
    X, y, _, _ = _make_osc_dataset()

    # Act
    fit_transformer = OrthogonalSignalCorrection(
        method=method,
        n_components=n_components,
    )
    Xt_fit_transform = fit_transformer.fit_transform(X, y)

    transformer = OrthogonalSignalCorrection(
        method=method,
        n_components=n_components,
    )
    transformer.fit(X, y)
    Xt_transform = transformer.transform(X)

    # Assert
    np.testing.assert_allclose(Xt_fit_transform, Xt_transform)


def test_transform_before_fit_raises_not_fitted_error():
    """Ensure calling `transform()` before `fit()` raises `NotFittedError`."""
    # Arrange
    X = np.ones((4, 3), dtype=float)
    transformer = OrthogonalSignalCorrection()

    # Act / Assert
    with pytest.raises(NotFittedError):
        transformer.transform(X)


def test_fit_rejects_single_sample():
    """Reject datasets with fewer than two samples."""
    # Arrange
    X = np.array([[1.0, 2.0, 3.0]])
    y = np.array([1.0])
    transformer = OrthogonalSignalCorrection()

    # Act / Assert
    with pytest.raises(ValueError, match="At least 2 samples are required"):
        transformer.fit(X, y)


@pytest.mark.parametrize("method", ["wold", "sjoblom"])
def test_iterative_methods_warn_when_not_converged(method):
    """Emit a convergence warning when iterative OSC methods hit `max_iter`."""
    # Arrange
    X, y, _, _ = _make_osc_dataset()
    transformer = OrthogonalSignalCorrection(
        method=method,
        n_components=1,
        max_iter=1,
        tol=0.0,
    )

    # Act / Assert
    with pytest.warns(ConvergenceWarning, match="did not converge"):
        transformer.fit(X, y)

    assert transformer.n_iter_.shape == (1,)
    assert transformer.n_iter_[0] == 1


@pytest.mark.parametrize("method", ["wold", "sjoblom"])
def test_iterative_methods_scores_are_approximately_orthogonal_to_centered_y(method):
    """Learn scores that are nearly orthogonal to the centered target."""
    # Arrange
    X, y, _, _ = _make_osc_dataset(
        signal_scale=1.0,
        nuisance_scale=20.0,
        noise_scale=1e-6,
    )
    transformer = OrthogonalSignalCorrection(method=method, n_components=1)

    # Act
    transformer.fit(X, y)
    score = transformer.scores_[:, 0]
    y_centered = y - y.mean()
    relative_projection = abs(score @ y_centered) / (
        np.linalg.norm(score) * np.linalg.norm(y_centered)
    )

    # Assert
    assert relative_projection < 1e-6


@pytest.mark.parametrize("method", ["wold", "sjoblom", "fearn"])
def test_transform_reduces_known_orthogonal_variation(method):
    """Reduce variance along a known nuisance direction after OSC transform."""
    # Arrange
    X, y, _, nuisance_vector = _make_osc_dataset()
    n_components = 2 if method == "fearn" else 1
    transformer = OrthogonalSignalCorrection(method=method, n_components=n_components)
    X_centered = X - X.mean(axis=0)
    before = np.std(X_centered @ nuisance_vector)

    # Act
    Xt = transformer.fit_transform(X, y)
    Xt_centered = Xt - Xt.mean(axis=0)
    after = np.std(Xt_centered @ nuisance_vector)

    # Assert
    assert after < before * 0.25


def test_fearn_supports_multiple_components():
    """Allow Fearn OSC to extract more than one orthogonal component."""
    # Arrange
    X, y, _, _ = _make_osc_dataset()
    transformer = OrthogonalSignalCorrection(method="fearn", n_components=2)

    # Act
    Xt = transformer.fit_transform(X, y)

    # Assert
    assert Xt.shape == X.shape
    assert transformer.scores_.shape == (X.shape[0], 2)
    assert transformer.weights_.shape == (X.shape[1], 2)
    assert transformer.loadings_.shape == (X.shape[1], 2)
    np.testing.assert_array_equal(transformer.n_iter_, np.array([1, 1]))


def test_wold_raises_for_zero_norm_orthogonal_score_vector():
    """Raise when Wold encounters an initially zero orthogonal score vector."""
    # Arrange
    X = np.zeros((6, 4), dtype=float)
    y = np.linspace(0.0, 1.0, 6)
    transformer = OrthogonalSignalCorrection(method="wold", n_components=1)

    # Act / Assert
    with pytest.raises(ValueError, match="zero-norm orthogonal score vector"):
        transformer._wold_method(X, y)


def test_wold_raises_for_zero_norm_weight_vector(monkeypatch):
    """Raise when Wold computes a zero-norm weight vector inside the loop."""
    # Arrange
    X, y, _, _ = _make_osc_dataset()
    transformer = OrthogonalSignalCorrection(method="wold", n_components=1)
    original_norm = osc_module.np.linalg.norm

    def fake_norm(x, *args, **kwargs):
        arr = np.asarray(x)
        if arr.ndim == 1 and arr.shape[0] == X.shape[1]:
            return 0.0
        return original_norm(x, *args, **kwargs)

    monkeypatch.setattr(osc_module.np.linalg, "norm", fake_norm)

    # Act / Assert
    with pytest.raises(ValueError, match="zero-norm weight vector"):
        transformer._wold_method(X, y)


def test_wold_raises_for_zero_norm_orthogonal_score_after_convergence(monkeypatch):
    """Raise when Wold's final orthogonal score degenerates after convergence."""
    # Arrange
    X, y, _, _ = _make_osc_dataset()
    transformer = OrthogonalSignalCorrection(method="wold", n_components=1, tol=np.inf)
    original_isclose = osc_module.np.isclose
    scalar_calls = {"count": 0}

    def fake_isclose(a, b, *args, **kwargs):
        if np.isscalar(a) and np.isscalar(b):
            scalar_calls["count"] += 1
            if scalar_calls["count"] == 3:
                return True
        return original_isclose(a, b, *args, **kwargs)

    monkeypatch.setattr(osc_module.np, "isclose", fake_isclose)

    # Act / Assert
    with pytest.raises(
        ValueError,
        match="zero-norm orthogonal score vector after convergence",
    ):
        transformer._wold_method(X, y)


def test_wold_raises_for_zero_norm_weight_after_convergence(monkeypatch):
    """Raise when Wold's final weight vector degenerates after convergence."""
    # Arrange
    X, y, _, _ = _make_osc_dataset()
    transformer = OrthogonalSignalCorrection(method="wold", n_components=1, tol=np.inf)
    original_isclose = osc_module.np.isclose
    scalar_calls = {"count": 0}

    def fake_isclose(a, b, *args, **kwargs):
        if np.isscalar(a) and np.isscalar(b):
            scalar_calls["count"] += 1
            if scalar_calls["count"] == 4:
                return True
        return original_isclose(a, b, *args, **kwargs)

    monkeypatch.setattr(osc_module.np, "isclose", fake_isclose)

    # Act / Assert
    with pytest.raises(
        ValueError,
        match="zero-norm weight vector after convergence",
    ):
        transformer._wold_method(X, y)


def test_sjoblom_raises_for_zero_norm_orthogonal_score_vector():
    """Raise when Sjöblom encounters an initially zero orthogonal score vector."""
    # Arrange
    X = np.zeros((6, 4), dtype=float)
    y = np.linspace(0.0, 1.0, 6)
    transformer = OrthogonalSignalCorrection(method="sjoblom", n_components=1)

    # Act / Assert
    with pytest.raises(ValueError, match="zero-norm orthogonal score vector"):
        transformer._sjoblom_method(X, y)


def test_sjoblom_raises_for_zero_norm_weight_vector(monkeypatch):
    """Raise when Sjöblom computes a zero-norm weight vector inside the loop."""
    # Arrange
    X, y, _, _ = _make_osc_dataset()
    transformer = OrthogonalSignalCorrection(method="sjoblom", n_components=1)
    original_norm = osc_module.np.linalg.norm

    def fake_norm(x, *args, **kwargs):
        arr = np.asarray(x)
        if arr.ndim == 1 and arr.shape[0] == X.shape[1]:
            return 0.0
        return original_norm(x, *args, **kwargs)

    monkeypatch.setattr(osc_module.np.linalg, "norm", fake_norm)

    # Act / Assert
    with pytest.raises(ValueError, match="zero-norm weight vector"):
        transformer._sjoblom_method(X, y)


def test_sjoblom_raises_for_zero_norm_final_score_vector(monkeypatch):
    """Raise when Sjöblom's final score vector degenerates after projection."""
    # Arrange
    X, y, _, _ = _make_osc_dataset()
    transformer = OrthogonalSignalCorrection(
        method="sjoblom",
        n_components=1,
        tol=np.inf,
    )
    original_isclose = osc_module.np.isclose
    scalar_calls = {"count": 0}

    def fake_isclose(a, b, *args, **kwargs):
        if np.isscalar(a) and np.isscalar(b):
            scalar_calls["count"] += 1
            if scalar_calls["count"] == 3:
                return True
        return original_isclose(a, b, *args, **kwargs)

    monkeypatch.setattr(osc_module.np, "isclose", fake_isclose)

    # Act / Assert
    with pytest.raises(ValueError, match="zero-norm final score vector"):
        transformer._sjoblom_method(X, y)


def test_fearn_raises_for_zero_norm_weight_vector(monkeypatch):
    """Raise when Fearn's SVD returns a zero-norm weight vector."""
    # Arrange
    X, y, _, _ = _make_osc_dataset()
    transformer = OrthogonalSignalCorrection(method="fearn", n_components=1)

    def fake_svd(z, full_matrices=False):
        u = np.zeros((z.shape[0], min(z.shape)))
        s = np.zeros(min(z.shape))
        vt = np.zeros((min(z.shape), z.shape[1]))
        return u, s, vt

    monkeypatch.setattr(osc_module, "svd", fake_svd)

    # Act / Assert
    with pytest.raises(ValueError, match="zero-norm weight vector"):
        transformer._fearn_method(X, y)


def test_fearn_raises_for_zero_norm_score_vector():
    """Raise when Fearn produces zero scores from degenerate input data."""
    # Arrange
    X = np.zeros((6, 4), dtype=float)
    y = np.linspace(0.0, 1.0, 6)
    transformer = OrthogonalSignalCorrection(method="fearn", n_components=1)

    # Act / Assert
    with pytest.raises(ValueError, match="zero-norm score vector"):
        transformer._fearn_method(X, y)


def test_raises_error_for_invalid_method():
    """Raise when an invalid method name is provided."""
    # Arrange
    X, y, _, _ = _make_osc_dataset()
    transformer = OrthogonalSignalCorrection(method="invalid_method", n_components=1)

    # Act / Assert
    with pytest.raises(
        ValueError, match="The 'method' parameter of OrthogonalSignalCorrection"
    ):
        transformer.fit(X, y)


@pytest.mark.parametrize("method", ["wold", "sjoblom", "fearn"])
def test_osc_has_correct_variance_ratios(method):
    """Test that OrthogonalSignalCorrection has correct variance ratios."""
    # Arrange
    X, y, _, _ = _make_osc_dataset()
    transformer = OrthogonalSignalCorrection(method=method, n_components=1)
    transformer.fit(X, y)

    # Act
    _ = transformer.transform(X, y)

    # Assert
    assert 0.0 <= transformer.retained_variance_ratio_ <= 1.0
    assert 0.0 <= transformer.removed_variance_ratio_ <= 1.0
    assert np.isclose(
        transformer.retained_variance_ratio_ + transformer.removed_variance_ratio_, 1.0
    )


def test_orthogonal_signal_correction_snapshot_wold():
    # Snapshot of exact output for method='wold', n_components=1.
    # Arrange
    rng = np.random.default_rng(0)
    X = rng.normal(size=(5, 20))
    y = rng.normal(size=5)
    osc = OrthogonalSignalCorrection(n_components=1, method="wold")

    # Act
    result = osc.fit_transform(X, y)

    # Assert
    expected = np.array(
        [
            [
                -0.06347351285252828,
                0.11160469075244239,
                0.7268494499086522,
                0.18720018013306733,
                -0.6525895940412725,
                0.3332627501105291,
                1.453151839905149,
                1.3147157575700639,
                -0.36718359151224844,
                -0.6015624314150045,
                -0.43452255426880326,
                -0.1333947418236929,
                -2.263459802675753,
                -0.11803519245208735,
                -1.5839466857952926,
                -0.4337723628454729,
                -0.2810016643577612,
                -0.2747301941479937,
                -0.02714237108173384,
                0.39867317931318247,
            ],
            [
                -0.3415790039075124,
                1.640881634741904,
                -0.5678776694842232,
                0.4441803483859638,
                0.7718174341968461,
                0.06210998231505397,
                -0.5755535901500394,
                -0.5077667779610159,
                -0.0787670082930362,
                0.9677036881333039,
                -0.7970826005244732,
                -0.40591197176103133,
                -0.0898957907136313,
                0.6542978703076922,
                -0.16597079233675274,
                0.69147954739364,
                -0.3573995656893228,
                -0.0828056504644089,
                0.28991500438213574,
                0.7684639041876433,
            ],
            [
                -0.7387215439473088,
                0.8436790673101615,
                1.1081863223066777,
                0.5549715600246382,
                0.5860070752196438,
                -0.23600392696850744,
                1.0478266661389797,
                0.9491971120277344,
                0.8760578679708372,
                -0.5106269069145464,
                -0.16172097636090899,
                -0.7278055104126806,
                -0.173785279169865,
                0.3793766817314268,
                -0.35870293817906884,
                -0.42579236507246054,
                -0.2941408486540427,
                0.5817179192744509,
                0.02258572665935793,
                1.108972695730134,
            ],
            [
                -0.7992535108095852,
                -0.702462942558107,
                1.9051004483764773,
                -0.3380916128924523,
                0.10476268257535497,
                -0.31290275037231885,
                1.869487293393215,
                2.0253397684228234,
                1.2787261610235832,
                -0.930489633873933,
                0.4139808141811935,
                0.3486406429973725,
                1.1220304567992465,
                -0.42469580477436575,
                1.1737919167178104,
                -0.7480351065681695,
                -0.15670414121057252,
                1.0147648076855469,
                -0.792339057995568,
                0.7677606078927467,
            ],
            [
                0.4332415429314178,
                -0.9484160662140451,
                -0.48935077847128505,
                -1.1975957337660774,
                -1.126451696056977,
                0.6670574283434827,
                0.3882479612409108,
                0.819047849784138,
                -1.1899127892323587,
                0.8304502730255688,
                -0.5315256519929199,
                1.8003978228145125,
                -0.5124237732538197,
                -0.865805036032944,
                0.6870119459840155,
                0.6453698164671194,
                -0.17949612264159454,
                -0.6392967844559652,
                -0.7736963625372755,
                -0.5687560757714759,
            ],
        ]
    )
    assert np.allclose(result, expected, atol=1e-12)


def test_orthogonal_signal_correction_snapshot_sjoblom():
    # Snapshot of exact output for method='sjoblom', n_components=1.
    # Arrange
    rng = np.random.default_rng(0)
    X = rng.normal(size=(5, 20))
    y = rng.normal(size=5)
    osc = OrthogonalSignalCorrection(n_components=1, method="sjoblom")

    # Act
    result = osc.fit_transform(X, y)

    # Assert
    expected = np.array(
        [
            [
                -0.02991294882597212,
                0.14331106962895657,
                0.639514348905148,
                0.18313171360853925,
                -0.6698963872712741,
                0.36030662582201045,
                1.38687848135817,
                1.2516836435080203,
                -0.4413510758432565,
                -0.5379290183309986,
                -0.47365325044251855,
                -0.11699311243369334,
                -2.324563981799465,
                -0.11018477388065362,
                -1.6220146612771706,
                -0.384150402482715,
                -0.28476763531880284,
                -0.3277557777617675,
                -0.00860151802902559,
                0.37087571744176195,
            ],
            [
                -0.3037898270328667,
                1.6765835004978369,
                -0.6662174265903335,
                0.4395993275968532,
                0.75232974037695,
                0.09256151870478675,
                -0.650177621761061,
                -0.5787408896321359,
                -0.1622796520088469,
                1.0393560878735417,
                -0.8411437809007342,
                -0.3874438433915436,
                -0.1586993983139076,
                0.6631376091098997,
                -0.2088359666123291,
                0.7473545326892133,
                -0.3616397602959922,
                -0.14251273365290285,
                0.31079159898001124,
                0.7371630290506775,
            ],
            [
                -0.8310192682377291,
                0.7564812145748789,
                1.3483734137683978,
                0.5661606645876357,
                0.6336036753708097,
                -0.310379406512575,
                1.2300903156263236,
                1.1225470351430307,
                1.0800319917675556,
                -0.6856292358656841,
                -0.05410440339060585,
                -0.7729131096970232,
                -0.00573789604392394,
                0.3577867655861813,
                -0.2540096852699941,
                -0.5622611700865574,
                -0.28378343712103377,
                0.7275477102717202,
                -0.02840544215774526,
                1.1854198411059125,
            ],
            [
                -0.7348974858923332,
                -0.6416627291044914,
                1.7376261128710841,
                -0.3458933686625863,
                0.0715751343938086,
                -0.2610432445101636,
                1.7424010452835637,
                1.9044688407186525,
                1.136501998641747,
                -0.8084661648002935,
                0.3389434720134018,
                0.38009258885911834,
                1.0048567004784918,
                -0.40964184704258433,
                1.100792715907781,
                -0.6528798981599869,
                -0.16392591976030307,
                0.9130826225760881,
                -0.7567847931739409,
                0.7144563552309758,
            ],
            [
                0.38983350140338435,
                -0.989426671564825,
                -0.3763886763179979,
                -1.1923335952453025,
                -1.104066260976699,
                0.6320779899241808,
                0.4739679500212187,
                0.9005750801061763,
                -1.093982622600422,
                0.7481433200788238,
                -0.4809130062454551,
                1.779183718477622,
                -0.43338961333501774,
                -0.8759592349931211,
                0.7362510436424244,
                0.5811864674147027,
                -0.17462559005716177,
                -0.5707117235415079,
                -0.7976769061923836,
                -0.5328006314770973,
            ],
        ]
    )
    assert np.allclose(result, expected, atol=1e-12)


def test_orthogonal_signal_correction_snapshot_fearn():
    # Snapshot of exact output for method='fearn', n_components=1.
    # Arrange
    rng = np.random.default_rng(0)
    X = rng.normal(size=(5, 20))
    y = rng.normal(size=5)
    osc = OrthogonalSignalCorrection(n_components=1, method="fearn")

    # Act
    result = osc.fit_transform(X, y)

    # Assert
    expected = np.array(
        [
            [
                -0.07880463679661476,
                0.13952023989930407,
                0.7356626224334437,
                0.2033600246458646,
                -0.6518003451726295,
                0.3286624707220186,
                1.4626240608079328,
                1.3335817918333353,
                -0.3412735291501985,
                -0.566867733964542,
                -0.42435795055854275,
                -0.1590660483628362,
                -2.2673646637194036,
                -0.10379554288670026,
                -1.6175732866000954,
                -0.4197994663868513,
                -0.26639264666131196,
                -0.2698095382964756,
                -0.04629566945163255,
                0.36833494545589907,
            ],
            [
                -0.39740337369218115,
                1.7235248147874709,
                -0.5399981727783824,
                0.48093929629471593,
                0.7508116810677304,
                0.05072118570578162,
                -0.5349820556150376,
                -0.4136555998458168,
                0.01874362687357242,
                1.1384701041504017,
                -0.748134998354615,
                -0.4725983805999694,
                -0.08342075309139446,
                0.692012281285081,
                -0.2739049053798712,
                0.7661234199385966,
                -0.28856293126432325,
                -0.06849997965826886,
                0.18201436744881055,
                0.6071984241017954,
            ],
            [
                -0.6911500274057787,
                0.7597241768876883,
                1.0814302464749947,
                0.5079256872177412,
                0.5869071687193887,
                -0.2224815542897108,
                1.0175821252870116,
                0.8870925222954454,
                0.7952165742594498,
                -0.6245142653957361,
                -0.19493508440462648,
                -0.6519061912512403,
                -0.16457098960432454,
                0.3371744461727604,
                -0.25639651066002866,
                -0.47248241724538753,
                -0.34166543320684756,
                0.5669559604700568,
                0.08736893273252783,
                1.2102344848761328,
            ],
            [
                -0.7161597115692364,
                -0.7983239734076961,
                1.8696192701285015,
                -0.3612557108329949,
                0.17014744000893622,
                -0.3036115650769179,
                1.800409093432278,
                1.8489444198754972,
                1.129059847575989,
                -1.2481589346333415,
                0.3240697175970727,
                0.4096275134209635,
                1.082826480137082,
                -0.4606368918228696,
                1.3137212366787339,
                -0.8930959249683439,
                -0.2815144980752068,
                0.9986311467261544,
                -0.577211070140077,
                1.0803776248127082,
            ],
            [
                0.3737317208782943,
                -0.8791588741344114,
                -0.46380619362225894,
                -1.1803045554401874,
                -1.1725200427298306,
                0.660232946367068,
                0.4375269466160303,
                0.9445705756852822,
                -1.0828258796020356,
                1.0565458187986068,
                -0.4675126532452004,
                1.7558693486075627,
                -0.4850042627357819,
                -0.8396157739685496,
                0.5863369123519729,
                0.7485039180366427,
                -0.09060683334560427,
                -0.6276274913498369,
                -0.9265536211627128,
                -0.791031167894305,
            ],
        ]
    )
    assert np.allclose(result, expected, atol=1e-12)
