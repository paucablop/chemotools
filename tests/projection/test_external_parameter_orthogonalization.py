"""Tests for ExternalParameterOrthogonalization."""

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.utils.estimator_checks import check_estimator

from chemotools.projection import ExternalParameterOrthogonalization

# --- Fixtures ---


@pytest.fixture()
def epo_dataset():
    """Create a dataset with known signal and external nuisance directions."""
    rng = np.random.default_rng(0)
    n_samples, n_features = 40, 10

    signal_scores = rng.normal(size=n_samples)
    nuisance_scores = rng.normal(size=n_samples)

    signal_vector = rng.normal(size=n_features)
    signal_vector /= np.linalg.norm(signal_vector)

    nuisance_vector = rng.normal(size=n_features)
    nuisance_vector -= signal_vector * (signal_vector @ nuisance_vector)
    nuisance_vector /= np.linalg.norm(nuisance_vector)

    X = (
        2.0 * np.outer(signal_scores, signal_vector)
        + 1.5 * np.outer(nuisance_scores, nuisance_vector)
        + 1e-3 * rng.normal(size=(n_samples, n_features))
    )
    X_external = (
        2.0 * np.outer(signal_scores, signal_vector)
        + 8.0 * np.outer(nuisance_scores, nuisance_vector)
        + 1e-3 * rng.normal(size=(n_samples, n_features))
    )

    return X, X_external, nuisance_vector


@pytest.fixture()
def paired_epo_dataset():
    """Create a paired-sample dataset with known nuisance direction."""
    rng = np.random.default_rng(0)
    n_samples, n_features = 12, 8
    sample_ids = np.repeat(np.arange(n_samples), 2)
    condition = np.tile(np.array([-1.0, 1.0]), n_samples)

    signal_vector = rng.normal(size=n_features)
    signal_vector /= np.linalg.norm(signal_vector)
    nuisance_vector = rng.normal(size=n_features)
    nuisance_vector -= signal_vector * (signal_vector @ nuisance_vector)
    nuisance_vector /= np.linalg.norm(nuisance_vector)

    latent_signal = rng.normal(size=n_samples)
    base = 3.0 * np.outer(latent_signal, signal_vector)
    base = np.repeat(base, 2, axis=0)

    X = base + 0.5 * np.outer(condition, nuisance_vector)
    X_external = base + 4.0 * np.outer(condition, nuisance_vector)

    return X, X_external, sample_ids, nuisance_vector


# Test compliance with scikit-learn


def test_compliance_epo():
    """Check sklearn estimator compliance for the EPO transformer."""
    # Arrange
    transformer = ExternalParameterOrthogonalization()

    # Act & Assert
    check_estimator(transformer)


# Test functionality


class TestEPOFitTransform:
    """Test basic fit / transform behaviour."""

    def test_fit_transform_matches_fit_then_transform(self, epo_dataset):
        """Ensure `fit_transform()` matches `fit()` followed by `transform()`."""
        # Arrange
        X, X_external, _ = epo_dataset

        # Act
        fit_transformer = ExternalParameterOrthogonalization(n_components=1)
        Xt_fit_transform = fit_transformer.fit_transform(X, X_external=X_external)

        transformer = ExternalParameterOrthogonalization(n_components=1)
        transformer.fit(X, X_external=X_external)
        Xt_transform = transformer.transform(X)

        # Assert
        np.testing.assert_allclose(Xt_fit_transform, Xt_transform)

    def test_transform_before_fit_raises_not_fitted_error(self):
        """Ensure calling `transform()` before `fit()` raises `NotFittedError`."""
        # Arrange
        X = np.ones((4, 3), dtype=float)
        transformer = ExternalParameterOrthogonalization()

        # Act / Assert
        with pytest.raises(NotFittedError):
            transformer.transform(X)

    def test_fit_without_external_leaves_data_unchanged(self):
        """Default to a no-op transform when `X_external` is not provided."""
        # Arrange
        rng = np.random.default_rng(0)
        X = rng.normal(size=(8, 5))
        transformer = ExternalParameterOrthogonalization(n_components=1)

        # Act
        Xt = transformer.fit_transform(X)

        # Assert
        np.testing.assert_allclose(Xt, X)
        assert transformer.V_epo_.shape == (X.shape[1], 0)
        np.testing.assert_allclose(transformer.mean_X_, X.mean(axis=0))


class TestEPONuisanceRemoval:
    """Test that EPO effectively removes nuisance variation."""

    def test_transform_reduces_known_nuisance_variation(self, epo_dataset):
        """Reduce variance along a known nuisance direction after EPO transform."""
        # Arrange
        X, X_external, nuisance_vector = epo_dataset
        transformer = ExternalParameterOrthogonalization(n_components=1)
        X_centered = X - X.mean(axis=0)
        before = np.std(X_centered @ nuisance_vector)

        # Act
        Xt = transformer.fit_transform(X, X_external=X_external)
        Xt_centered = Xt - Xt.mean(axis=0)
        after = np.std(Xt_centered @ nuisance_vector)

        # Assert
        assert after < before * 0.25

    def test_sample_ids_reduce_within_sample_external_differences(
        self, paired_epo_dataset
    ):
        """Use `sample_ids` to suppress paired external-condition differences."""
        # Arrange
        X, X_external, sample_ids, nuisance_vector = paired_epo_dataset
        transformer = ExternalParameterOrthogonalization(n_components=1)
        before = np.mean(
            [abs((X[i] - X[i + 1]) @ nuisance_vector) for i in range(0, len(X), 2)]
        )

        # Act
        Xt = transformer.fit_transform(X, X_external=X_external, sample_ids=sample_ids)
        after = np.mean(
            [abs((Xt[i] - Xt[i + 1]) @ nuisance_vector) for i in range(0, len(Xt), 2)]
        )

        # Assert
        assert after < before * 0.1


def test_external_parameter_orthogonalization_snapshot_n_components_1():
    # Snapshot of exact output for n_components=1.
    # Arrange
    rng = np.random.default_rng(0)
    X = rng.normal(size=(3, 50))
    X_external = X + 0.5 * rng.normal(size=(3, 50))
    epo = ExternalParameterOrthogonalization(n_components=1)

    # Act
    result = epo.fit_transform(X, X_external=X_external)

    # Assert
    expected = np.array(
        [
            [
                2.6208226824967396e-01,
                -4.0131560695365259e-01,
                3.2839659830812296e-01,
                2.2617083147040523e-01,
                -2.9577294135786314e-01,
                4.4912817405635486e-01,
                5.0017547183707700e-01,
                6.2736375880677397e-01,
                -1.0123956423313052e00,
                3.5376430964398464e-01,
                -7.3462652628485570e-01,
                7.0459815464460285e-02,
                -6.2192794632214943e-01,
                -1.0347892293327476e-01,
                -8.5470888752302998e-01,
                -3.9619297675129961e-01,
                4.6731998637997024e-01,
                7.1647350012531086e-01,
                -1.4953479397912739e-01,
                1.6979693989973593e-01,
                8.4412241788946296e-02,
                7.8800963868610618e-01,
                -3.8989614881987722e-01,
                -4.3549531049711765e-01,
                6.6648376952263222e-01,
                -2.8197993042298186e-01,
                -2.8740899160069366e-01,
                -4.4099041630567498e-01,
                -2.0281265820548122e-03,
                6.0644616260210993e-01,
                -6.3424547779805474e-01,
                -2.6434740878673724e-01,
                -1.1364864647145254e-01,
                4.1509700138325806e-01,
                -7.1005962734102046e-01,
                2.5747059698208385e-01,
                -1.1326410442200113e-01,
                -3.0370536669364845e-02,
                -1.0206173098204343e00,
                1.5133184697527309e00,
                -4.4150875933253730e-03,
                1.4697088872151518e00,
                4.7825720062687233e-01,
                2.0159271287538461e-01,
                5.4599112836491070e-01,
                -2.1982794924960969e-01,
                6.1545492743888153e-01,
                1.3161052944957166e00,
                1.8957825833345171e-01,
                8.2680097833443100e-02,
            ],
            [
                2.6878557642360384e-01,
                -1.0333987707431274e00,
                1.9828500386068776e-01,
                5.7767920698407238e-01,
                -1.4442343314660269e00,
                3.3824735721526944e-01,
                9.5214925129486727e-01,
                9.0377919224393566e-01,
                -9.8356566027296399e-01,
                -1.7137696207431543e00,
                -3.6408416892165729e-01,
                -1.1887316373609087e00,
                6.3277567035866400e-01,
                -5.7083525936866775e-01,
                7.4785828357555512e-02,
                -4.7693709830893816e-01,
                9.2619876400408052e-01,
                6.4931561000074367e-01,
                9.9797017318916503e-01,
                -1.6364617942970070e00,
                -8.6333420139123623e-02,
                1.0595369566108008e00,
                8.2508617450601096e-01,
                -1.0654976999211252e-01,
                1.9759934438067397e00,
                -1.0761297671750998e00,
                -9.5787297317251863e-01,
                6.2269212494724069e-01,
                -2.4703527227386080e-01,
                1.7514257115020286e00,
                -5.5379475202105621e-02,
                -5.9734615404382696e-01,
                -4.0717677799619389e-01,
                -1.0094408911457646e00,
                -6.7684353748620074e-01,
                6.9402345384432462e-01,
                2.2993366216298244e-01,
                1.2300755504828298e00,
                4.1792957971762046e-01,
                1.6761856499695591e00,
                -1.1025976769508472e00,
                1.6031369317149458e00,
                1.3094967614200198e-01,
                -3.5881068549737105e-01,
                6.6857492586348710e-02,
                9.7031488294039725e-01,
                7.0846724274023531e-01,
                -1.6698996539371125e-01,
                -2.9378484782162284e-01,
                -6.0075210945827129e-01,
            ],
            [
                4.5492563695393712e-01,
                1.0840039154091112e00,
                -5.5007677470903155e-02,
                -1.1168398444535370e00,
                7.8901858853493201e-01,
                -1.3110523608947036e00,
                -4.3152907823884901e-01,
                7.3299858958149944e-01,
                -2.1420330735345035e00,
                -1.8074913441961216e-01,
                -5.4263985088356359e-01,
                9.9075590948642045e-02,
                -6.7221214776603877e-01,
                1.6172618497120472e-01,
                5.5715367807961991e-01,
                -8.7607957601554687e-01,
                1.0666771678726779e00,
                3.6436550953439506e-01,
                1.0402804422922787e00,
                1.4705323243043267e00,
                7.1300371137189877e-01,
                1.0466817466079053e00,
                -2.0829512587692778e-02,
                -1.1511257451150829e00,
                -5.2040768386253822e-02,
                -6.3782361483104277e-01,
                -1.5824870749112288e00,
                9.0075694127024569e-02,
                -7.2815726713375439e-01,
                -1.1650886050002982e00,
                -1.1744751180976749e00,
                2.8774097747553906e-01,
                3.4270885849070498e-01,
                1.3665008265959526e00,
                3.0996745242286033e-01,
                1.0761299081945668e00,
                1.2129324720260075e00,
                1.1154058080845746e00,
                -1.7332464973727619e00,
                1.2217181971455648e00,
                -9.9820467119971446e-02,
                4.3925758750689892e-01,
                6.7511011756709194e-01,
                5.8580324125089966e-01,
                2.2080660118909190e-01,
                -3.9186997189636369e-01,
                -1.6065272083028086e00,
                1.1669943531719978e-01,
                -2.3911010324305010e-01,
                1.5118189739796555e00,
            ],
        ]
    )
    assert np.allclose(result, expected, atol=1e-12)


def test_external_parameter_orthogonalization_snapshot_n_components_2():
    # Snapshot of exact output for n_components=2.
    # Arrange
    rng = np.random.default_rng(0)
    X = rng.normal(size=(3, 50))
    X_external = X + 0.5 * rng.normal(size=(3, 50))
    epo = ExternalParameterOrthogonalization(n_components=2)

    # Act
    result = epo.fit_transform(X, X_external=X_external)

    # Assert
    expected = np.array(
        [
            [
                0.3031736029105294,
                -0.24288712649262795,
                0.3994154144877667,
                0.05989557535714687,
                -0.1479269760593483,
                0.3842155226599153,
                0.3800808341200741,
                0.6636724670570517,
                -1.1012536831083657,
                0.4895073082854182,
                -0.7743334096228358,
                0.06430149509075067,
                -0.714693780877942,
                -0.08550724566521915,
                -0.8159756847678602,
                -0.41536717049306526,
                0.4582426346453918,
                0.7116514232319664,
                -0.16573823145647382,
                0.24508770993593537,
                0.23111802037593476,
                0.8872609982599035,
                -0.4497345662737614,
                -0.5722829568413343,
                0.5164439725625259,
                -0.29925945593195336,
                -0.3599222660551066,
                -0.49498140646463057,
                -0.05999978203645434,
                0.494555620830972,
                -0.6695500581475042,
                -0.16968307249734932,
                -0.11111032053376316,
                0.5018210092819906,
                -0.701792351091054,
                0.32540853295454325,
                -0.02443985658213732,
                -0.03907385820769715,
                -1.0840641182497794,
                1.545242925243834,
                -0.0177254669276718,
                1.356803575952498,
                0.5057652489512359,
                0.3041572384417096,
                0.4667731921733346,
                -0.2976681288598919,
                0.46421284562688797,
                1.2594951730105617,
                0.15976006811629478,
                0.288162630834234,
            ],
            [
                0.49243707465353986,
                -0.1711058491370947,
                0.584825494019665,
                -0.32732206530904795,
                -0.6395398157202687,
                -0.01505856138564732,
                0.2984993789684788,
                1.1013998601046957,
                -1.4672013005749558,
                -0.974949008783291,
                -0.580200389576819,
                -1.2222500808363452,
                0.12787072983557513,
                -0.47301919717747104,
                0.28560251024280153,
                -0.5812982049288727,
                0.8767926463358561,
                0.6230700589453334,
                0.9097782685408141,
                -1.2266699571503152,
                0.7121552989733695,
                1.5997412467183185,
                0.49939824478251016,
                -0.8510561798404666,
                1.1593583634132576,
                -1.1701785928138855,
                -1.35254748562483,
                0.32883051301300353,
                -0.5625628087789546,
                1.1424290091215354,
                -0.24753488594020934,
                -0.0821080676004371,
                -0.39336120340740055,
                -0.5374203432823406,
                -0.6318464904239361,
                1.0637953605564356,
                0.7133853746102117,
                1.1827052005613612,
                0.07260193630455791,
                1.8499437515179573,
                -1.1750432741723777,
                0.9886170522702677,
                0.2806702016956799,
                0.19942647076621212,
                -0.36430908455594346,
                0.546647145301512,
                -0.1147116214130833,
                -0.4751069586632103,
                -0.4560789901660777,
                0.5176461305845006,
            ],
            [
                0.19018280406314567,
                0.06328251334205433,
                -0.5125669838095236,
                -0.04556331604715826,
                -0.16352189250934213,
                -0.8928337908973479,
                0.3422154318045419,
                0.49906921347046157,
                -1.569539392455452,
                -1.0553127450209083,
                -0.28681674689042164,
                0.13875235479778786,
                -0.07454137268715569,
                0.04593844551195214,
                0.30760379343920363,
                -0.7525442756538469,
                1.1251606372754808,
                0.3954331374831497,
                1.1446757844179758,
                0.985449717121435,
                -0.23219078632758242,
                0.4072260969265903,
                0.36469683458969215,
                -0.2698316889225124,
                0.9146341089673347,
                -0.5264952636832853,
                -1.1152992880045045,
                0.4379282962202175,
                -0.3546580751742611,
                -0.4442013608486669,
                -0.9470151270101221,
                -0.3221614452572387,
                0.32635495796422215,
                0.8077562708337953,
                0.25670312911062926,
                0.6384200655099966,
                0.6406565117389142,
                1.1714794795443757,
                -1.3244720455303547,
                1.0160356401060635,
                -0.01406449056409431,
                1.1666827782142308,
                0.4978815436890509,
                -0.07499844057900848,
                0.7311911145229602,
                0.1096379453528038,
                -0.6321062623374964,
                0.48142655007185337,
                -0.04699777068143772,
                0.18793820093609262,
            ],
        ]
    )
    assert np.allclose(result, expected, atol=1e-12)
