import numpy as np
import pytest
from sklearn.utils.estimator_checks import check_estimator

from chemotools.physics import IntensityConversion


# Test compliance with scikit-learn
def test_compliance_intensity_conversion():
    """Test compliance with scikit-learn estimator interface."""
    # Arrange
    transformer = IntensityConversion()
    # Act & Assert
    check_estimator(transformer)


# Test functionality
class TestAbsorbanceTransmittance:
    def test_absorbance_to_transmittance_zero(self):
        """Test that zero absorbance converts to transmittance of one."""
        # Arrange
        X = np.array([[0.0, 0.0]])

        # Act
        result = IntensityConversion("absorbance", "transmittance").fit_transform(X)

        # Assert
        assert np.allclose(result, [[1.0, 1.0]], atol=1e-10)

    def test_absorbance_to_transmittance_known_values(self):
        """Test absorbance to transmittance conversion with known values."""
        # Arrange
        X = np.array([[1.0, 2.0]])

        # Act
        result = IntensityConversion("absorbance", "transmittance").fit_transform(X)

        # Assert
        assert np.allclose(result, [[0.1, 0.01]], atol=1e-10)

    def test_transmittance_to_absorbance_one(self):
        """Test that transmittance of one converts to absorbance of zero."""
        # Arrange
        X = np.array([[1.0]])

        # Act
        result = IntensityConversion("transmittance", "absorbance").fit_transform(X)

        # Assert
        assert np.allclose(result, [[0.0]], atol=1e-10)

    def test_transmittance_to_absorbance_known_values(self):
        """Test transmittance to absorbance conversion with known values."""
        # Arrange
        X = np.array([[0.1, 0.01]])

        # Act
        result = IntensityConversion("transmittance", "absorbance").fit_transform(X)

        # Assert
        assert np.allclose(result, [[1.0, 2.0]], atol=1e-10)

    def test_absorbance_transmittance_round_trip(self):
        """Test that absorbance -> transmittance -> absorbance is lossless."""
        # Arrange
        X_A = np.array([[0.5, 1.0, 1.5]])

        # Act
        X_T = IntensityConversion("absorbance", "transmittance").fit_transform(X_A)
        X_A_back = IntensityConversion("transmittance", "absorbance").fit_transform(X_T)

        # Assert
        assert np.allclose(X_A, X_A_back, atol=1e-10)


class TestReflectanceKubelkaMunk:
    def test_reflectance_to_kubelka_munk_one(self):
        """Test that reflectance of one converts to Kubelka-Munk of zero."""
        # Arrange
        X = np.array([[1.0]])

        # Act
        result = IntensityConversion("reflectance", "kubelka_munk").fit_transform(X)

        # Assert
        assert np.allclose(result, [[0.0]], atol=1e-10)

    def test_reflectance_to_kubelka_munk_half(self):
        """Test reflectance of 0.5 converts to expected Kubelka-Munk value."""
        # Arrange
        X = np.array([[0.5]])

        # Act
        result = IntensityConversion("reflectance", "kubelka_munk").fit_transform(X)

        # Assert
        assert np.allclose(result, [[0.25]], atol=1e-10)

    def test_kubelka_munk_to_reflectance_zero(self):
        """Test that Kubelka-Munk of zero converts to reflectance of one."""
        # Arrange
        X = np.array([[0.0]])

        # Act
        result = IntensityConversion("kubelka_munk", "reflectance").fit_transform(X)

        # Assert
        assert np.allclose(result, [[1.0]], atol=1e-10)

    def test_reflectance_kubelka_munk_round_trip(self):
        """Test that reflectance -> Kubelka-Munk -> reflectance is lossless."""
        # Arrange
        X_R = np.array([[0.1, 0.5, 0.9]])

        # Act
        X_KM = IntensityConversion("reflectance", "kubelka_munk").fit_transform(X_R)
        X_R_back = IntensityConversion("kubelka_munk", "reflectance").fit_transform(
            X_KM
        )

        # Assert
        assert np.allclose(X_R, X_R_back, atol=1e-10)


class TestReflectancePseudoAbsorbance:
    def test_reflectance_to_pseudoabsorbance_one(self):
        """Test that reflectance of one converts to pseudoabsorbance of zero."""
        # Arrange
        X = np.array([[1.0]])

        # Act
        result = IntensityConversion("reflectance", "pseudoabsorbance").fit_transform(X)

        # Assert
        assert np.allclose(result, [[0.0]], atol=1e-10)

    def test_reflectance_to_pseudoabsorbance_known_values(self):
        """Test reflectance to pseudoabsorbance conversion with known values."""
        # Arrange
        X = np.array([[0.1]])

        # Act
        result = IntensityConversion("reflectance", "pseudoabsorbance").fit_transform(X)

        # Assert
        assert np.allclose(result, [[1.0]], atol=1e-10)

    def test_pseudoabsorbance_to_reflectance_zero(self):
        """Test that pseudoabsorbance of zero converts to reflectance of one."""
        # Arrange
        X = np.array([[0.0]])

        # Act
        result = IntensityConversion("pseudoabsorbance", "reflectance").fit_transform(X)

        # Assert
        assert np.allclose(result, [[1.0]], atol=1e-10)

    def test_pseudoabsorbance_to_reflectance_one(self):
        """Test pseudoabsorbance to reflectance conversion with known values."""
        # Arrange
        X = np.array([[1.0]])

        # Act
        result = IntensityConversion("pseudoabsorbance", "reflectance").fit_transform(X)

        # Assert
        assert np.allclose(result, [[0.1]], atol=1e-10)

    def test_reflectance_pseudoabsorbance_round_trip(self):
        """Test that reflectance -> pseudoabsorbance -> reflectance is lossless."""
        # Arrange
        X_R = np.array([[0.1, 0.5, 0.9]])

        # Act
        X_PA = IntensityConversion("reflectance", "pseudoabsorbance").fit_transform(X_R)
        X_R_back = IntensityConversion("pseudoabsorbance", "reflectance").fit_transform(
            X_PA
        )

        # Assert
        assert np.allclose(X_R, X_R_back, atol=1e-10)


class TestMultiSamples:
    def test_multiple_samples_absorbance_to_transmittance(self):
        """Test absorbance to transmittance conversion with multiple samples."""
        # Arrange
        X = np.array([[0.0], [1.0], [2.0]])
        expected = np.array([[1.0], [0.1], [0.01]])

        # Act
        result = IntensityConversion("absorbance", "transmittance").fit_transform(X)

        # Assert
        assert np.allclose(result, expected, atol=1e-10)


class TestValidationErrors:
    def test_unsupported_conversion_raises(self):
        """Test that an unsupported conversion pair raises a ValueError."""
        # Arrange
        t = IntensityConversion(input_unit="absorbance", output_unit="reflectance")
        X = np.array([[1.0, 2.0]])

        # Act & Assert
        with pytest.raises(ValueError, match="not supported"):
            t.fit(X)

    def test_invalid_input_unit_raises(self):
        """Test that an invalid input unit raises a ValueError."""
        # Arrange
        t = IntensityConversion(input_unit="banana", output_unit="transmittance")
        X = np.array([[1.0]])

        # Act & Assert
        with pytest.raises(ValueError):
            t.fit(X)

    def test_invalid_output_unit_raises(self):
        """Test that an invalid output unit raises a ValueError."""
        # Arrange
        t = IntensityConversion(input_unit="absorbance", output_unit="banana")
        X = np.array([[1.0]])

        # Act & Assert
        with pytest.raises(ValueError):
            t.fit(X)


def test_intensity_conversion_snapshot_absorbance_to_transmittance():
    # Snapshot of exact output for absorbance -> transmittance.
    # Arrange
    rng = np.random.default_rng(0)
    X = np.abs(rng.normal(size=(3, 50))) + 0.1
    ic = IntensityConversion("absorbance", "transmittance")

    # Act
    result = ic.fit_transform(X)

    # Assert
    expected = np.array(
        [
            [
                0.594661440986358,
                0.585996654640532,
                0.18179308078966222,
                0.6238783039777809,
                0.23138256270219598,
                0.3454657085258492,
                0.03944572610848628,
                0.08972615073850987,
                0.15713204554687438,
                0.04311005023776994,
                0.18911480857267116,
                0.7222274996181394,
                0.00375810772933294,
                0.4799636375078028,
                0.04509091546735824,
                0.14714064155420264,
                0.22685116667381588,
                0.38344214338346416,
                0.30787148296871214,
                0.0720255578348083,
                0.5908338066599212,
                0.03416146834292097,
                0.17171385033903155,
                0.35358182241661174,
                0.09920414497565265,
                0.6397167205975391,
                0.1433840194341388,
                0.09512060934954089,
                0.2768688994099187,
                0.4784150976783276,
                0.07769298659217494,
                0.49070945395103416,
                0.5505223943876207,
                0.22864116025765555,
                0.4845525433824371,
                0.35045098982785555,
                0.17626715323272932,
                0.5893677480830217,
                0.13062446657136104,
                0.02550168369614257,
                0.04374560910082517,
                0.02432630935189093,
                0.03581991708474758,
                0.13142821190040452,
                0.4320603067796905,
                0.3855468733184022,
                0.02766809871282707,
                0.00870445698673948,
                0.01254195187881999,
                0.03844999036110031,
            ],
            [
                0.34883462763067374,
                0.04916786695229584,
                0.786223221680152,
                0.17519635443243037,
                0.04089201733699523,
                0.31979961738981083,
                0.29521356216540173,
                0.15994006789376364,
                0.05198547699841998,
                0.1731001435220589,
                0.29078014773090366,
                0.05372768054467471,
                0.01447545164312943,
                0.2535649793334033,
                0.3724177487972784,
                0.4379529484679439,
                0.02072655490525987,
                0.03798735128736833,
                0.18477677257105135,
                0.00497153062852484,
                0.7046460564447983,
                0.1645560330278873,
                0.07871154265676643,
                0.19146656913684573,
                0.01196709218964502,
                0.03798123043589836,
                0.17316972935093264,
                0.09224652435544509,
                0.709488542054831,
                0.00789964211271902,
                0.5146130645283661,
                0.18484423484975213,
                0.33299406629078365,
                0.06439525723470049,
                0.04191020967666508,
                0.1860323658623126,
                0.20836951822434663,
                0.0403126344616672,
                0.1397636418876513,
                0.0162514661483691,
                0.4098380647982133,
                0.02116370606554967,
                0.29323388399988426,
                0.14605509369291028,
                0.4469043978502942,
                0.07388340720315414,
                0.5482648748880529,
                0.20628667503077297,
                0.03620597822811882,
                0.03151227697957923,
            ],
            [
                0.24964171078660036,
                0.08133677838306774,
                0.5441334285461557,
                0.06693220642241811,
                0.10640397375169931,
                0.04164914163609255,
                0.15379134847627268,
                0.19010001295412896,
                0.00446538414917788,
                0.3263100144119969,
                0.20814173174863637,
                0.617618508880287,
                0.6672651971955875,
                0.4987530964346121,
                0.16063051926569677,
                0.13855756711176406,
                0.0301313075062381,
                0.1492472064934957,
                0.11383277881248363,
                0.05434205014627268,
                0.12954235210638085,
                0.11374212020043674,
                0.6674310229034364,
                0.02973213862449528,
                0.5820427714270391,
                0.13504713023589185,
                0.03000946351203052,
                0.43807372848446935,
                0.21451148325693034,
                0.07416441270770124,
                0.07194471887674782,
                0.4281371564495221,
                0.34779877725135705,
                0.03780441571571783,
                0.7692815761782277,
                0.07213735938926738,
                0.03145829445794697,
                0.05621268939786048,
                0.00342528011814229,
                0.04691549261470457,
                0.36339587377807414,
                0.2993840417403918,
                0.3378878553395163,
                0.32903556269005163,
                0.3807025447027963,
                0.3476055416131306,
                0.00996241666617349,
                0.6181377572541542,
                0.12481539385360142,
                0.06604448948070078,
            ],
        ]
    )
    assert np.allclose(result, expected, atol=1e-12)


def test_intensity_conversion_snapshot_reflectance_to_kubelka_munk():
    # Snapshot of exact output for reflectance -> kubelka_munk.
    # Arrange
    rng = np.random.default_rng(0)
    # consume same draws as abs snapshot to get the reflectance rng state
    _ = np.abs(rng.normal(size=(3, 50))) + 0.1
    X = rng.uniform(0.05, 0.95, size=(3, 50))
    ic = IntensityConversion("reflectance", "kubelka_munk")

    # Act
    result = ic.fit_transform(X)

    # Assert
    expected = np.array(
        [
            [
                1.0317103396472729e-01,
                8.3477643595020556e-01,
                7.3915626017843808e-02,
                5.6236594794422433e-03,
                2.1283333051617861e00,
                1.7795915394675073e-02,
                3.8818490266186036e00,
                4.6960450901682077e-01,
                3.6309573811491436e-01,
                2.6546420029969625e-01,
                2.7277461789887119e-03,
                4.2401163247324562e-02,
                6.8850793550636769e-01,
                8.5376715575258844e-01,
                1.8139276919181109e-02,
                1.4583912988490672e-02,
                2.3591059416056137e-01,
                5.6933852612725455e-01,
                1.5751443777005714e-03,
                6.6261712589502653e-01,
                1.4388627059267012e00,
                1.4806108710915609e-02,
                3.0672309846973708e-02,
                9.3480877417771793e-02,
                4.1879467965030871e-03,
                7.7312103895869778e-03,
                5.2869764087317604e-02,
                1.8647238545664656e-02,
                9.7153596008756760e-01,
                1.9114755454785570e00,
                9.2159350711748078e-02,
                6.7915973043860350e-02,
                1.5958359092155872e00,
                4.3452317762521786e-01,
                9.8367173150545311e-03,
                1.7810827997629239e-01,
                1.6167178169025662e-01,
                1.3373812255835875e00,
                2.1696578603046462e-01,
                2.2007046205292052e-01,
                2.9099304284207395e00,
                2.3503500703090955e-03,
                1.6825056775729702e-01,
                7.9936000032781340e00,
                4.3487075024974529e-02,
                2.6002307305480611e-03,
                1.5120000982602783e-01,
                6.4940175595595018e-01,
                1.3950202007611463e00,
                9.0676603026234481e-02,
            ],
            [
                1.3291432125951776e00,
                1.6227704658033137e-01,
                1.4058045518406670e-01,
                3.8342110675737410e-03,
                3.4038813380462569e00,
                2.5003669045389554e-01,
                5.4589625840489042e-02,
                1.4913405540324063e00,
                4.5194663153378289e-01,
                3.7434727680339064e00,
                6.2587857271730371e-02,
                2.9407320291451220e00,
                4.3558596217824957e-01,
                1.6049449577086861e-02,
                2.9001095725220899e-01,
                9.4956856833874250e-03,
                4.5954895792217676e-02,
                9.1146275305205478e-03,
                2.1188415975613526e00,
                3.3607843627844587e00,
                3.4699576762410476e00,
                1.6968460930164058e-02,
                1.1592166600880373e-01,
                2.5466653046269117e-01,
                1.6342319759903321e00,
                8.9957011490026759e-02,
                6.5524859681798076e-01,
                6.9752256318083683e-02,
                3.0900392731722820e-01,
                2.4009407784895756e-01,
                3.7639644863102437e-02,
                2.8128696042117585e00,
                1.6127806348563697e-01,
                1.3114472902026904e00,
                3.1894725430880982e-02,
                2.6546920130506180e-01,
                1.9263859014006031e-03,
                1.4367088312289551e00,
                3.7830752106212213e-03,
                3.4068047437973085e-02,
                2.7647583572995948e-01,
                3.0328841158582025e-02,
                1.4007269930389293e-01,
                1.0153211732421616e-01,
                9.3438877009765827e-03,
                3.6523529534326875e00,
                2.4583273120358424e-02,
                4.6703438281328907e-01,
                6.2925949354623256e-01,
                1.6231276412777135e-03,
            ],
            [
                4.0483348380094659e-02,
                2.7022359749378355e-01,
                3.7698579008737088e-01,
                1.5284864541752137e-02,
                2.9662503429616915e00,
                7.0979848081228342e-02,
                3.7807352397239834e-02,
                3.4599526419656662e-02,
                6.4036621317624998e-01,
                3.5399138592540937e-02,
                1.1042803433688326e00,
                5.1755299456739268e-01,
                3.8737860117413336e-01,
                1.9926703184739691e-01,
                2.3792268940022132e00,
                4.0931899487041412e-01,
                8.9713024574165452e00,
                5.4471086554632120e-02,
                2.0572824901081452e-02,
                1.9440334966635158e00,
                7.3331834216403838e-02,
                2.8215750659310989e-02,
                2.3579190196696392e-03,
                2.2438508545868832e-02,
                3.7407080887872646e-01,
                2.5019219195069851e-03,
                2.9083219664281629e-03,
                2.4507959542730215e-01,
                5.0768022058950923e-02,
                9.3231329341555841e-03,
                2.8412758948921002e-01,
                1.8000880135981799e-02,
                7.4476632455454897e-02,
                7.4693014535735347e-01,
                4.5310229898526989e-02,
                1.6893649528947055e-01,
                2.7857906881057115e00,
                4.4415280733434220e-01,
                3.3549375002600694e00,
                2.8409745325088209e-01,
                3.6545923716268641e-01,
                3.7479653964948961e-01,
                1.5438075531428527e-01,
                2.1969981761280586e00,
                6.7464157649503526e-03,
                8.3973385685052304e-02,
                2.7490800088399811e-02,
                1.1908689021574886e-02,
                1.5707732246535933e-01,
                4.8438045732094794e00,
            ],
        ]
    )
    assert np.allclose(result, expected, atol=1e-12)


class TestEdgeCases:
    def test_zero_transmittance_warns(self):
        """Test that zero transmittance values trigger a UserWarning."""
        # Arrange
        X = np.array([[0.0, 0.5]])
        t = IntensityConversion("transmittance", "absorbance").fit(X)

        # Act & Assert
        with pytest.warns(UserWarning):
            t.transform(X)

    def test_zero_reflectance_kubelka_munk_warns(self):
        """Test that zero reflectance values warn during Kubelka-Munk conversion."""
        # Arrange
        X = np.array([[0.0, 0.5]])
        t = IntensityConversion("reflectance", "kubelka_munk").fit(X)

        # Act & Assert
        with pytest.warns(UserWarning):
            t.transform(X)

    def test_zero_kubelka_munk_reflectance_warns(self):
        """
        Test that zero reflectance values warn during inverse Kubelka-Munk conversion.
        """
        # Arrange
        X = np.array([[-0.1, 0.0, 0.5]])
        t = IntensityConversion("kubelka_munk", "reflectance").fit(X)

        # Act & Assert
        with pytest.warns(UserWarning):
            t.transform(X)

    def test_zero_reflectance_pseudoabsorbance_warns(self):
        """Test that zero reflectance values warn during pseudoabsorbance conversion."""
        # Arrange
        X = np.array([[0.0, 0.5]])
        t = IntensityConversion("reflectance", "pseudoabsorbance").fit(X)

        # Act & Assert
        with pytest.warns(UserWarning):
            t.transform(X)
