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
