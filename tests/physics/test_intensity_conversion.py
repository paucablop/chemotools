import numpy as np
import pytest
from sklearn.utils.estimator_checks import check_estimator

from chemotools.physics import IntensityConversion


# --- sklearn compliance ---

def test_compliance_intensity_conversion():
    check_estimator(IntensityConversion())


# --- absorbance <-> transmittance ---

def test_absorbance_to_transmittance_zero():
    X = np.array([[0.0, 0.0]])
    result = IntensityConversion("absorbance", "transmittance").fit_transform(X)
    assert np.allclose(result, [[1.0, 1.0]], atol=1e-10)


def test_absorbance_to_transmittance_known_values():
    X = np.array([[1.0, 2.0]])
    result = IntensityConversion("absorbance", "transmittance").fit_transform(X)
    assert np.allclose(result, [[0.1, 0.01]], atol=1e-10)


def test_transmittance_to_absorbance_one():
    X = np.array([[1.0]])
    result = IntensityConversion("transmittance", "absorbance").fit_transform(X)
    assert np.allclose(result, [[0.0]], atol=1e-10)


def test_transmittance_to_absorbance_known_values():
    X = np.array([[0.1, 0.01]])
    result = IntensityConversion("transmittance", "absorbance").fit_transform(X)
    assert np.allclose(result, [[1.0, 2.0]], atol=1e-10)


def test_absorbance_transmittance_round_trip():
    X_A = np.array([[0.5, 1.0, 1.5]])
    X_T = IntensityConversion("absorbance", "transmittance").fit_transform(X_A)
    X_A_back = IntensityConversion("transmittance", "absorbance").fit_transform(X_T)
    assert np.allclose(X_A, X_A_back, atol=1e-10)


# --- reflectance <-> kubelka_munk ---

def test_reflectance_to_kubelka_munk_one():
    X = np.array([[1.0]])
    result = IntensityConversion("reflectance", "kubelka_munk").fit_transform(X)
    assert np.allclose(result, [[0.0]], atol=1e-10)


def test_reflectance_to_kubelka_munk_half():
    X = np.array([[0.5]])
    result = IntensityConversion("reflectance", "kubelka_munk").fit_transform(X)
    assert np.allclose(result, [[0.25]], atol=1e-10)


def test_kubelka_munk_to_reflectance_zero():
    X = np.array([[0.0]])
    result = IntensityConversion("kubelka_munk", "reflectance").fit_transform(X)
    assert np.allclose(result, [[1.0]], atol=1e-10)


def test_reflectance_kubelka_munk_round_trip():
    X_R = np.array([[0.1, 0.5, 0.9]])
    X_KM = IntensityConversion("reflectance", "kubelka_munk").fit_transform(X_R)
    X_R_back = IntensityConversion("kubelka_munk", "reflectance").fit_transform(X_KM)
    assert np.allclose(X_R, X_R_back, atol=1e-10)


# --- reflectance <-> pseudoabsorbance ---

def test_reflectance_to_pseudoabsorbance_one():
    X = np.array([[1.0]])
    result = IntensityConversion("reflectance", "pseudoabsorbance").fit_transform(X)
    assert np.allclose(result, [[0.0]], atol=1e-10)


def test_reflectance_to_pseudoabsorbance_known_values():
    X = np.array([[0.1]])
    result = IntensityConversion("reflectance", "pseudoabsorbance").fit_transform(X)
    assert np.allclose(result, [[1.0]], atol=1e-10)


def test_pseudoabsorbance_to_reflectance_zero():
    X = np.array([[0.0]])
    result = IntensityConversion("pseudoabsorbance", "reflectance").fit_transform(X)
    assert np.allclose(result, [[1.0]], atol=1e-10)


def test_pseudoabsorbance_to_reflectance_one():
    X = np.array([[1.0]])
    result = IntensityConversion("pseudoabsorbance", "reflectance").fit_transform(X)
    assert np.allclose(result, [[0.1]], atol=1e-10)


def test_reflectance_pseudoabsorbance_round_trip():
    X_R = np.array([[0.1, 0.5, 0.9]])
    X_PA = IntensityConversion("reflectance", "pseudoabsorbance").fit_transform(X_R)
    X_R_back = IntensityConversion("pseudoabsorbance", "reflectance").fit_transform(X_PA)
    assert np.allclose(X_R, X_R_back, atol=1e-10)


# --- multiple samples ---

def test_multiple_samples_absorbance_to_transmittance():
    X = np.array([[0.0], [1.0], [2.0]])
    result = IntensityConversion("absorbance", "transmittance").fit_transform(X)
    expected = np.array([[1.0], [0.1], [0.01]])
    assert np.allclose(result, expected, atol=1e-10)


# --- validation errors ---

def test_unsupported_conversion_raises():
    t = IntensityConversion(input_unit="absorbance", output_unit="reflectance")
    with pytest.raises(ValueError, match="not supported"):
        t.fit(np.array([[1.0, 2.0]]))


def test_invalid_input_unit_raises():
    t = IntensityConversion(input_unit="banana", output_unit="transmittance")
    with pytest.raises(ValueError):
        t.fit(np.array([[1.0]]))


def test_invalid_output_unit_raises():
    t = IntensityConversion(input_unit="absorbance", output_unit="banana")
    with pytest.raises(ValueError):
        t.fit(np.array([[1.0]]))


# --- numerical edge case warnings ---

def test_zero_transmittance_warns():
    X = np.array([[0.0, 0.5]])
    t = IntensityConversion("transmittance", "absorbance").fit(X)
    with pytest.warns(UserWarning):
        t.transform(X)


def test_zero_reflectance_kubelka_munk_warns():
    X = np.array([[0.0, 0.5]])
    t = IntensityConversion("reflectance", "kubelka_munk").fit(X)
    with pytest.warns(UserWarning):
        t.transform(X)


def test_zero_reflectance_pseudoabsorbance_warns():
    X = np.array([[0.0, 0.5]])
    t = IntensityConversion("reflectance", "pseudoabsorbance").fit(X)
    with pytest.warns(UserWarning):
        t.transform(X)
