"""Tests for :class:`chemotools.baseline.RubberbandCorrection`."""

import numpy as np
from numpy.testing import assert_allclose
from sklearn.utils.estimator_checks import check_estimator

from chemotools.baseline import RubberbandCorrection


# Test compliance with scikit-learn
def test_compliance_rubberband():
    # Arrange
    transformer = RubberbandCorrection()
    # Act & Assert
    check_estimator(transformer)


# Test functionality: a known spectrum gives a known baseline-corrected result
def test_rubberband_known_correction():
    # Arrange
    rubberband = RubberbandCorrection()
    spectrum = np.array([[0.0, 1.0, 10.0, 3.0, 4.0]])
    # The lower convex hull is the straight line through the first and last
    # points ([0, 1, 2, 3, 4]); subtracting it leaves only the single peak.
    expected = np.array([[0.0, 0.0, 8.0, 0.0, 0.0]])

    # Act
    corrected = rubberband.fit_transform(spectrum)

    # Assert
    assert_allclose(corrected, expected)


# Test functionality: a spectrum that is already linear collapses to zero
def test_rubberband_linear_spectrum_becomes_zero():
    # Arrange
    rubberband = RubberbandCorrection()
    spectrum = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])

    # Act
    corrected = rubberband.fit_transform(spectrum)

    # Assert
    assert_allclose(corrected, np.zeros_like(spectrum), atol=1e-12)


# Test functionality: peaks rest on a flat, non-negative, zero background
def test_rubberband_endpoints_and_nonnegativity():
    # Arrange
    rubberband = RubberbandCorrection()
    feature_axis = np.linspace(0, 1, 200)
    sloped_baseline = 5.0 + 3.0 * feature_axis
    peak = 10.0 * np.exp(-((feature_axis - 0.5) ** 2) / 0.001)
    spectrum = (sloped_baseline + peak).reshape(1, -1)

    # Act
    corrected = rubberband.fit_transform(spectrum)

    # Assert
    assert corrected.shape == spectrum.shape
    assert_allclose(corrected[0, [0, -1]], [0.0, 0.0], atol=1e-9)
    assert np.all(corrected >= -1e-9)
    assert np.isclose(corrected.min(), 0.0, atol=1e-9)


# Test functionality: each spectrum in a batch is corrected independently
def test_rubberband_batch_is_row_wise():
    # Arrange
    rubberband = RubberbandCorrection()
    spectrum_a = np.array([0.0, 1.0, 10.0, 3.0, 4.0])
    spectrum_b = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    batch = np.vstack([spectrum_a, spectrum_b])

    # Act
    corrected = rubberband.fit_transform(batch)

    # Assert
    assert_allclose(corrected[0], [0.0, 0.0, 8.0, 0.0, 0.0])
    assert_allclose(corrected[1], np.zeros(5), atol=1e-12)
