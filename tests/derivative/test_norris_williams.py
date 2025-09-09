import numpy as np
import pytest
from sklearn.utils.estimator_checks import check_estimator

from chemotools.derivative import NorrisWilliams


# Test compliance with scikit-learn
def test_compliance_norris_williams_der_1():
    # Arrange
    transformer = NorrisWilliams()
    # Act & Assert
    check_estimator(transformer)


def test_compliance_norris_williams_der_2():
    # Arrange
    transformer = NorrisWilliams(derivative_order=2)
    # Act & Assert
    check_estimator(transformer)


# Test functionality
def test_norris_williams_filter_1():
    # Arrange
    norris_williams_filter = NorrisWilliams()
    array = np.ones((1, 10)).reshape(1, -1)

    # Act
    spectrum_corrected = norris_williams_filter.fit_transform(array)

    # Assert
    assert np.allclose(spectrum_corrected[0], np.zeros((1, 10)), atol=1e-2)


def test_norris_williams_filter_2():
    # Arrange
    norris_williams_filter = NorrisWilliams(derivative_order=2)
    array = np.ones((1, 10)).reshape(1, -1)

    # Act
    spectrum_corrected = norris_williams_filter.fit_transform(array)

    # Assert
    assert np.allclose(spectrum_corrected[0], np.zeros((1, 10)), atol=1e-2)


def test_norris_williams_wrong_filter():
    # Arrange
    norris_williams_filter = NorrisWilliams(derivative_order=5)
    array = np.ones((1, 10)).reshape(1, -1)

    # Act & Assert
    with pytest.raises(ValueError):
        norris_williams_filter.fit_transform(array)
