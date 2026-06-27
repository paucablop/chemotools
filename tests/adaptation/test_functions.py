import numpy as np

from chemotools.adaptation.functions import (
    add_offset,
    divide_by_reference,
    scale_by_factor,
    subtract_reference,
)


class TestSubtractReference:
    def test_subtract_reference(self):
        # Arrange
        X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        reference = np.array([[0.5, 0.5, 0.5]])

        # Act
        result = subtract_reference(X, reference)

        # Assert
        np.testing.assert_allclose(result, [[0.5, 1.5, 2.5], [3.5, 4.5, 5.5]])


class TestDivideByReference:
    def test_divide_by_reference(self):
        # Arrange
        X = np.array([[2.0, 4.0, 6.0], [8.0, 10.0, 12.0]])
        reference = np.array([[2.0, 2.0, 2.0]])

        # Act
        result = divide_by_reference(X, reference)

        # Assert
        np.testing.assert_allclose(result, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])


class TestScaleByFactor:
    def test_scale_by_factor_scalar(self):
        # Arrange
        X = np.array([[1.0, 2.0], [3.0, 4.0]])

        # Act
        result = scale_by_factor(X, 2.0)

        # Assert
        np.testing.assert_allclose(result, [[2.0, 4.0], [6.0, 8.0]])

    def test_scale_by_factor_per_sample(self):
        # Arrange
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        factor = np.array([[2.0], [0.5]])

        # Act
        result = scale_by_factor(X, factor)

        # Assert
        np.testing.assert_allclose(result, [[2.0, 4.0], [1.5, 2.0]])


class TestAddOffset:
    def test_add_offset(self):
        # Arrange
        X = np.array([[1.0, 2.0, 3.0]])
        offset = np.array([[0.1, 0.2, 0.3]])

        # Act
        result = add_offset(X, offset)

        # Assert
        np.testing.assert_allclose(result, [[1.1, 2.2, 3.3]])
