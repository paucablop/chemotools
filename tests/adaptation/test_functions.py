import numpy as np
import pytest

from chemotools.adaptation.functions import (
    add_offset,
    divide_by_reference,
    scale_by_factor,
    subtract_reference,
)
from chemotools.adaptation.validation import check_metadata_function


class TestValidationWithMetadataFunctionValidation:
    @pytest.mark.parametrize(
        "func, metadata, expected",
        [
            (
                subtract_reference,
                {"reference": np.array([[0.5, 0.5, 0.5]])},
                [[0.5, 1.5, 2.5], [3.5, 4.5, 5.5]],
            ),
            (
                divide_by_reference,
                {"reference": np.array([[2.0, 2.0, 2.0]])},
                [[0.5, 1.0, 1.5], [2.0, 2.5, 3.0]],
            ),
            (
                scale_by_factor,
                {"factor": np.array([[2.0], [0.5]])},
                [[2.0, 4.0, 6.0], [2.0, 2.5, 3.0]],
            ),
            (
                add_offset,
                {"offset": np.array([[0.5], [1.0]])},
                [[1.5, 2.5, 3.5], [5.0, 6.0, 7.0]],
            ),
        ],
    )
    def test_validation_with_metadata_function(self, func, metadata, expected):
        """
        Test that check_metadata_function validates the function and returns the
        expected output.
        """
        # Arrange
        X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        # Act
        result = check_metadata_function(func, X, metadata=metadata)

        # Assert
        np.testing.assert_allclose(result, expected)


class TestSubtractReference:
    def test_subtract_reference_shared(self):
        """Single shared reference broadcast across all samples."""
        # Arrange
        X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        reference = np.array([[0.5, 0.5, 0.5]])

        # Act
        result = subtract_reference(X, reference)

        # Assert
        np.testing.assert_allclose(result, [[0.5, 1.5, 2.5], [3.5, 4.5, 5.5]])

    def test_subtract_reference_scalar(self):
        """Scalar reference subtracted from every element."""
        # Arrange
        X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        # Act
        result = subtract_reference(X, 1.0)

        # Assert
        np.testing.assert_allclose(result, [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])

    def test_subtract_reference_per_sample(self):
        """Per-sample scalar reference of shape (n_samples, 1)."""
        # Arrange
        X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        reference = np.array([[0.5], [1.0]])

        # Act
        result = subtract_reference(X, reference)

        # Assert
        np.testing.assert_allclose(result, [[0.5, 1.5, 2.5], [3.0, 4.0, 5.0]])

    def test_subtract_reference_1d_raises(self):
        """1-D array input raises ValueError — user must provide explicit shape."""
        # Arrange
        X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        reference = np.array([0.5, 0.5, 0.5])  # shape (3,)

        # Act & Assert
        with pytest.raises(ValueError, match="Invalid metadata argument `reference`"):
            subtract_reference(X, reference)


class TestDivideByReference:
    def test_divide_by_reference_shared(self):
        """Single shared reference broadcast across all samples."""
        # Arrange
        X = np.array([[2.0, 4.0, 6.0], [8.0, 10.0, 12.0]])
        reference = np.array([[2.0, 2.0, 2.0]])

        # Act
        result = divide_by_reference(X, reference)

        # Assert
        np.testing.assert_allclose(result, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    def test_divide_by_reference_scalar(self):
        """Scalar reference divides every element."""
        # Arrange
        X = np.array([[2.0, 4.0, 6.0], [8.0, 10.0, 12.0]])

        # Act
        result = divide_by_reference(X, 2.0)

        # Assert
        np.testing.assert_allclose(result, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    def test_divide_by_reference_per_sample(self):
        """Per-sample scalar reference of shape (n_samples, 1)."""
        # Arrange
        X = np.array([[2.0, 4.0, 6.0], [8.0, 10.0, 12.0]])
        reference = np.array([[2.0], [4.0]])

        # Act
        result = divide_by_reference(X, reference)

        # Assert
        np.testing.assert_allclose(result, [[1.0, 2.0, 3.0], [2.0, 2.5, 3.0]])

    def test_divide_by_reference_1d_raises(self):
        """1-D array input raises ValueError — user must provide explicit shape."""
        # Arrange
        X = np.array([[2.0, 4.0, 6.0], [8.0, 10.0, 12.0]])
        reference = np.array([2.0, 2.0, 2.0])  # shape (3,)

        # Act & Assert
        with pytest.raises(ValueError, match="Invalid metadata argument `reference`"):
            divide_by_reference(X, reference)


class TestScaleByFactor:
    def test_scale_by_factor_scalar(self):
        """Scalar factor multiplies every element."""
        # Arrange
        X = np.array([[1.0, 2.0], [3.0, 4.0]])

        # Act
        result = scale_by_factor(X, 2.0)

        # Assert
        np.testing.assert_allclose(result, [[2.0, 4.0], [6.0, 8.0]])

    def test_scale_by_factor_per_sample(self):
        """Per-sample factor of shape (n_samples, 1)."""
        # Arrange
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        factor = np.array([[2.0], [0.5]])

        # Act
        result = scale_by_factor(X, factor)

        # Assert
        np.testing.assert_allclose(result, [[2.0, 4.0], [1.5, 2.0]])

    def test_scale_by_factor_1d_raises(self):
        """1-D array input raises ValueError — user must provide explicit shape."""
        # Arrange
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        factor = np.array([2.0, 0.5])  # shape (2,)

        # Act & Assert
        with pytest.raises(ValueError, match="Invalid metadata argument `factor`"):
            scale_by_factor(X, factor)


class TestAddOffset:
    def test_add_offset_shared(self):
        """Single shared offset broadcast across all samples."""
        # Arrange
        X = np.array([[1.0, 2.0, 3.0]])
        offset = np.array([[0.1, 0.2, 0.3]])

        # Act
        result = add_offset(X, offset)

        # Assert
        np.testing.assert_allclose(result, [[1.1, 2.2, 3.3]])

    def test_add_offset_scalar(self):
        """Scalar offset added to every element."""
        # Arrange
        X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        # Act
        result = add_offset(X, 0.5)

        # Assert
        np.testing.assert_allclose(result, [[1.5, 2.5, 3.5], [4.5, 5.5, 6.5]])

    def test_add_offset_per_sample(self):
        """Per-sample scalar offset of shape (n_samples, 1)."""
        # Arrange
        X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        offset = np.array([[0.5], [1.0]])

        # Act
        result = add_offset(X, offset)

        # Assert
        np.testing.assert_allclose(result, [[1.5, 2.5, 3.5], [5.0, 6.0, 7.0]])

    def test_add_offset_1d_raises(self):
        """1-D array input raises ValueError — user must provide explicit shape."""
        # Arrange
        X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        offset = np.array([0.5, 1.0])  # shape (2,)

        # Act & Assert
        with pytest.raises(ValueError, match="Invalid metadata argument `offset`"):
            add_offset(X, offset)
