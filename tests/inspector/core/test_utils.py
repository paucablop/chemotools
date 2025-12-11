"""Tests for inspector utility functions."""

import pytest

import numpy as np

from chemotools.inspector.core.utils import (
    normalize_datasets,
    normalize_components,
    get_xlabel_for_features,
    prepare_annotations,
    select_components,
)


class TestSelectComponents:
    """Tests for select_components function."""

    def test_returns_full_matrix_when_components_is_none(self):
        """Test that full matrix is returned when components is None."""
        # Arrange
        matrix = np.array([[1, 2, 3], [4, 5, 6]])

        # Act
        result = select_components(matrix, None)

        # Assert
        np.testing.assert_array_equal(result, matrix)

    def test_selects_single_component_from_int(self):
        """Test selecting a single component using an integer."""
        # Arrange
        matrix = np.array([[1, 2, 3], [4, 5, 6]])

        # Act
        result = select_components(matrix, 0)

        # Assert
        expected = np.array([[1], [4]])
        np.testing.assert_array_equal(result, expected)

    def test_selects_single_component_from_list(self):
        """Test selecting a single component using a list."""
        # Arrange
        matrix = np.array([[1, 2, 3], [4, 5, 6]])

        # Act
        result = select_components(matrix, [1])

        # Assert
        expected = np.array([[2], [5]])
        np.testing.assert_array_equal(result, expected)

    def test_selects_multiple_components(self):
        """Test selecting multiple components."""
        # Arrange
        matrix = np.array([[1, 2, 3], [4, 5, 6]])

        # Act
        result = select_components(matrix, [0, 2])

        # Assert
        expected = np.array([[1, 3], [4, 6]])
        np.testing.assert_array_equal(result, expected)

    def test_preserves_order_of_components(self):
        """Test that component order is preserved."""
        # Arrange
        matrix = np.array([[1, 2, 3], [4, 5, 6]])

        # Act
        result = select_components(matrix, [2, 0])

        # Assert
        expected = np.array([[3, 1], [6, 4]])
        np.testing.assert_array_equal(result, expected)

    def test_works_with_tuple_of_components(self):
        """Test selecting components using a tuple."""
        # Arrange
        matrix = np.array([[1, 2, 3], [4, 5, 6]])

        # Act
        result = select_components(matrix, (0, 2))

        # Assert
        expected = np.array([[1, 3], [4, 6]])
        np.testing.assert_array_equal(result, expected)

    def test_raises_index_error_for_out_of_bounds_component(self):
        """Test that IndexError is raised for out of bounds component index."""

        # Arrange
        matrix = np.array([[1, 2, 3], [4, 5, 6]])  # 3 components (0, 1, 2)

        # Act & Assert
        with pytest.raises(IndexError, match="Component index 5 is out of bounds"):
            select_components(matrix, 5)

    def test_raises_index_error_for_negative_component(self):
        """Test that IndexError is raised for negative component index."""

        # Arrange
        matrix = np.array([[1, 2, 3], [4, 5, 6]])

        # Act & Assert
        with pytest.raises(IndexError, match="Component index -1 is out of bounds"):
            select_components(matrix, -1)

    def test_raises_index_error_for_out_of_bounds_in_list(self):
        """Test that IndexError is raised when any component in list is out of bounds."""
        # Arrange
        matrix = np.array([[1, 2, 3], [4, 5, 6]])  # 3 components (0, 1, 2)

        # Act & Assert
        with pytest.raises(IndexError, match="Component index 10 is out of bounds"):
            select_components(matrix, [0, 1, 10])

    def test_error_message_includes_valid_range(self):
        """Test that error message includes the valid range of components."""
        # Arrange
        matrix = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])  # 5 components

        # Act & Assert
        with pytest.raises(
            IndexError, match=r"Valid range: 0 to 4 \(model has 5 components\)"
        ):
            select_components(matrix, 99)


class TestNormalizeDatasets:
    """Tests for normalize_datasets function."""

    def test_single_string(self):
        """Test normalization of single dataset name."""
        # Arrange
        dataset = "train"

        # Act
        result = normalize_datasets(dataset)

        # Assert
        assert result == ["train"]

    def test_list_of_strings(self):
        """Test normalization of list of dataset names."""
        # Arrange
        datasets = ["train", "test"]

        # Act
        result = normalize_datasets(datasets)

        # Assert
        assert result == ["train", "test"]

    def test_tuple_of_strings(self):
        """Test normalization of tuple of dataset names."""
        # Arrange
        datasets = ("train", "test", "val")

        # Act
        result = normalize_datasets(datasets)

        # Assert
        assert result == ["train", "test", "val"]


class TestNormalizeComponents:
    """Tests for normalize_components function."""

    def test_single_int(self):
        """Test normalization of single component index."""
        # Arrange
        component = 0

        # Act
        result = normalize_components(component)

        # Assert
        assert result == [0]

    def test_single_tuple_pair(self):
        """Test normalization of single component pair."""
        # Arrange
        components = (0, 1)

        # Act
        result = normalize_components(components)

        # Assert
        assert result == [(0, 1)]

    def test_list_of_ints(self):
        """Test normalization of list of component indices."""
        # Arrange
        components = [0, 1, 2]

        # Act
        result = normalize_components(components)

        # Assert
        assert result == [0, 1, 2]

    def test_tuple_of_pairs(self):
        """Test normalization of tuple of component pairs."""
        # Arrange
        components = ((0, 1), (1, 2))

        # Act
        result = normalize_components(components)

        # Assert
        assert result == [(0, 1), (1, 2)]

    def test_mixed_list(self):
        """Test normalization of mixed components."""
        # Arrange
        components = [0, (0, 1), 2]

        # Act
        result = normalize_components(components)

        # Assert
        assert result == [0, (0, 1), 2]


class TestGetXlabelForFeatures:
    """Tests for get_xlabel_for_features function."""

    def test_with_wavenumbers(self):
        """Test label when wavenumbers are provided."""
        # Arrange
        wavenumbers_provided = True

        # Act
        result = get_xlabel_for_features(wavenumbers_provided=wavenumbers_provided)

        # Assert
        assert "Wavenumber" in result
        assert "cm⁻¹" in result or "cm-1" in result

    def test_without_wavenumbers(self):
        """Test label when no wavenumbers are provided."""
        # Arrange
        wavenumbers_provided = False

        # Act
        result = get_xlabel_for_features(wavenumbers_provided=wavenumbers_provided)

        # Assert
        assert "Feature" in result
        assert "Index" in result


class TestPrepareAnnotations:
    """Tests for prepare_annotations function."""

    def test_none_annotation(self):
        """Test with no annotations."""
        # Arrange
        scores = np.random.rand(10, 3)
        annotate_by = None
        dataset = "train"
        y = None

        # Act
        result = prepare_annotations(annotate_by, dataset, scores, y)

        # Assert
        assert result is None

    def test_sample_index_annotation(self):
        """Test with sample index annotations."""
        # Arrange
        scores = np.random.rand(5, 3)
        annotate_by = "sample_index"
        dataset = "train"
        y = None

        # Act
        result = prepare_annotations(annotate_by, dataset, scores, y)

        # Assert
        assert result is not None
        assert len(result) == 5
        assert list(result) == [0, 1, 2, 3, 4]

    def test_y_annotation_with_y_data(self):
        """Test with y value annotations."""
        # Arrange
        scores = np.random.rand(5, 3)
        annotate_by = "y"
        dataset = "train"
        y = np.array([10, 20, 30, 40, 50])

        # Act
        result = prepare_annotations(annotate_by, dataset, scores, y)

        # Assert
        assert result is not None
        assert len(result) == 5
        np.testing.assert_array_equal(result, y)

    def test_y_annotation_without_y_data(self):
        """Test y annotation when y is None."""
        # Arrange
        scores = np.random.rand(5, 3)
        annotate_by = "y"
        dataset = "train"
        y = None

        # Act
        result = prepare_annotations(annotate_by, dataset, scores, y)

        # Assert
        assert result is None

    def test_dict_annotation_with_dataset(self):
        """Test with dictionary annotations for specific dataset."""
        # Arrange
        scores = np.random.rand(5, 3)
        labels = np.array(["a", "b", "c", "d", "e"])
        annotate_by = {"train": labels}
        dataset = "train"
        y = None

        # Act
        result = prepare_annotations(annotate_by, dataset, scores, y)

        # Assert
        assert result is not None
        np.testing.assert_array_equal(result, labels)

    def test_dict_annotation_without_dataset(self):
        """Test with dictionary annotations for different dataset."""
        # Arrange
        scores = np.random.rand(5, 3)
        labels = np.array(["a", "b", "c", "d", "e"])
        annotate_by = {"test": labels}
        dataset = "train"
        y = None

        # Act
        result = prepare_annotations(annotate_by, dataset, scores, y)

        # Assert
        assert result is None

    def test_invalid_annotation_type(self):
        """Test with invalid annotation type."""
        # Arrange
        scores = np.random.rand(5, 3)
        annotate_by = "invalid"
        dataset = "train"
        y = None

        # Act
        result = prepare_annotations(annotate_by, dataset, scores, y)

        # Assert
        assert result is None
