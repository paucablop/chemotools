"""Tests for inspector utility functions."""

import numpy as np

from chemotools.inspector._utils import (
    normalize_datasets,
    normalize_components,
    get_xlabel_for_features,
    prepare_annotations,
)


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
