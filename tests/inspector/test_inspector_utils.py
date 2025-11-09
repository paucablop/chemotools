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
        result = normalize_datasets("train")
        assert result == ["train"]

    def test_list_of_strings(self):
        """Test normalization of list of dataset names."""
        result = normalize_datasets(["train", "test"])
        assert result == ["train", "test"]

    def test_tuple_of_strings(self):
        """Test normalization of tuple of dataset names."""
        result = normalize_datasets(("train", "test", "val"))
        assert result == ["train", "test", "val"]


class TestNormalizeComponents:
    """Tests for normalize_components function."""

    def test_single_int(self):
        """Test normalization of single component index."""
        result = normalize_components(0)
        assert result == [0]

    def test_single_tuple_pair(self):
        """Test normalization of single component pair."""
        result = normalize_components((0, 1))
        assert result == [(0, 1)]

    def test_list_of_ints(self):
        """Test normalization of list of component indices."""
        result = normalize_components([0, 1, 2])
        assert result == [0, 1, 2]

    def test_tuple_of_pairs(self):
        """Test normalization of tuple of component pairs."""
        result = normalize_components(((0, 1), (1, 2)))
        assert result == [(0, 1), (1, 2)]

    def test_mixed_list(self):
        """Test normalization of mixed components."""
        result = normalize_components([0, (0, 1), 2])
        assert result == [0, (0, 1), 2]


class TestGetXlabelForFeatures:
    """Tests for get_xlabel_for_features function."""

    def test_with_wavenumbers(self):
        """Test label when wavenumbers are provided."""
        result = get_xlabel_for_features(wavenumbers_provided=True)
        assert "Wavenumber" in result
        assert "cm⁻¹" in result or "cm-1" in result

    def test_without_wavenumbers(self):
        """Test label when no wavenumbers are provided."""
        result = get_xlabel_for_features(wavenumbers_provided=False)
        assert "Feature" in result
        assert "Index" in result


class TestPrepareAnnotations:
    """Tests for prepare_annotations function."""

    def test_none_annotation(self):
        """Test with no annotations."""
        scores = np.random.rand(10, 3)
        result = prepare_annotations(None, "train", scores, None)
        assert result is None

    def test_sample_index_annotation(self):
        """Test with sample index annotations."""
        scores = np.random.rand(5, 3)
        result = prepare_annotations("sample_index", "train", scores, None)
        assert result is not None
        assert len(result) == 5
        assert list(result) == [0, 1, 2, 3, 4]

    def test_y_annotation_with_y_data(self):
        """Test with y value annotations."""
        scores = np.random.rand(5, 3)
        y = np.array([10, 20, 30, 40, 50])
        result = prepare_annotations("y", "train", scores, y)
        assert result is not None
        assert len(result) == 5
        np.testing.assert_array_equal(result, y)

    def test_y_annotation_without_y_data(self):
        """Test y annotation when y is None."""
        scores = np.random.rand(5, 3)
        result = prepare_annotations("y", "train", scores, None)
        assert result is None

    def test_dict_annotation_with_dataset(self):
        """Test with dictionary annotations for specific dataset."""
        scores = np.random.rand(5, 3)
        labels = np.array(["a", "b", "c", "d", "e"])
        result = prepare_annotations({"train": labels}, "train", scores, None)
        assert result is not None
        np.testing.assert_array_equal(result, labels)

    def test_dict_annotation_without_dataset(self):
        """Test with dictionary annotations for different dataset."""
        scores = np.random.rand(5, 3)
        labels = np.array(["a", "b", "c", "d", "e"])
        result = prepare_annotations({"test": labels}, "train", scores, None)
        assert result is None

    def test_invalid_annotation_type(self):
        """Test with invalid annotation type."""
        scores = np.random.rand(5, 3)
        result = prepare_annotations("invalid", "train", scores, None)
        assert result is None
