"""Tests for core plot creation functions."""

import numpy as np
import pytest
import matplotlib.pyplot as plt

from chemotools.inspector._plot_core import (
    create_variance_plot,
    create_loadings_plot,
    create_scores_plot_single_dataset,
    create_scores_plot_multi_dataset,
)


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    np.random.seed(42)
    return {
        "scores": np.random.rand(50, 5),
        "loadings": np.random.rand(100, 5),
        "explained_var": np.array([0.45, 0.25, 0.15, 0.10, 0.05]),
        "wavenumbers": np.linspace(4000, 400, 100),
        "y": np.random.randint(0, 3, 50),
    }


class TestCreateVariancePlot:
    """Tests for create_variance_plot function."""

    def test_basic_variance_plot(self, sample_data):
        """Test basic variance plot creation."""
        fig = create_variance_plot(
            explained_variance_ratio=sample_data["explained_var"],
            variance_threshold=0.95,
            figsize=(10, 5),
        )
        assert fig is not None
        assert len(fig.axes) == 1
        plt.close(fig)

    def test_different_threshold(self, sample_data):
        """Test with different variance threshold."""
        fig = create_variance_plot(
            explained_variance_ratio=sample_data["explained_var"],
            variance_threshold=0.90,
            figsize=(10, 5),
        )
        assert fig is not None
        plt.close(fig)


class TestCreateLoadingsPlot:
    """Tests for create_loadings_plot function."""

    def test_single_component(self, sample_data):
        """Test loadings plot for single component."""
        fig = create_loadings_plot(
            loadings=sample_data["loadings"],
            feature_names=sample_data["wavenumbers"],
            loadings_components=0,
            xlabel="Wavenumber (cm⁻¹)",
            figsize=(10, 5),
        )
        assert fig is not None
        assert len(fig.axes) == 1
        plt.close(fig)

    def test_multiple_components(self, sample_data):
        """Test loadings plot for multiple components."""
        fig = create_loadings_plot(
            loadings=sample_data["loadings"],
            feature_names=sample_data["wavenumbers"],
            loadings_components=[0, 1, 2],
            xlabel="Wavenumber (cm⁻¹)",
            figsize=(10, 5),
        )
        assert fig is not None
        plt.close(fig)

    def test_with_feature_indices(self, sample_data):
        """Test loadings plot with feature indices instead of wavenumbers."""
        fig = create_loadings_plot(
            loadings=sample_data["loadings"],
            feature_names=np.arange(100),
            loadings_components=0,
            xlabel="Feature Index",
            figsize=(10, 5),
        )
        assert fig is not None
        plt.close(fig)


class TestCreateScoresPlotSingleDataset:
    """Tests for create_scores_plot_single_dataset function."""

    def test_2d_plot(self, sample_data):
        """Test 2D scores plot."""
        fig = create_scores_plot_single_dataset(
            component_spec=(0, 1),
            scores=sample_data["scores"],
            y=sample_data["y"],
            explained_var=sample_data["explained_var"],
            dataset_name="train",
            color_by_y=True,
            annotate_by=None,
            figsize=(6, 6),
        )
        assert fig is not None
        assert len(fig.axes) == 1
        plt.close(fig)

    def test_1d_plot(self, sample_data):
        """Test 1D scores plot."""
        fig = create_scores_plot_single_dataset(
            component_spec=0,
            scores=sample_data["scores"],
            y=sample_data["y"],
            explained_var=sample_data["explained_var"],
            dataset_name="train",
            color_by_y=False,
            annotate_by=None,
            figsize=(6, 6),
        )
        assert fig is not None
        plt.close(fig)

    def test_without_y(self, sample_data):
        """Test scores plot without y data."""
        fig = create_scores_plot_single_dataset(
            component_spec=(0, 1),
            scores=sample_data["scores"],
            y=None,
            explained_var=sample_data["explained_var"],
            dataset_name="train",
            color_by_y=False,
            annotate_by=None,
            figsize=(6, 6),
        )
        assert fig is not None
        plt.close(fig)

    def test_with_annotations(self, sample_data):
        """Test scores plot with annotations."""
        fig = create_scores_plot_single_dataset(
            component_spec=(0, 1),
            scores=sample_data["scores"],
            y=sample_data["y"],
            explained_var=sample_data["explained_var"],
            dataset_name="train",
            color_by_y=False,
            annotate_by="sample_index",
            figsize=(6, 6),
        )
        assert fig is not None
        plt.close(fig)


class TestCreateScoresPlotMultiDataset:
    """Tests for create_scores_plot_multi_dataset function."""

    def test_2d_multi_dataset(self, sample_data):
        """Test 2D multi-dataset scores plot."""
        datasets_data = {
            "train": {
                "scores": sample_data["scores"],
                "y": sample_data["y"],
            },
            "test": {
                "scores": np.random.rand(30, 5),
                "y": np.random.randint(0, 3, 30),
            },
        }

        fig = create_scores_plot_multi_dataset(
            component_spec=(0, 1),
            datasets_data=datasets_data,
            explained_var=sample_data["explained_var"],
            color_by_y=False,
            annotate_by=None,
            figsize=(6, 6),
        )
        assert fig is not None
        assert len(fig.axes) == 1
        plt.close(fig)

    def test_1d_multi_dataset(self, sample_data):
        """Test 1D multi-dataset scores plot."""
        datasets_data = {
            "train": {
                "scores": sample_data["scores"],
                "y": None,
            },
            "test": {
                "scores": np.random.rand(30, 5),
                "y": None,
            },
        }

        fig = create_scores_plot_multi_dataset(
            component_spec=0,
            datasets_data=datasets_data,
            explained_var=sample_data["explained_var"],
            color_by_y=False,
            annotate_by=None,
            figsize=(6, 6),
        )
        assert fig is not None
        plt.close(fig)

    def test_with_none_y_values(self, sample_data):
        """Test multi-dataset plot with None y values."""
        datasets_data = {
            "train": {
                "scores": sample_data["scores"],
                "y": None,
            },
        }

        fig = create_scores_plot_multi_dataset(
            component_spec=(0, 1),
            datasets_data=datasets_data,
            explained_var=sample_data["explained_var"],
            color_by_y=False,
            annotate_by=None,
            figsize=(6, 6),
        )
        assert fig is not None
        plt.close(fig)
