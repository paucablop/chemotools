"""Tests for core plot creation functions."""

import numpy as np
import pytest
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from chemotools.inspector._plot_utils_latent_space import (
    create_model_distances_plot,
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


@pytest.fixture
def sample_data_distances():
    """Create sample data for testing."""
    np.random.seed(42)
    return {
        "X": np.random.rand(50, 50),
        "y": np.random.randint(0, 3, 50),
    }


@pytest.fixture
def pca_model():
    """Create a fitted PCA model for testing."""
    np.random.seed(42)
    X = np.random.rand(100, 50)
    model = PCA(n_components=5)
    model.fit(X)
    return model


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


class TestCreateModelDistancesPlot:
    """Tests for the unified create_model_distances_plot function."""

    def test_single_dataset_with_targets(self, pca_model, sample_data_distances):
        """Single dataset renders successfully and adds colour mapping."""
        datasets = {
            "train": {"X": sample_data_distances["X"], "y": sample_data_distances["y"]}
        }

        fig = create_model_distances_plot(
            datasets_data=datasets,
            model=pca_model,
            confidence=0.95,
            color_by_y=True,
            figsize=(8, 6),
        )

        assert fig is not None
        assert len(fig.axes) == 1
        plt.close(fig)

    def test_single_dataset_without_targets(self, pca_model, sample_data_distances):
        """Single dataset works when targets are missing or skipped."""
        datasets = {"train": {"X": sample_data_distances["X"], "y": None}}

        fig = create_model_distances_plot(
            datasets_data=datasets,
            model=pca_model,
            confidence=0.95,
            color_by_y=False,
            figsize=(8, 6),
        )

        assert fig is not None
        plt.close(fig)

    def test_single_dataset_draws_confidence_lines(
        self, pca_model, sample_data_distances
    ):
        """Training-only plots include both confidence limits."""
        datasets = {"train": {"X": sample_data_distances["X"], "y": None}}

        fig = create_model_distances_plot(
            datasets_data=datasets,
            model=pca_model,
            confidence=0.99,
            color_by_y=False,
            figsize=(8, 6),
        )

        ax = fig.axes[0]
        dashed_lines = [line for line in ax.lines if line.get_linestyle() == "--"]
        assert len(dashed_lines) == 2
        plt.close(fig)

    def test_multiple_datasets(self, pca_model):
        """Multiple datasets are composed on the same axes."""
        datasets = {
            "train": {
                "X": np.random.rand(50, 50),
                "y": np.random.randint(0, 3, 50),
            },
            "test": {
                "X": np.random.rand(30, 50),
                "y": np.random.randint(0, 3, 30),
            },
        }

        fig = create_model_distances_plot(
            datasets_data=datasets,
            model=pca_model,
            confidence=0.95,
            color_by_y=False,
            figsize=(8, 6),
        )

        assert fig is not None
        assert len(fig.axes) == 1
        plt.close(fig)

    def test_multiple_datasets_only_train_has_confidence_lines(self, pca_model):
        """Confidence limits are drawn only for the training dataset."""
        datasets = {
            "train": {"X": np.random.rand(40, 50), "y": None},
            "test": {"X": np.random.rand(35, 50), "y": None},
            "val": {"X": np.random.rand(30, 50), "y": None},
        }

        fig = create_model_distances_plot(
            datasets_data=datasets,
            model=pca_model,
            confidence=0.95,
            color_by_y=False,
            figsize=(8, 6),
        )

        ax = fig.axes[0]
        dashed_lines = [line for line in ax.lines if line.get_linestyle() == "--"]
        assert len(dashed_lines) == 2
        plt.close(fig)

    def test_multiple_datasets_without_targets(self, pca_model):
        """Datasets lacking targets fall back to dataset colours."""
        datasets = {
            "train": {"X": np.random.rand(40, 50), "y": None},
            "val": {"X": np.random.rand(35, 50), "y": None},
        }

        fig = create_model_distances_plot(
            datasets_data=datasets,
            model=pca_model,
            confidence=0.95,
            color_by_y=True,
            figsize=(8, 6),
        )

        assert fig is not None
        plt.close(fig)

    def test_raises_with_missing_x(self, pca_model):
        """Missing X arrays raise an informative error."""
        datasets = {"train": {"X": None, "y": np.array([1, 2, 3])}}

        with pytest.raises(ValueError, match="X data is required"):
            create_model_distances_plot(
                datasets_data=datasets,
                model=pca_model,
                confidence=0.95,
                color_by_y=False,
                figsize=(8, 6),
            )

    def test_raises_with_no_datasets(self, pca_model):
        """Empty dataset mapping is rejected."""
        with pytest.raises(ValueError, match="must contain at least one dataset"):
            create_model_distances_plot(
                datasets_data={},
                model=pca_model,
                confidence=0.95,
                color_by_y=False,
                figsize=(8, 6),
            )
