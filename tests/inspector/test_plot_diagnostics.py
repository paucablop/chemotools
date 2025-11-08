"""Tests for diagnostic plot creation functions."""

import numpy as np
import pytest
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from chemotools.inspector._plot_diagnostics import (
    create_distances_plot_single_dataset,
    create_distances_plot_multi_dataset,
)


@pytest.fixture
def pca_model():
    """Create a fitted PCA model for testing."""
    np.random.seed(42)
    X = np.random.rand(100, 50)
    model = PCA(n_components=5)
    model.fit(X)
    return model


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    np.random.seed(42)
    return {
        "X": np.random.rand(50, 50),
        "y": np.random.randint(0, 3, 50),
    }


class TestCreateDistancesPlotSingleDataset:
    """Tests for create_distances_plot_single_dataset function."""

    def test_basic_distances_plot(self, pca_model, sample_data):
        """Test basic distances plot creation."""
        fig = create_distances_plot_single_dataset(
            X=sample_data["X"],
            y=sample_data["y"],
            model=pca_model,
            confidence=0.95,
            dataset_name="train",
            color_by_y=True,
            figsize=(8, 6),
        )
        assert fig is not None
        assert len(fig.axes) == 1
        plt.close(fig)

    def test_without_y(self, pca_model, sample_data):
        """Test distances plot without y data."""
        fig = create_distances_plot_single_dataset(
            X=sample_data["X"],
            y=None,
            model=pca_model,
            confidence=0.95,
            dataset_name="train",
            color_by_y=False,
            figsize=(8, 6),
        )
        assert fig is not None
        plt.close(fig)

    def test_different_confidence(self, pca_model, sample_data):
        """Test with different confidence level."""
        fig = create_distances_plot_single_dataset(
            X=sample_data["X"],
            y=sample_data["y"],
            model=pca_model,
            confidence=0.99,
            dataset_name="test",
            color_by_y=False,
            figsize=(8, 6),
        )
        assert fig is not None
        plt.close(fig)


class TestCreateDistancesPlotMultiDataset:
    """Tests for create_distances_plot_multi_dataset function."""

    def test_multi_dataset_distances(self, pca_model):
        """Test multi-dataset distances plot."""
        datasets_data = {
            "train": {
                "X": np.random.rand(50, 50),
                "y": np.random.randint(0, 3, 50),
            },
            "test": {
                "X": np.random.rand(30, 50),
                "y": np.random.randint(0, 3, 30),
            },
        }

        fig = create_distances_plot_multi_dataset(
            datasets_data=datasets_data,
            model=pca_model,
            confidence=0.95,
            color_by_y=False,
            figsize=(8, 6),
        )
        assert fig is not None
        assert len(fig.axes) == 1
        plt.close(fig)

    def test_with_none_y(self, pca_model):
        """Test multi-dataset plot with None y values."""
        datasets_data = {
            "train": {
                "X": np.random.rand(50, 50),
                "y": None,
            },
            "test": {
                "X": np.random.rand(30, 50),
                "y": None,
            },
        }

        fig = create_distances_plot_multi_dataset(
            datasets_data=datasets_data,
            model=pca_model,
            confidence=0.95,
            color_by_y=False,
            figsize=(8, 6),
        )
        assert fig is not None
        plt.close(fig)

    def test_single_dataset_multi(self, pca_model):
        """Test multi-dataset function with single dataset."""
        datasets_data = {
            "train": {
                "X": np.random.rand(50, 50),
                "y": np.random.randint(0, 3, 50),
            },
        }

        fig = create_distances_plot_multi_dataset(
            datasets_data=datasets_data,
            model=pca_model,
            confidence=0.95,
            color_by_y=True,
            figsize=(8, 6),
        )
        assert fig is not None
        plt.close(fig)
