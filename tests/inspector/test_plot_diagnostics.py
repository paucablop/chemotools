"""Tests for diagnostic plot creation functions."""

import numpy as np
import pytest
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from chemotools.inspector._plot_diagnostics import create_model_distances_plot


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


class TestCreateModelDistancesPlot:
    """Tests for the unified create_model_distances_plot function."""

    def test_single_dataset_with_targets(self, pca_model, sample_data):
        """Single dataset renders successfully and adds colour mapping."""
        datasets = {"train": {"X": sample_data["X"], "y": sample_data["y"]}}

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

    def test_single_dataset_without_targets(self, pca_model, sample_data):
        """Single dataset works when targets are missing or skipped."""
        datasets = {"train": {"X": sample_data["X"], "y": None}}

        fig = create_model_distances_plot(
            datasets_data=datasets,
            model=pca_model,
            confidence=0.95,
            color_by_y=False,
            figsize=(8, 6),
        )

        assert fig is not None
        plt.close(fig)

    def test_single_dataset_draws_confidence_lines(self, pca_model, sample_data):
        """Training-only plots include both confidence limits."""
        datasets = {"train": {"X": sample_data["X"], "y": None}}

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
