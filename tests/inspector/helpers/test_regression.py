"""Tests for regression plot creation functions."""

import numpy as np
import pytest
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression

from chemotools.inspector.helpers._regression import (
    create_predicted_vs_actual_plot,
    create_y_residual_plot,
    create_qq_plot,
    create_residual_distribution_plot,
    create_regression_distances_plot,
)
from chemotools.outliers import Leverage, StudentizedResiduals


@pytest.fixture
def sample_regression_data():
    """Create sample regression data for testing."""
    np.random.seed(42)
    n_samples = 50
    n_features = 20

    X = np.random.rand(n_samples, n_features)
    y_true = np.random.rand(n_samples)
    y_pred = y_true + np.random.normal(0, 0.1, n_samples)  # Add some noise

    return {
        "X": X,
        "y_true": y_true,
        "y_pred": y_pred,
        "y": y_true,
    }


@pytest.fixture
def sample_detectors(sample_regression_data):
    """Create sample outlier detectors."""
    X = sample_regression_data["X"]
    y_true = sample_regression_data["y_true"]

    # Fit a simple PLS model
    model = PLSRegression(n_components=3)
    model.fit(X, y_true)

    # Create and fit detectors - both need the fitted model
    leverage_detector = Leverage(model, confidence=0.95)
    leverage_detector.fit(X)

    student_detector = StudentizedResiduals(model, confidence=0.95)
    student_detector.fit(X, y_true)

    return leverage_detector, student_detector


class TestCreatePredictedVsActualPlot:
    """Tests for create_predicted_vs_actual_plot function."""

    def test_single_dataset(self, sample_regression_data):
        """Test predicted vs actual plot for single dataset."""
        # Arrange
        datasets_data = {
            "train": {
                "y_true": sample_regression_data["y_true"],
                "y_pred": sample_regression_data["y_pred"],
                "y": sample_regression_data["y"],
            }
        }
        color_by = None
        figsize = (6, 6)

        # Act
        fig = create_predicted_vs_actual_plot(
            datasets_data=datasets_data,
            color_by=color_by,
            figsize=figsize,
        )

        # Assert
        assert fig is not None
        assert len(fig.axes) == 1

        # Cleanup
        plt.close(fig)

    def test_multi_dataset(self, sample_regression_data):
        """Test predicted vs actual plot for multiple datasets."""
        # Arrange
        datasets_data = {
            "train": {
                "y_true": sample_regression_data["y_true"],
                "y_pred": sample_regression_data["y_pred"],
                "y": sample_regression_data["y"],
            },
            "test": {
                "y_true": sample_regression_data["y_true"][:30],
                "y_pred": sample_regression_data["y_pred"][:30],
                "y": sample_regression_data["y"][:30],
            },
        }
        color_by = None
        figsize = (6, 6)

        # Act
        fig = create_predicted_vs_actual_plot(
            datasets_data=datasets_data,
            color_by=color_by,
            figsize=figsize,
        )

        # Assert
        assert fig is not None
        assert len(fig.axes) == 1

        # Cleanup
        plt.close(fig)

    def test_color_by_y(self, sample_regression_data):
        """Test predicted vs actual plot with y-coloring."""
        # Arrange
        datasets_data = {
            "train": {
                "y_true": sample_regression_data["y_true"],
                "y_pred": sample_regression_data["y_pred"],
                "y": sample_regression_data["y"],
            }
        }
        color_by = "y"
        figsize = (6, 6)

        # Act
        fig = create_predicted_vs_actual_plot(
            datasets_data=datasets_data,
            color_by=color_by,
            figsize=figsize,
        )

        # Assert
        assert fig is not None

        # Cleanup
        plt.close(fig)


class TestCreateYResidualPlot:
    """Tests for create_y_residual_plot function."""

    def test_single_dataset(self, sample_regression_data):
        """Test residual plot for single dataset."""
        # Arrange
        datasets_data = {
            "train": {
                "y_true": sample_regression_data["y_true"],
                "y_pred": sample_regression_data["y_pred"],
                "y": sample_regression_data["y"],
            }
        }
        color_by = None
        figsize = (6, 6)

        # Act
        fig = create_y_residual_plot(
            datasets_data=datasets_data,
            color_by=color_by,
            figsize=figsize,
        )

        # Assert
        assert fig is not None
        assert len(fig.axes) == 1

        # Cleanup
        plt.close(fig)

    def test_multi_dataset(self, sample_regression_data):
        """Test residual plot for multiple datasets (side-by-side)."""
        # Arrange
        datasets_data = {
            "train": {
                "y_true": sample_regression_data["y_true"],
                "y_pred": sample_regression_data["y_pred"],
                "y": sample_regression_data["y"],
            },
            "test": {
                "y_true": sample_regression_data["y_true"][:30],
                "y_pred": sample_regression_data["y_pred"][:30],
                "y": sample_regression_data["y"][:30],
            },
        }
        color_by = None
        figsize = (6, 6)

        # Act
        fig = create_y_residual_plot(
            datasets_data=datasets_data,
            color_by=color_by,
            figsize=figsize,
        )

        # Assert
        assert fig is not None
        assert len(fig.axes) == 2  # Side-by-side subplots

        # Cleanup
        plt.close(fig)


class TestCreateQQPlot:
    """Tests for create_qq_plot function."""

    def test_single_dataset(self, sample_regression_data):
        """Test Q-Q plot for single dataset."""
        # Arrange
        datasets_data = {
            "train": {
                "y_true": sample_regression_data["y_true"],
                "y_pred": sample_regression_data["y_pred"],
            }
        }
        figsize = (6, 6)

        # Act
        fig = create_qq_plot(
            datasets_data=datasets_data,
            figsize=figsize,
        )

        # Assert
        assert fig is not None
        assert len(fig.axes) == 1

        # Cleanup
        plt.close(fig)

    def test_multi_dataset(self, sample_regression_data):
        """Test Q-Q plot for multiple datasets."""
        # Arrange
        datasets_data = {
            "train": {
                "y_true": sample_regression_data["y_true"],
                "y_pred": sample_regression_data["y_pred"],
            },
            "test": {
                "y_true": sample_regression_data["y_true"][:30],
                "y_pred": sample_regression_data["y_pred"][:30],
            },
        }
        figsize = (6, 6)

        # Act
        fig = create_qq_plot(
            datasets_data=datasets_data,
            figsize=figsize,
        )

        # Assert
        assert fig is not None
        assert len(fig.axes) == 2  # Side-by-side subplots

        # Cleanup
        plt.close(fig)


class TestCreateResidualDistributionPlot:
    """Tests for create_residual_distribution_plot function."""

    def test_single_dataset(self, sample_regression_data):
        """Test residual distribution plot for single dataset."""
        # Arrange
        datasets_data = {
            "train": {
                "y_true": sample_regression_data["y_true"],
                "y_pred": sample_regression_data["y_pred"],
            }
        }
        figsize = (6, 6)

        # Act
        fig = create_residual_distribution_plot(
            datasets_data=datasets_data,
            figsize=figsize,
        )

        # Assert
        assert fig is not None
        assert len(fig.axes) == 1

        # Cleanup
        plt.close(fig)

    def test_multi_dataset(self, sample_regression_data):
        """Test residual distribution plot for multiple datasets."""
        # Arrange
        datasets_data = {
            "train": {
                "y_true": sample_regression_data["y_true"],
                "y_pred": sample_regression_data["y_pred"],
            },
            "test": {
                "y_true": sample_regression_data["y_true"][:30],
                "y_pred": sample_regression_data["y_pred"][:30],
            },
        }
        figsize = (6, 6)

        # Act
        fig = create_residual_distribution_plot(
            datasets_data=datasets_data,
            figsize=figsize,
        )

        # Assert
        assert fig is not None
        assert len(fig.axes) == 2  # Side-by-side subplots

        # Cleanup
        plt.close(fig)


class TestCreateRegressionDistancesPlot:
    """Tests for create_regression_distances_plot function."""

    def test_single_dataset(self, sample_regression_data, sample_detectors):
        """Test regression distances plot for single dataset."""
        # Arrange
        leverage_detector, student_detector = sample_detectors
        X = sample_regression_data["X"]
        y_true = sample_regression_data["y_true"]

        color_by = None
        figsize = (6, 6)

        # Act
        fig = create_regression_distances_plot(
            X=X,
            y_true=y_true,
            leverage_detector=leverage_detector,
            student_detector=student_detector,
            color_by=color_by,
            figsize=figsize,
        )

        # Assert
        assert fig is not None
        assert len(fig.axes) == 1

        # Cleanup
        plt.close(fig)

    def test_uses_distances_plot_class(self, sample_regression_data, sample_detectors):
        """Test that the function uses DistancesPlot class."""
        # Arrange
        leverage_detector, student_detector = sample_detectors
        X = sample_regression_data["X"]
        y_true = sample_regression_data["y_true"]

        color_by = None
        figsize = (6, 6)

        # Act
        fig = create_regression_distances_plot(
            X=X,
            y_true=y_true,
            leverage_detector=leverage_detector,
            student_detector=student_detector,
            color_by=color_by,
            figsize=figsize,
        )

        # Assert
        assert fig is not None
        # Check that confidence lines are present
        ax = fig.axes[0]
        # Should have vertical and horizontal lines for confidence limits
        assert len(ax.lines) > 0

        # Cleanup
        plt.close(fig)
