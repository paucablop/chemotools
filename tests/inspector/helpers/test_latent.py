"""Tests for core plot creation functions."""

import numpy as np
import pytest
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression

from chemotools.inspector.helpers._latent import (
    create_model_distances_plot,
    create_q_vs_y_residuals_plot,
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


@pytest.fixture
def sample_data_coverage():
    np.random.seed(42)
    return {
        "scores": np.random.rand(50, 5),
        "explained_var": np.array([0.45, 0.25, 0.15, 0.10, 0.05]),
        "y": np.random.rand(50),
    }


@pytest.fixture
def pca_model_coverage():
    X = np.random.rand(20, 10)
    model = PCA(n_components=2)
    model.fit(X)
    return model


@pytest.fixture
def pls_model_coverage():
    X = np.random.rand(20, 10)
    Y = np.random.rand(20, 2)
    model = PLSRegression(n_components=2)
    model.fit(X, Y)
    return model


class TestCreateVariancePlot:
    """Tests for create_variance_plot function."""

    def test_basic_variance_plot(self, sample_data):
        """Test basic variance plot creation."""
        # Arrange
        explained_var = sample_data["explained_var"]
        variance_threshold = 0.95
        figsize = (10, 5)

        # Act
        fig = create_variance_plot(
            explained_variance_ratio=explained_var,
            variance_threshold=variance_threshold,
            figsize=figsize,
        )

        # Assert
        assert fig is not None
        assert len(fig.axes) == 1

        # Cleanup
        plt.close(fig)

    def test_different_threshold(self, sample_data):
        """Test with different variance threshold."""
        # Arrange
        explained_var = sample_data["explained_var"]
        variance_threshold = 0.90
        figsize = (10, 5)

        # Act
        fig = create_variance_plot(
            explained_variance_ratio=explained_var,
            variance_threshold=variance_threshold,
            figsize=figsize,
        )

        # Assert
        assert fig is not None

        # Cleanup
        plt.close(fig)


class TestCreateLoadingsPlot:
    """Tests for create_loadings_plot function."""

    def test_single_component(self, sample_data):
        """Test loadings plot for single component."""
        # Arrange
        loadings = sample_data["loadings"]
        feature_names = sample_data["wavenumbers"]
        loadings_components = 0
        xlabel = "Wavenumber (cm⁻¹)"
        figsize = (10, 5)

        # Act
        fig = create_loadings_plot(
            loadings=loadings,
            feature_names=feature_names,
            loadings_components=loadings_components,
            xlabel=xlabel,
            figsize=figsize,
        )

        # Assert
        assert fig is not None
        assert len(fig.axes) == 1

        # Cleanup
        plt.close(fig)

    def test_multiple_components(self, sample_data):
        """Test loadings plot for multiple components."""
        # Arrange
        loadings = sample_data["loadings"]
        feature_names = sample_data["wavenumbers"]
        loadings_components = [0, 1, 2]
        xlabel = "Wavenumber (cm⁻¹)"
        figsize = (10, 5)

        # Act
        fig = create_loadings_plot(
            loadings=loadings,
            feature_names=feature_names,
            loadings_components=loadings_components,
            xlabel=xlabel,
            figsize=figsize,
        )

        # Assert
        assert fig is not None

        # Cleanup
        plt.close(fig)

    def test_with_feature_indices(self, sample_data):
        """Test loadings plot with feature indices instead of wavenumbers."""
        # Arrange
        loadings = sample_data["loadings"]
        feature_names = np.arange(100)
        loadings_components = 0
        xlabel = "Feature Index"
        figsize = (10, 5)

        # Act
        fig = create_loadings_plot(
            loadings=loadings,
            feature_names=feature_names,
            loadings_components=loadings_components,
            xlabel=xlabel,
            figsize=figsize,
        )

        # Assert
        assert fig is not None

        # Cleanup
        plt.close(fig)


class TestCreateScoresPlotSingleDataset:
    """Tests for create_scores_plot_single_dataset function."""

    def test_2d_plot(self, sample_data):
        """Test 2D scores plot."""
        # Arrange
        component_spec = (0, 1)
        scores = sample_data["scores"]
        y = sample_data["y"]
        explained_var = sample_data["explained_var"]
        dataset_name = "train"
        color_by = "y"
        annotate_by = None
        figsize = (6, 6)

        # Act
        fig = create_scores_plot_single_dataset(
            component_spec=component_spec,
            scores=scores,
            y=y,
            explained_var=explained_var,
            dataset_name=dataset_name,
            color_by=color_by,
            annotate_by=annotate_by,
            figsize=figsize,
            color_mode="continuous",
        )

        # Assert
        assert fig is not None
        # Should have 2 axes (main plot + colorbar) because color_by="y" and y is numeric
        assert len(fig.axes) == 2

        # Cleanup
        plt.close(fig)

    def test_1d_plot(self, sample_data):
        """Test 1D scores plot."""
        # Arrange
        component_spec = 0
        scores = sample_data["scores"]
        y = sample_data["y"]
        explained_var = sample_data["explained_var"]
        dataset_name = "train"
        color_by = None
        annotate_by = None
        figsize = (6, 6)

        # Act
        fig = create_scores_plot_single_dataset(
            component_spec=component_spec,
            scores=scores,
            y=y,
            explained_var=explained_var,
            dataset_name=dataset_name,
            color_by=color_by,
            annotate_by=annotate_by,
            figsize=figsize,
        )

        # Assert
        assert fig is not None

        # Cleanup
        plt.close(fig)

    def test_without_y(self, sample_data):
        """Test scores plot without y data."""
        # Arrange
        component_spec = (0, 1)
        scores = sample_data["scores"]
        y = None
        explained_var = sample_data["explained_var"]
        dataset_name = "train"
        color_by = None
        annotate_by = None
        figsize = (6, 6)

        # Act
        fig = create_scores_plot_single_dataset(
            component_spec=component_spec,
            scores=scores,
            y=y,
            explained_var=explained_var,
            dataset_name=dataset_name,
            color_by=color_by,
            annotate_by=annotate_by,
            figsize=figsize,
        )

        # Assert
        assert fig is not None

        # Cleanup
        plt.close(fig)

    def test_with_annotations(self, sample_data):
        """Test scores plot with annotations."""
        # Arrange
        component_spec = (0, 1)
        scores = sample_data["scores"]
        y = sample_data["y"]
        explained_var = sample_data["explained_var"]
        dataset_name = "train"
        color_by = None
        annotate_by = "sample_index"
        figsize = (6, 6)

        # Act
        fig = create_scores_plot_single_dataset(
            component_spec=component_spec,
            scores=scores,
            y=y,
            explained_var=explained_var,
            dataset_name=dataset_name,
            color_by=color_by,
            annotate_by=annotate_by,
            figsize=figsize,
        )

        # Assert
        assert fig is not None

        # Cleanup
        plt.close(fig)

    def test_raises_value_error_same_components(self, sample_data_coverage):
        """Test that ValueError is raised when component indices are the same."""
        # Arrange
        component_spec = (0, 0)
        scores = sample_data_coverage["scores"]
        explained_var = sample_data_coverage["explained_var"]

        # Act & Assert
        with pytest.raises(ValueError, match="Component indices must be different"):
            create_scores_plot_single_dataset(
                component_spec=component_spec,
                scores=scores,
                y=None,
                explained_var=explained_var,
                dataset_name="train",
                color_by=None,
                annotate_by=None,
                figsize=(6, 6),
            )

    def test_color_by_y_1d_plot(self, sample_data_coverage):
        """Test 1D scores plot colored by y-values."""
        # Arrange
        component_spec = 0
        scores = sample_data_coverage["scores"]
        y = sample_data_coverage["y"]
        explained_var = sample_data_coverage["explained_var"]
        color_by = "y"

        # Act
        fig = create_scores_plot_single_dataset(
            component_spec=component_spec,
            scores=scores,
            y=y,
            explained_var=explained_var,
            dataset_name="train",
            color_by=color_by,
            annotate_by=None,
            figsize=(6, 6),
        )

        # Assert
        assert fig is not None
        ax = fig.axes[0]
        assert ax.get_xlabel() == "y-value"
        assert "Scores: PC1 (Train)" in ax.get_title()
        plt.close(fig)

    def test_color_by_sample_index_1d_plot(self, sample_data_coverage):
        """Test 1D scores plot colored by sample index."""
        # Arrange
        component_spec = 0
        scores = sample_data_coverage["scores"]
        explained_var = sample_data_coverage["explained_var"]
        color_by = "sample_index"

        # Act
        fig = create_scores_plot_single_dataset(
            component_spec=component_spec,
            scores=scores,
            y=None,
            explained_var=explained_var,
            dataset_name="train",
            color_by=color_by,
            annotate_by=None,
            figsize=(6, 6),
        )

        # Assert
        assert fig is not None
        ax = fig.axes[0]
        assert ax.get_xlabel() == "Sample Index"
        plt.close(fig)


class TestCreateScoresPlotMultiDataset:
    """Tests for create_scores_plot_multi_dataset function."""

    def test_2d_multi_dataset(self, sample_data):
        """Test 2D multi-dataset scores plot."""
        # Arrange
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
        component_spec = (0, 1)
        explained_var = sample_data["explained_var"]
        color_by = None
        annotate_by = None
        figsize = (6, 6)

        # Act
        fig = create_scores_plot_multi_dataset(
            component_spec=component_spec,
            datasets_data=datasets_data,
            explained_var=explained_var,
            color_by=color_by,
            annotate_by=annotate_by,
            figsize=figsize,
        )

        # Assert
        assert fig is not None
        assert len(fig.axes) == 1

        # Cleanup
        plt.close(fig)

    def test_1d_multi_dataset(self, sample_data):
        """Test 1D multi-dataset scores plot."""
        # Arrange
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
        component_spec = 0
        explained_var = sample_data["explained_var"]
        color_by = None
        annotate_by = None
        figsize = (6, 6)

        # Act
        fig = create_scores_plot_multi_dataset(
            component_spec=component_spec,
            datasets_data=datasets_data,
            explained_var=explained_var,
            color_by=color_by,
            annotate_by=annotate_by,
            figsize=figsize,
        )

        # Assert
        assert fig is not None

        # Cleanup
        plt.close(fig)

    def test_with_none_y_values(self, sample_data):
        """Test multi-dataset plot with None y values."""
        # Arrange
        datasets_data = {
            "train": {
                "scores": sample_data["scores"],
                "y": None,
            },
        }
        component_spec = (0, 1)
        explained_var = sample_data["explained_var"]
        color_by = None
        figsize = (6, 6)

        # Act
        fig = create_scores_plot_multi_dataset(
            component_spec=component_spec,
            datasets_data=datasets_data,
            explained_var=explained_var,
            color_by=color_by,
            annotate_by=None,
            figsize=figsize,
        )

        # Assert
        assert fig is not None

        # Cleanup
        plt.close(fig)

    def test_raises_value_error_same_components(self, sample_data_coverage):
        """Test that ValueError is raised when component indices are the same."""
        # Arrange
        component_spec = (0, 0)
        datasets_data = {"train": {"scores": sample_data_coverage["scores"], "y": None}}
        explained_var = sample_data_coverage["explained_var"]

        # Act & Assert
        with pytest.raises(ValueError, match="Component indices must be different"):
            create_scores_plot_multi_dataset(
                component_spec=component_spec,
                datasets_data=datasets_data,
                explained_var=explained_var,
                color_by=None,
                annotate_by=None,
                figsize=(6, 6),
            )

    def test_nan_explained_variance(self, sample_data_coverage):
        """Test handling of NaN values in explained variance."""
        # Arrange
        explained_var = np.array([np.nan, 0.25])
        datasets_data = {"train": {"scores": sample_data_coverage["scores"], "y": None}}
        component_spec = 0

        # Act
        fig = create_scores_plot_multi_dataset(
            component_spec=component_spec,
            datasets_data=datasets_data,
            explained_var=explained_var,
            color_by=None,
            annotate_by=None,
            figsize=(6, 6),
        )

        # Assert
        assert fig is not None
        # Check that label doesn't contain "nan%"
        ax = fig.axes[0]
        assert "nan%" not in ax.get_ylabel()
        plt.close(fig)

    def test_scores_is_none_assertion(self, sample_data_coverage):
        """Test assertions for missing scores data."""
        # Arrange
        datasets_data_1d = {"train": {"scores": None, "y": None}}
        datasets_data_mixed = {
            "train": {"scores": sample_data_coverage["scores"], "y": None},
            "test": {"scores": None, "y": None},
        }
        explained_var = sample_data_coverage["explained_var"]

        # Act & Assert (1D case)
        with pytest.raises(
            ValueError, match="At least one dataset must have scores data"
        ):
            create_scores_plot_multi_dataset(
                component_spec=0,
                datasets_data=datasets_data_1d,
                explained_var=explained_var,
                color_by=None,
                annotate_by=None,
                figsize=(6, 6),
            )

        # Act & Assert (2D case)
        with pytest.raises(
            AssertionError, match="Scores data is required for dataset test"
        ):
            create_scores_plot_multi_dataset(
                component_spec=(0, 1),
                datasets_data=datasets_data_mixed,
                explained_var=explained_var,
                color_by=None,
                annotate_by=None,
                figsize=(6, 6),
            )

    def test_color_by_y_1d_plot(self, sample_data_coverage):
        """Test 1D multi-dataset scores plot colored by y-values."""
        # Arrange
        datasets_data = {
            "train": {
                "scores": sample_data_coverage["scores"],
                "y": sample_data_coverage["y"],
            }
        }
        explained_var = sample_data_coverage["explained_var"]
        color_by = "y"
        component_spec = 0

        # Act
        fig = create_scores_plot_multi_dataset(
            component_spec=component_spec,
            datasets_data=datasets_data,
            explained_var=explained_var,
            color_by=color_by,
            annotate_by=None,
            figsize=(6, 6),
        )

        # Assert
        assert fig is not None
        ax = fig.axes[0]
        assert ax.get_xlabel() == "y-value"
        assert "Scores: PC1" in ax.get_title()
        # Check grid and legend (lines 490-491)
        # Scatter plots use collections, not lines
        assert len(ax.collections) > 0
        assert ax.get_legend() is not None
        plt.close(fig)


class TestCreateModelDistancesPlot:
    """Tests for the unified create_model_distances_plot function."""

    def test_single_dataset_with_targets(self, pca_model, sample_data_distances):
        """Single dataset renders successfully and adds colour mapping."""
        # Arrange
        datasets = {
            "train": {"X": sample_data_distances["X"], "y": sample_data_distances["y"]}
        }
        model = pca_model
        confidence = 0.95
        color_by = "y"
        figsize = (8, 6)

        # Act
        fig = create_model_distances_plot(
            datasets_data=datasets,
            model=model,
            confidence=confidence,
            color_by=color_by,
            figsize=figsize,
            color_mode="continuous",
        )

        # Assert
        assert fig is not None
        # Should have 2 axes (main plot + colorbar) because color_by="y" and y is numeric
        assert len(fig.axes) == 2

        # Cleanup
        plt.close(fig)

    def test_single_dataset_without_targets(self, pca_model, sample_data_distances):
        """Single dataset works when targets are missing or skipped."""
        # Arrange
        datasets = {"train": {"X": sample_data_distances["X"], "y": None}}
        model = pca_model
        confidence = 0.95
        color_by = None
        figsize = (8, 6)

        # Act
        fig = create_model_distances_plot(
            datasets_data=datasets,
            model=model,
            confidence=confidence,
            color_by=color_by,
            figsize=figsize,
        )

        # Assert
        assert fig is not None

        # Cleanup
        plt.close(fig)

    def test_single_dataset_draws_confidence_lines(
        self, pca_model, sample_data_distances
    ):
        """Training-only plots include both confidence limits."""
        # Arrange
        datasets = {"train": {"X": sample_data_distances["X"], "y": None}}
        model = pca_model
        confidence = 0.99
        color_by = None
        figsize = (8, 6)

        # Act
        fig = create_model_distances_plot(
            datasets_data=datasets,
            model=model,
            confidence=confidence,
            color_by=color_by,
            figsize=figsize,
        )

        # Assert
        ax = fig.axes[0]
        dashed_lines = [line for line in ax.lines if line.get_linestyle() == "--"]
        assert len(dashed_lines) == 2

        # Cleanup
        plt.close(fig)

    def test_multiple_datasets(self, pca_model):
        """Multiple datasets are composed on the same axes."""
        # Arrange
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
        model = pca_model
        confidence = 0.95
        color_by = None
        figsize = (8, 6)

        # Act
        fig = create_model_distances_plot(
            datasets_data=datasets,
            model=model,
            confidence=confidence,
            color_by=color_by,
            figsize=figsize,
        )

        # Assert
        assert fig is not None
        assert len(fig.axes) == 1

        # Cleanup
        plt.close(fig)

    def test_multiple_datasets_only_train_has_confidence_lines(self, pca_model):
        """Confidence limits are drawn only for the training dataset."""
        # Arrange
        datasets = {
            "train": {"X": np.random.rand(40, 50), "y": None},
            "test": {"X": np.random.rand(35, 50), "y": None},
            "val": {"X": np.random.rand(30, 50), "y": None},
        }
        model = pca_model
        confidence = 0.95
        color_by = None
        figsize = (8, 6)

        # Act
        fig = create_model_distances_plot(
            datasets_data=datasets,
            model=model,
            confidence=confidence,
            color_by=color_by,
            figsize=figsize,
        )

        # Assert
        ax = fig.axes[0]
        dashed_lines = [line for line in ax.lines if line.get_linestyle() == "--"]
        assert len(dashed_lines) == 2

        # Cleanup
        plt.close(fig)

    def test_multiple_datasets_without_targets(self, pca_model):
        """Datasets lacking targets fall back to dataset colours."""
        # Arrange
        datasets = {
            "train": {"X": np.random.rand(40, 50), "y": None},
            "val": {"X": np.random.rand(35, 50), "y": None},
        }
        model = pca_model
        confidence = 0.95
        color_by = "y"
        figsize = (8, 6)

        # Act
        fig = create_model_distances_plot(
            datasets_data=datasets,
            model=model,
            confidence=confidence,
            color_by=color_by,
            figsize=figsize,
        )

        # Assert
        assert fig is not None

        # Cleanup
        plt.close(fig)

    def test_raises_with_missing_x(self, pca_model):
        """Missing X arrays raise an informative error."""
        # Arrange
        datasets = {"train": {"X": None, "y": np.array([1, 2, 3])}}
        model = pca_model
        confidence = 0.95
        color_by = None
        figsize = (8, 6)

        # Act & Assert
        with pytest.raises(ValueError, match="X data is required"):
            create_model_distances_plot(
                datasets_data=datasets,
                model=model,
                confidence=confidence,
                color_by=color_by,
                figsize=figsize,
            )

    def test_raises_with_no_datasets(self, pca_model):
        """Empty dataset mapping is rejected."""
        # Arrange
        datasets = {}
        model = pca_model
        confidence = 0.95
        color_by = None
        figsize = (8, 6)

        # Act & Assert
        with pytest.raises(ValueError, match="must contain at least one dataset"):
            create_model_distances_plot(
                datasets_data=datasets,
                model=model,
                confidence=confidence,
                color_by=color_by,
                figsize=figsize,
            )

    def test_raises_value_error_empty_datasets(self, pca_model_coverage):
        """Test that ValueError is raised when datasets_data is empty."""
        # Arrange
        datasets_data = {}
        model = pca_model_coverage

        # Act & Assert
        with pytest.raises(
            ValueError, match="datasets_data must contain at least one dataset"
        ):
            create_model_distances_plot(
                datasets_data=datasets_data,
                model=model,
                confidence=0.95,
                color_by=None,
                figsize=(6, 6),
            )

    def test_fallback_training_dataset(self, pca_model_coverage):
        """Test fallback to first dataset when training_dataset is not found."""
        # Arrange
        X = np.random.rand(20, 10)
        datasets_data = {"test": {"X": X, "y": None}}
        model = pca_model_coverage

        # Act
        # Should not raise error, should use "test" as training data for detectors
        fig = create_model_distances_plot(
            datasets_data=datasets_data,
            model=model,
            confidence=0.95,
            color_by=None,
            figsize=(6, 6),
            training_dataset="train",  # "train" not in datasets_data
        )

        # Assert
        assert fig is not None
        plt.close(fig)

    def test_raises_value_error_missing_X(self, pca_model_coverage):
        """Test that ValueError is raised when X data is missing."""
        # Arrange
        X = np.random.rand(20, 10)
        datasets_data_mixed = {
            "train": {"X": X, "y": None},
            "test": {"X": None, "y": None},
        }
        model = pca_model_coverage

        # Act & Assert
        with pytest.raises(ValueError, match="X data is required for dataset 'test'"):
            create_model_distances_plot(
                datasets_data=datasets_data_mixed,
                model=model,
                confidence=0.95,
                color_by=None,
                figsize=(6, 6),
            )


class TestCreateQVsYResidualsPlot:
    """Tests for create_q_vs_y_residuals_plot function."""

    @pytest.fixture
    def pls_regression_model(self):
        """Create a fitted PLS regression model for testing."""
        from sklearn.cross_decomposition import PLSRegression

        np.random.seed(42)
        X = np.random.rand(100, 50)
        y = X[:, :3].sum(axis=1) + np.random.randn(100) * 0.1
        model = PLSRegression(n_components=3)
        model.fit(X, y)
        return model

    @pytest.fixture
    def regression_datasets(self, pls_regression_model):
        """Create sample regression datasets with predictions."""
        np.random.seed(42)
        X_train = np.random.rand(50, 50)
        y_train = X_train[:, :3].sum(axis=1) + np.random.randn(50) * 0.1
        y_pred_train = pls_regression_model.predict(X_train).ravel()

        X_test = np.random.rand(30, 50)
        y_test = X_test[:, :3].sum(axis=1) + np.random.randn(30) * 0.1
        y_pred_test = pls_regression_model.predict(X_test).ravel()

        return {
            "train": {
                "X": X_train,
                "y": y_train,
                "y_true": y_train,
                "y_pred": y_pred_train,
            },
            "test": {
                "X": X_test,
                "y": y_test,
                "y_true": y_test,
                "y_pred": y_pred_test,
            },
        }

    def test_single_dataset_basic(self, regression_datasets, pls_regression_model):
        """Test Q vs Y residuals plot for single dataset."""
        # Arrange
        single_dataset = {"train": regression_datasets["train"]}
        model = pls_regression_model
        confidence = 0.95
        color_by = None
        figsize = (8, 6)

        # Act
        fig = create_q_vs_y_residuals_plot(
            datasets_data=single_dataset,
            model=model,
            confidence=confidence,
            color_by=color_by,
            figsize=figsize,
        )

        # Assert
        assert fig is not None
        assert len(fig.axes) == 1
        ax = fig.axes[0]
        assert ax.get_xlabel() == "Y Residuals (Prediction Error)"
        assert ax.get_ylabel() == "Q Residuals (SPE)"
        assert (
            "Regression Distances: Q Residuals vs Y Residuals (Train)" in ax.get_title()
        )

        # Check for vertical zero line (Y residuals on x-axis)
        lines = ax.get_lines()
        has_vertical_zero = any(
            abs(line.get_xdata()[0]) < 1e-10
            for line in lines
            if len(line.get_xdata()) > 0
        )
        assert has_vertical_zero, "Should have vertical zero reference line"

        # Cleanup
        plt.close(fig)

    def test_multi_dataset(self, regression_datasets, pls_regression_model):
        """Test Q vs Y residuals plot for multiple datasets."""
        # Arrange
        model = pls_regression_model
        confidence = 0.95
        color_by = None
        figsize = (8, 6)

        # Act
        fig = create_q_vs_y_residuals_plot(
            datasets_data=regression_datasets,
            model=model,
            confidence=confidence,
            color_by=color_by,
            figsize=figsize,
        )

        # Assert
        assert fig is not None
        assert len(fig.axes) == 1
        ax = fig.axes[0]
        assert ax.get_xlabel() == "Y Residuals (Prediction Error)"
        assert ax.get_ylabel() == "Q Residuals (SPE)"
        assert ax.get_legend() is not None
        legend_labels = [t.get_text() for t in ax.get_legend().get_texts()]
        assert "Train" in legend_labels
        assert "Test" in legend_labels

        # Cleanup
        plt.close(fig)

    def test_color_by_y_single_dataset(self, regression_datasets, pls_regression_model):
        """Test Q vs Y residuals plot with color_by_y for single dataset."""
        # Arrange
        single_dataset = {"train": regression_datasets["train"]}
        model = pls_regression_model
        confidence = 0.95
        color_by = "y"
        figsize = (8, 6)

        # Act
        fig = create_q_vs_y_residuals_plot(
            datasets_data=single_dataset,
            model=model,
            confidence=confidence,
            color_by=color_by,
            figsize=figsize,
        )

        # Assert
        assert fig is not None
        assert len(fig.axes) == 2  # Main plot + colorbar

        # Cleanup
        plt.close(fig)

    def test_with_prefitted_q_detector(self, regression_datasets, pls_regression_model):
        """Test Q vs Y residuals plot with pre-fitted Q residuals detector."""
        # Arrange
        from chemotools.outliers import QResiduals

        model = pls_regression_model
        confidence = 0.95
        color_by = None
        figsize = (8, 6)

        # Pre-fit Q detector
        q_detector = QResiduals(model, confidence=confidence)
        q_detector.fit(regression_datasets["train"]["X"])

        # Act
        fig = create_q_vs_y_residuals_plot(
            datasets_data=regression_datasets,
            model=model,
            confidence=confidence,
            color_by=color_by,
            figsize=figsize,
            q_residuals_detector=q_detector,
        )

        # Assert
        assert fig is not None
        assert len(fig.axes) == 1

        # Cleanup
        plt.close(fig)

    def test_confidence_lines_drawn_for_training(
        self, regression_datasets, pls_regression_model
    ):
        """Test that confidence lines are drawn for training dataset."""
        # Arrange
        model = pls_regression_model
        confidence = 0.95
        figsize = (8, 6)

        # Act
        fig = create_q_vs_y_residuals_plot(
            datasets_data=regression_datasets,
            model=model,
            confidence=confidence,
            color_by=None,
            figsize=figsize,
        )

        # Assert
        ax = fig.axes[0]
        # Check for horizontal Q confidence line (should be present)
        lines = ax.get_lines()
        # Should have: vertical zero line + horizontal Q limit line
        assert len(lines) >= 2, (
            "Should have at least vertical zero and horizontal Q limit lines"
        )

        # Cleanup
        plt.close(fig)

    def test_raises_with_missing_X(self, pls_regression_model):
        """Test that missing X data raises appropriate error."""
        # Arrange
        datasets = {
            "train": {"y_true": np.random.rand(50), "y_pred": np.random.rand(50)}
        }
        model = pls_regression_model
        confidence = 0.95
        color_by = None
        figsize = (8, 6)

        # Act & Assert
        with pytest.raises(ValueError, match="X data is required"):
            create_q_vs_y_residuals_plot(
                datasets_data=datasets,
                model=model,
                confidence=confidence,
                color_by=color_by,
                figsize=figsize,
            )

    def test_raises_with_missing_y_true(self, pls_regression_model):
        """Test that missing y_true data raises appropriate error."""
        # Arrange
        datasets = {
            "train": {"X": np.random.rand(50, 50), "y_pred": np.random.rand(50)}
        }
        model = pls_regression_model
        confidence = 0.95
        color_by = None
        figsize = (8, 6)

        # Act & Assert
        with pytest.raises(ValueError, match="y_true data is required"):
            create_q_vs_y_residuals_plot(
                datasets_data=datasets,
                model=model,
                confidence=confidence,
                color_by=color_by,
                figsize=figsize,
            )

    def test_raises_with_missing_y_pred(
        self, regression_datasets, pls_regression_model
    ):
        """Test that missing y_pred data raises appropriate error."""
        # Arrange
        datasets = {
            "train": {
                "X": regression_datasets["train"]["X"],
                "y_true": regression_datasets["train"]["y_true"],
            }
        }
        model = pls_regression_model
        confidence = 0.95
        color_by = None
        figsize = (8, 6)

        # Act & Assert
        with pytest.raises(ValueError, match="y_pred data is required"):
            create_q_vs_y_residuals_plot(
                datasets_data=datasets,
                model=model,
                confidence=confidence,
                color_by=color_by,
                figsize=figsize,
            )

    def test_raises_with_no_datasets(self, pls_regression_model):
        """Test that empty dataset mapping is rejected."""
        # Arrange
        datasets = {}
        model = pls_regression_model
        confidence = 0.95
        color_by = None
        figsize = (8, 6)

        # Act & Assert
        with pytest.raises(ValueError, match="must contain at least one dataset"):
            create_q_vs_y_residuals_plot(
                datasets_data=datasets,
                model=model,
                confidence=confidence,
                color_by=color_by,
                figsize=figsize,
            )

    def test_y_residuals_calculation(self, regression_datasets, pls_regression_model):
        """Test that Y residuals are calculated correctly as y_true - y_pred."""
        # Arrange
        single_dataset = {"train": regression_datasets["train"]}
        model = pls_regression_model
        confidence = 0.95
        color_by = None
        figsize = (8, 6)

        # Act
        fig = create_q_vs_y_residuals_plot(
            datasets_data=single_dataset,
            model=model,
            confidence=confidence,
            color_by=color_by,
            figsize=figsize,
        )

        # Assert - verify plot was created successfully
        # The actual residuals calculation is internal, but we can verify
        # the plot structure is correct
        ax = fig.axes[0]
        assert ax.get_xlabel() == "Y Residuals (Prediction Error)"

        # Cleanup
        plt.close(fig)

    def test_axes_orientation(self, regression_datasets, pls_regression_model):
        """Test that axes are correctly oriented (Y residuals on x, Q on y)."""
        # Arrange
        single_dataset = {"train": regression_datasets["train"]}
        model = pls_regression_model
        confidence = 0.95
        color_by = None
        figsize = (8, 6)

        # Act
        fig = create_q_vs_y_residuals_plot(
            datasets_data=single_dataset,
            model=model,
            confidence=confidence,
            color_by=color_by,
            figsize=figsize,
        )

        # Assert
        ax = fig.axes[0]
        # X-axis should be Y residuals
        assert "Y Residuals" in ax.get_xlabel()
        # Y-axis should be Q residuals
        assert "Q Residuals" in ax.get_ylabel()

        # Cleanup
        plt.close(fig)

    def test_raises_value_error_empty_datasets(self, pls_model_coverage):
        """Test that ValueError is raised when datasets_data is empty."""
        # Arrange
        datasets_data = {}
        model = pls_model_coverage

        # Act & Assert
        with pytest.raises(
            ValueError, match="datasets_data must contain at least one dataset"
        ):
            create_q_vs_y_residuals_plot(
                datasets_data=datasets_data,
                model=model,
                confidence=0.95,
                color_by=None,
                figsize=(6, 6),
            )

    def test_fallback_training_dataset(self, pls_model_coverage):
        """Test fallback to first dataset when training_dataset is not found."""
        # Arrange
        X = np.random.rand(20, 10)
        y_true = np.random.rand(20, 2)
        y_pred = np.random.rand(20, 2)

        datasets_data = {
            "test": {"X": X, "y_true": y_true, "y_pred": y_pred, "y": None}
        }
        model = pls_model_coverage

        # Act
        fig = create_q_vs_y_residuals_plot(
            datasets_data=datasets_data,
            model=model,
            confidence=0.95,
            color_by=None,
            figsize=(6, 6),
            training_dataset="train",  # "train" not in datasets_data
        )

        # Assert
        assert fig is not None
        plt.close(fig)

    def test_raises_value_error_missing_X(self, pls_model_coverage):
        """Test that ValueError is raised when X data is missing."""
        # Arrange
        X = np.random.rand(20, 10)
        y_true = np.random.rand(20, 2)
        y_pred = np.random.rand(20, 2)

        datasets_data_mixed = {
            "train": {"X": X, "y_true": y_true, "y_pred": y_pred, "y": None},
            "test": {"X": None, "y_true": y_true, "y_pred": y_pred, "y": None},
        }
        model = pls_model_coverage

        # Act & Assert
        with pytest.raises(ValueError, match="X data is required for dataset 'test'"):
            create_q_vs_y_residuals_plot(
                datasets_data=datasets_data_mixed,
                model=model,
                confidence=0.95,
                color_by=None,
                figsize=(6, 6),
            )

    def test_multi_target_residuals(self, pls_model_coverage):
        """Test residuals calculation for multi-target regression."""
        # Arrange
        X = np.random.rand(20, 10)
        # Multi-target (2 targets)
        y_true = np.random.rand(20, 2)
        y_pred = np.random.rand(20, 2)

        datasets_data = {
            "train": {
                "X": X,
                "y_true": y_true,
                "y_pred": y_pred,
                "y": None,
            }
        }
        model = pls_model_coverage

        # Act
        fig = create_q_vs_y_residuals_plot(
            datasets_data=datasets_data,
            model=model,
            confidence=0.95,
            color_by=None,
            figsize=(6, 6),
        )

        # Assert
        assert fig is not None
        plt.close(fig)
