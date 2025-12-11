"""Comprehensive test suite for PCAInspector class."""

import numpy as np
import pytest
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from chemotools.inspector import PCAInspector
from chemotools.inspector.core.summaries import InspectorSummary


class TestPCAInspectorInitialization:
    """Test PCAInspector initialization."""

    def test_init_with_fitted_pca_only_train(self, fitted_pca, dummy_data_loader):
        """Test initi        # Assert
        assert len(figures) == 4  # 1 scores + loadings + variance + distances
        assert "scores_1" in figures
        assert "loadings" in figures
        assert "variance" in figures
        assert "distances" in figurestion with only training data."""
        # Arrange
        X, y = dummy_data_loader

        # Act
        inspector = PCAInspector(model=fitted_pca, X_train=X, y_train=y)

        # Assert
        assert inspector.estimator is fitted_pca
        assert inspector.transformer is None
        assert inspector.n_components == 2
        assert inspector.n_features == 3
        assert inspector.n_samples == {"train": 100}
        assert inspector.x_axis.shape == (3,)
        np.testing.assert_array_equal(inspector.x_axis, np.arange(3))

    def test_init_with_pipeline(self, fitted_pipeline_pca, dummy_data_loader):
        """Test initialization with fitted pipeline."""
        # Arrange
        X, y = dummy_data_loader

        # Act
        inspector = PCAInspector(model=fitted_pipeline_pca, X_train=X, y_train=y)

        # Assert
        assert isinstance(inspector.estimator, PCA)
        assert inspector.transformer is not None
        assert len(inspector.transformer.steps) == 1
        assert isinstance(inspector.transformer.steps[0][1], StandardScaler)

    def test_init_with_test_data(self, fitted_pca, dummy_data_loader):
        """Test initialization with train and test data."""
        # Arrange
        X, y = dummy_data_loader
        X_train, X_test = X[:80], X[80:]
        y_train, y_test = y[:80], y[80:]

        # Act
        inspector = PCAInspector(
            model=fitted_pca,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
        )

        # Assert
        assert inspector.n_samples == {"train": 80, "test": 20}

    def test_init_with_validation_data(self, fitted_pca, dummy_data_loader):
        """Test initialization with train, test, and validation data."""
        # Arrange
        X, y = dummy_data_loader
        X_train, X_test, X_val = X[:60], X[60:80], X[80:]
        y_train, y_test, y_val = y[:60], y[60:80], y[80:]

        # Act
        inspector = PCAInspector(
            model=fitted_pca,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            X_val=X_val,
            y_val=y_val,
        )

        # Assert
        assert inspector.n_samples == {"train": 60, "test": 20, "val": 20}

    def test_init_with_custom_wavenumbers(self, fitted_pca, dummy_data_loader):
        """Test initialization with custom wavenumbers."""
        # Arrange
        X, y = dummy_data_loader
        wavenumbers = np.array([1000, 1500, 2000])

        # Act
        inspector = PCAInspector(
            model=fitted_pca, X_train=X, y_train=y, x_axis=wavenumbers
        )

        # Assert
        np.testing.assert_array_equal(inspector.x_axis, wavenumbers)

    def test_init_without_y(self, fitted_pca, dummy_data_loader):
        """Test initialization without y values (unsupervised)."""
        # Arrange
        X, _ = dummy_data_loader

        # Act
        inspector = PCAInspector(model=fitted_pca, X_train=X)

        # Assert
        assert inspector.n_samples == {"train": 100}

    def test_init_wavenumbers_length_mismatch_raises_error(
        self, fitted_pca, dummy_data_loader
    ):
        """Test that wavenumbers length mismatch raises ValueError."""
        # Arrange
        X, y = dummy_data_loader
        wavenumbers = np.array([1000, 1500])  # Wrong length

        # Assert
        with pytest.raises(ValueError, match="x_axis length"):
            PCAInspector(model=fitted_pca, X_train=X, y_train=y, x_axis=wavenumbers)


class TestPCAInspectorProperties:
    """Test PCAInspector properties."""

    def test_estimator_property(self, fitted_pca, dummy_data_loader):
        """Test estimator property returns correct model."""
        # Arrange
        X, _ = dummy_data_loader
        inspector = PCAInspector(model=fitted_pca, X_train=X)

        # Act & Assert
        assert inspector.estimator is fitted_pca

    def test_transformer_property_with_pipeline(
        self, fitted_pipeline_pca, dummy_data_loader
    ):
        """Test transformer property with pipeline."""
        # Arrange
        X, _ = dummy_data_loader
        inspector = PCAInspector(model=fitted_pipeline_pca, X_train=X)

        # Act & Assert
        assert inspector.transformer is not None
        assert len(inspector.transformer.steps) == 1

    def test_transformer_property_without_pipeline(self, fitted_pca, dummy_data_loader):
        """Test transformer property without pipeline."""
        # Arrange
        X, _ = dummy_data_loader
        inspector = PCAInspector(model=fitted_pca, X_train=X)

        # Act & Assert
        assert inspector.transformer is None

    def test_n_components_property(self, fitted_pca, dummy_data_loader):
        """Test n_components property."""
        # Arrange
        X, _ = dummy_data_loader
        inspector = PCAInspector(model=fitted_pca, X_train=X)

        # Act & Assert
        assert inspector.n_components == 2

    def test_n_features_property(self, fitted_pca, dummy_data_loader):
        """Test n_features property."""
        # Arrange
        X, _ = dummy_data_loader
        inspector = PCAInspector(model=fitted_pca, X_train=X)

        # Act & Assert
        assert inspector.n_features == 3

    def test_n_samples_property_single_dataset(self, fitted_pca, dummy_data_loader):
        """Test n_samples property with single dataset."""
        # Arrange
        X, _ = dummy_data_loader
        inspector = PCAInspector(model=fitted_pca, X_train=X)

        # Act & Assert
        assert inspector.n_samples == {"train": 100}

    def test_n_samples_property_multiple_datasets(self, fitted_pca, dummy_data_loader):
        """Test n_samples property with multiple datasets."""
        # Arrange
        X, _ = dummy_data_loader
        X_train, X_test = X[:80], X[80:]
        inspector = PCAInspector(model=fitted_pca, X_train=X_train, X_test=X_test)

        # Act & Assert
        assert inspector.n_samples == {"train": 80, "test": 20}


class TestPCAInspectorGetScores:
    """Test get_scores method."""

    def test_get_scores_train(self, fitted_pca, dummy_data_loader):
        """Test get_scores for training data."""
        # Arrange
        X, _ = dummy_data_loader
        inspector = PCAInspector(model=fitted_pca, X_train=X)

        # Act
        scores = inspector.get_scores("train")

        # Assert
        assert scores.shape == (100, 2)
        assert isinstance(scores, np.ndarray)

    def test_get_scores_test(self, fitted_pca, dummy_data_loader):
        """Test get_scores for test data."""
        # Arrange
        X, _ = dummy_data_loader
        X_train, X_test = X[:80], X[80:]
        inspector = PCAInspector(model=fitted_pca, X_train=X_train, X_test=X_test)

        # Act
        scores = inspector.get_scores("test")

        # Assert
        assert scores.shape == (20, 2)

    def test_get_scores_val(self, fitted_pca, dummy_data_loader):
        """Test get_scores for validation data."""
        # Arrange
        X, _ = dummy_data_loader
        X_train, X_val = X[:80], X[80:]
        inspector = PCAInspector(model=fitted_pca, X_train=X_train, X_val=X_val)

        # Act
        scores = inspector.get_scores("val")

        # Assert
        assert scores.shape == (20, 2)

    def test_get_scores_caching(self, fitted_pca, dummy_data_loader):
        """Test that scores are cached after first call."""
        # Arrange
        X, _ = dummy_data_loader
        inspector = PCAInspector(model=fitted_pca, X_train=X)

        # Act
        scores1 = inspector.get_scores("train")
        scores2 = inspector.get_scores("train")

        # Assert - same object should be returned (cached)
        assert scores1 is scores2

    def test_get_scores_invalid_dataset_raises_error(
        self, fitted_pca, dummy_data_loader
    ):
        """Test that invalid dataset raises ValueError."""
        # Arrange
        X, _ = dummy_data_loader
        inspector = PCAInspector(model=fitted_pca, X_train=X)

        # Assert
        with pytest.raises(ValueError, match="Invalid dataset"):
            inspector.get_scores("invalid")

    def test_get_scores_missing_test_raises_error(self, fitted_pca, dummy_data_loader):
        """Test that requesting missing test data raises ValueError."""
        # Arrange
        X, _ = dummy_data_loader
        inspector = PCAInspector(model=fitted_pca, X_train=X)

        # Assert
        with pytest.raises(ValueError, match="Test data not provided"):
            inspector.get_scores("test")

    def test_get_scores_with_preprocessing(
        self, fitted_pipeline_pca, dummy_data_loader
    ):
        """Test get_scores with preprocessing pipeline."""
        # Arrange
        X, _ = dummy_data_loader
        inspector = PCAInspector(model=fitted_pipeline_pca, X_train=X)

        # Act
        scores = inspector.get_scores("train")

        # Assert
        assert scores.shape == (100, 2)
        # Verify preprocessing was applied (scores should be different from raw)
        raw_scores = fitted_pipeline_pca[-1].transform(X)
        assert not np.allclose(scores, raw_scores)


class TestPCAInspectorGetLoadings:
    """Test get_loadings method."""

    def test_get_loadings_all_components(self, fitted_pca, dummy_data_loader):
        """Test get_loadings without specifying components."""
        # Arrange
        X, _ = dummy_data_loader
        inspector = PCAInspector(model=fitted_pca, X_train=X)

        # Act
        loadings = inspector.get_loadings()

        # Assert
        assert loadings.shape == (3, 2)  # (n_features, n_components)

    def test_get_loadings_single_component_int(self, fitted_pca, dummy_data_loader):
        """Test get_loadings with single component as int."""
        # Arrange
        X, _ = dummy_data_loader
        inspector = PCAInspector(model=fitted_pca, X_train=X)

        # Act
        loadings = inspector.get_loadings(components=0)

        # Assert
        assert loadings.shape == (3, 1)

    def test_get_loadings_multiple_components(self, fitted_pca, dummy_data_loader):
        """Test get_loadings with multiple components."""
        # Arrange
        X, _ = dummy_data_loader
        inspector = PCAInspector(model=fitted_pca, X_train=X)

        # Act
        loadings = inspector.get_loadings(components=[0, 1])

        # Assert
        assert loadings.shape == (3, 2)

    def test_get_loadings_subset_components(self, dummy_data_loader):
        """Test get_loadings with subset of components."""
        # Arrange
        X, _ = dummy_data_loader
        pca = PCA(n_components=3).fit(X)
        inspector = PCAInspector(model=pca, X_train=X)

        # Act
        loadings = inspector.get_loadings(components=[0, 2])

        # Assert
        assert loadings.shape == (3, 2)


class TestPCAInspectorGetExplainedVarianceRatio:
    """Test get_explained_variance_ratio method."""

    def test_get_explained_variance_ratio(self, fitted_pca, dummy_data_loader):
        """Test get_explained_variance_ratio returns correct values."""
        # Arrange
        X, _ = dummy_data_loader
        inspector = PCAInspector(model=fitted_pca, X_train=X)

        # Act
        var_ratio = inspector.get_explained_variance_ratio()

        # Assert
        assert var_ratio.shape == (2,)
        assert np.all(var_ratio >= 0)
        assert np.all(var_ratio <= 1)
        assert np.sum(var_ratio) <= 1.0

    def test_explained_variance_matches_model(self, fitted_pca, dummy_data_loader):
        """Test that explained variance matches the model's values."""
        # Arrange
        X, _ = dummy_data_loader
        inspector = PCAInspector(model=fitted_pca, X_train=X)

        # Act
        var_ratio = inspector.get_explained_variance_ratio()

        # Assert
        np.testing.assert_array_equal(var_ratio, fitted_pca.explained_variance_ratio_)


class TestPCAInspectorSummary:
    """Test summary method."""

    def test_summary_runs_without_error(self, fitted_pca, dummy_data_loader):
        """Test that summary method runs without error."""
        # Arrange
        X, y = dummy_data_loader
        inspector = PCAInspector(model=fitted_pca, X_train=X, y_train=y)

        # Act & Assert - should not raise
        result = inspector.summary()
        assert isinstance(result, InspectorSummary)

    def test_summary_with_pipeline(self, fitted_pipeline_pca, dummy_data_loader):
        """Test summary with preprocessing pipeline."""
        # Arrange
        X, y = dummy_data_loader
        inspector = PCAInspector(model=fitted_pipeline_pca, X_train=X, y_train=y)

        # Act & Assert - should not raise
        result = inspector.summary()
        assert isinstance(result, InspectorSummary)
        assert result.preprocessing_steps is not None
        assert len(result.preprocessing_steps) > 0

    def test_summary_with_multiple_datasets(self, fitted_pca, dummy_data_loader):
        """Test summary with multiple datasets."""
        # Arrange
        X, y = dummy_data_loader
        X_train, X_test, X_val = X[:60], X[60:80], X[80:]
        y_train, y_test, y_val = y[:60], y[60:80], y[80:]
        inspector = PCAInspector(
            model=fitted_pca,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            X_val=X_val,
            y_val=y_val,
        )

        # Act & Assert - should not raise
        result = inspector.summary()
        assert isinstance(result, InspectorSummary)
        assert result.n_samples is not None
        assert len(result.n_samples) == 3  # train, test, val

    def test_summary_output_format(self, fitted_pca, dummy_data_loader):
        """Test that summary returns dictionary with expected keys."""
        # Arrange
        X, _ = dummy_data_loader
        inspector = PCAInspector(model=fitted_pca, X_train=X)

        # Act
        result = inspector.summary()

        # Assert - check for key content
        assert isinstance(result, InspectorSummary)
        assert result.model_type is not None
        assert result.has_preprocessing is not None
        assert result.n_features is not None
        assert result.n_components is not None
        assert result.n_samples is not None
        assert result.explained_variance_ratio is not None
        assert result.cumulative_variance is not None
        assert result.pc_variances is not None
        assert result.total_variance is not None
        assert result.variance_thresholds is not None
        assert result.preprocessing_steps is not None

        # Check variance thresholds structure
        assert "90%" in result.variance_thresholds
        assert "95%" in result.variance_thresholds
        assert "99%" in result.variance_thresholds

        # Check each threshold has required keys
        for threshold in ["90%", "95%", "99%"]:
            assert "n_components" in result.variance_thresholds[threshold]
            assert "actual_variance" in result.variance_thresholds[threshold]

    def test_summary_includes_latent_info(self, fitted_pca, dummy_data_loader):
        """Test that summary includes latent variable information."""
        # Arrange
        X, y = dummy_data_loader
        inspector = PCAInspector(model=fitted_pca, X_train=X, y_train=y)

        # Act
        summary = inspector.summary()

        # Assert
        assert summary.n_components == 2
        assert isinstance(summary.hotelling_t2_limit, float)
        assert isinstance(summary.q_residuals_limit, float)


class TestPCAInspectorInspect:
    """Test inspect method."""

    def test_inspect_returns_dict_of_figures(self, fitted_pca, dummy_data_loader):
        """Test that inspect returns dictionary of matplotlib figures."""
        # Arrange
        X, y = dummy_data_loader
        inspector = PCAInspector(model=fitted_pca, X_train=X, y_train=y)

        # Act
        figures = inspector.inspect(
            components_scores=(0, 1), loadings_components=[0, 1]
        )

        # Assert
        assert isinstance(figures, dict)
        assert "scores_1" in figures
        assert "loadings" in figures
        assert "variance" in figures
        assert all(isinstance(fig, plt.Figure) for fig in figures.values())

    def test_inspect_default_creates_expected_plots(
        self, fitted_pca, dummy_data_loader
    ):
        """Test inspect with defaults creates expected plots."""
        # Arrange
        X, y = dummy_data_loader
        inspector = PCAInspector(model=fitted_pca, X_train=X, y_train=y)

        # Act
        figures = inspector.inspect(
            components_scores=(0, 1), loadings_components=[0, 1]
        )

        # Assert
        # Single scores plot + loadings + variance + distances
        assert len(figures) == 4
        assert "scores_1" in figures
        assert "loadings" in figures
        assert "variance" in figures
        assert "distances" in figures

    def test_inspect_single_2d_scores_plot(self, fitted_pca, dummy_data_loader):
        """Test inspect with single 2D scores plot."""
        # Arrange
        X, y = dummy_data_loader
        inspector = PCAInspector(model=fitted_pca, X_train=X, y_train=y)

        # Act
        figures = inspector.inspect(
            components_scores=(0, 1), loadings_components=[0, 1]
        )

        # Assert
        assert len(figures) == 4  # 1 scores + loadings + variance + distances
        assert "scores_1" in figures
        assert "loadings" in figures
        assert "variance" in figures
        assert "distances" in figures

    def test_inspect_single_1d_scores_plot(self, fitted_pca, dummy_data_loader):
        """Test inspect with single 1D scores plot."""
        # Arrange
        X, y = dummy_data_loader
        inspector = PCAInspector(model=fitted_pca, X_train=X, y_train=y)

        # Act
        figures = inspector.inspect(components_scores=0, loadings_components=[0, 1])

        # Assert
        assert len(figures) == 4  # 1 scores + loadings + variance + distances
        assert "scores_1" in figures
        assert "distances" in figures

    def test_inspect_multiple_mixed_scores_plots(self, fitted_pca, dummy_data_loader):
        """Test inspect with mixed 1D and 2D scores plots."""
        # Arrange
        X, y = dummy_data_loader
        inspector = PCAInspector(model=fitted_pca, X_train=X, y_train=y)

        # Act
        figures = inspector.inspect(
            components_scores=[(0, 1), 0, 1],
            loadings_components=[0, 1],
        )

        # Assert
        assert len(figures) == 6  # 3 scores + loadings + variance + distances
        assert "scores_1" in figures
        assert "scores_2" in figures
        assert "scores_3" in figures
        assert "loadings" in figures
        assert "variance" in figures
        assert "distances" in figures
        assert "scores_2" in figures
        assert "scores_3" in figures

    def test_inspect_custom_loadings_components(self, fitted_pca, dummy_data_loader):
        """Test inspect with custom loadings components."""
        # Arrange
        X, y = dummy_data_loader
        inspector = PCAInspector(model=fitted_pca, X_train=X, y_train=y)

        # Act
        figures = inspector.inspect(components_scores=(0, 1), loadings_components=0)

        # Assert
        assert "loadings" in figures

    def test_inspect_without_color_by_y(self, fitted_pca, dummy_data_loader):
        """Test inspect without coloring by y values."""
        # Arrange
        X, y = dummy_data_loader
        inspector = PCAInspector(model=fitted_pca, X_train=X, y_train=y)

        # Act
        figures = inspector.inspect(
            components_scores=(0, 1),
            loadings_components=[0, 1],
            color_by_y=False,
        )

        # Assert
        assert len(figures) == 4
        assert "distances" in figures

    def test_inspect_without_y_values(self, fitted_pca, dummy_data_loader):
        """Test inspect works without y values."""
        # Arrange
        X, _ = dummy_data_loader
        inspector = PCAInspector(model=fitted_pca, X_train=X)

        # Act
        figures = inspector.inspect(
            components_scores=(0, 1), loadings_components=[0, 1]
        )

        # Assert
        assert len(figures) == 4
        assert "distances" in figures

    def test_inspect_custom_figsize(self, fitted_pca, dummy_data_loader):
        """Test inspect with custom figure sizes."""
        # Arrange
        X, y = dummy_data_loader
        inspector = PCAInspector(model=fitted_pca, X_train=X, y_train=y)

        # Act
        figures = inspector.inspect(
            components_scores=(0, 1),
            loadings_components=[0, 1],
            scores_figsize=(8, 8),
            loadings_figsize=(12, 6),
            variance_figsize=(12, 6),
        )

        # Assert
        assert figures["scores_1"].get_size_inches()[0] == pytest.approx(8, rel=0.1)
        assert figures["loadings"].get_size_inches()[0] == pytest.approx(12, rel=0.1)

    def test_inspect_with_spectra_pipeline(
        self, fitted_pipeline_pca, dummy_data_loader
    ):
        """Test inspect includes spectra with preprocessing pipeline."""
        # Arrange
        X, y = dummy_data_loader
        inspector = PCAInspector(model=fitted_pipeline_pca, X_train=X, y_train=y)

        # Act
        figures = inspector.inspect(
            components_scores=(0, 1), loadings_components=[0, 1]
        )

        # Assert
        assert "raw_spectra" in figures
        assert "preprocessed_spectra" in figures

    def test_inspect_without_spectra_no_pipeline(self, fitted_pca, dummy_data_loader):
        """Test inspect without spectra when no pipeline."""
        # Arrange
        X, y = dummy_data_loader
        inspector = PCAInspector(model=fitted_pca, X_train=X, y_train=y)

        # Act
        figures = inspector.inspect(
            components_scores=(0, 1), loadings_components=[0, 1]
        )

        # Assert - no spectra plots should be created without preprocessing
        assert "raw_spectra" not in figures
        assert "preprocessed_spectra" not in figures

    def test_inspect_test_dataset(self, fitted_pca, dummy_data_loader):
        """Test inspect with test dataset."""
        # Arrange
        X, y = dummy_data_loader
        X_train, X_test = X[:80], X[80:]
        y_train, y_test = y[:80], y[80:]
        inspector = PCAInspector(
            model=fitted_pca,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
        )

        # Act
        figures = inspector.inspect(
            dataset="test",
            components_scores=(0, 1),
            loadings_components=[0, 1],
        )

        # Assert
        assert len(figures) == 4
        assert "distances" in figures

    def test_inspect_multi_dataset_returns_dataset_specific_scores(
        self, fitted_pca, dummy_data_loader
    ):
        """Multi-dataset inspect exposes per-dataset score figures."""
        # Arrange
        X, y = dummy_data_loader
        X_train, X_test = X[:80], X[80:]
        y_train, y_test = y[:80], y[80:]
        inspector = PCAInspector(
            model=fitted_pca,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
        )

        # Act
        figures = inspector.inspect(
            dataset=["train", "test"],
            components_scores=(0, 1),
            loadings_components=[0, 1],
        )

        # Assert
        # Multi-dataset mode only creates combined plots, not individual ones
        assert "scores_1" in figures
        assert (
            "scores_1_train" not in figures
        )  # Individual plots not created in multi-dataset mode
        assert (
            "scores_1_test" not in figures
        )  # Individual plots not created in multi-dataset mode

        legend = figures["scores_1"].axes[0].get_legend()
        assert legend is not None
        legend_labels = {text.get_text() for text in legend.get_texts()}
        assert legend_labels == {"Train", "Test"}

        for fig in figures.values():
            plt.close(fig)

    def test_inspect_closes_figures_properly(self, fitted_pca, dummy_data_loader):
        """Test that figures can be properly closed after creation."""
        # Arrange
        X, y = dummy_data_loader
        inspector = PCAInspector(model=fitted_pca, X_train=X, y_train=y)

        # Act
        figures = inspector.inspect(
            components_scores=(0, 1), loadings_components=[0, 1]
        )
        for fig in figures.values():
            plt.close(fig)

        # Assert - should not raise any errors
        assert True


class TestPCAInspectorFigureCleanup:
    """Test automatic figure cleanup in PCAInspector."""

    def test_inspect_tracks_figures(self, fitted_pca, dummy_data_loader):
        """Test that inspect() tracks created figures."""
        # Arrange
        X, y = dummy_data_loader
        inspector = PCAInspector(model=fitted_pca, X_train=X, y_train=y)

        # Act
        figures = inspector.inspect(
            components_scores=(0, 1), loadings_components=[0, 1]
        )

        # Assert
        assert len(inspector._tracked_figures) == len(figures)
        for fig in figures.values():
            assert fig in inspector._tracked_figures

    def test_inspect_cleans_up_previous_figures(self, fitted_pca, dummy_data_loader):
        """Test that calling inspect() twice cleans up previous figures."""
        # Arrange
        X, y = dummy_data_loader
        inspector = PCAInspector(model=fitted_pca, X_train=X, y_train=y)

        # Act - first call
        figures1 = inspector.inspect(
            components_scores=(0, 1), loadings_components=[0, 1]
        )
        first_call_figures = list(figures1.values())
        num_first_call = len(figures1)

        # Act - second call (should cleanup first figures)
        figures2 = inspector.inspect(
            components_scores=(0, 1), loadings_components=[0, 1]
        )

        # Assert - only second call figures are tracked (not accumulated)
        assert len(inspector._tracked_figures) == len(figures2)
        assert len(inspector._tracked_figures) == num_first_call  # Same number

        # Assert - tracked figures are the new ones, not the old ones
        for fig in figures2.values():
            assert fig in inspector._tracked_figures
        for fig in first_call_figures:
            assert fig not in inspector._tracked_figures

    def test_close_figures_clears_tracked(self, fitted_pca, dummy_data_loader):
        """Test that close_figures() properly clears tracked figures."""
        # Arrange
        X, y = dummy_data_loader
        inspector = PCAInspector(model=fitted_pca, X_train=X, y_train=y)
        figures = inspector.inspect(
            components_scores=(0, 1), loadings_components=[0, 1]
        )
        fig_nums = [fig.number for fig in figures.values()]

        # Act
        inspector.close_figures()

        # Assert
        assert inspector._tracked_figures == []
        for fig_num in fig_nums:
            assert fig_num not in plt.get_fignums()


class TestPCAInspectorInspectSpectra:
    """Test inspect_spectra method."""

    def test_inspect_spectra_with_pipeline(
        self, fitted_pipeline_pca, dummy_data_loader
    ):
        """Test inspect_spectra with preprocessing pipeline."""
        # Arrange
        X, y = dummy_data_loader
        inspector = PCAInspector(model=fitted_pipeline_pca, X_train=X, y_train=y)

        # Act
        figures = inspector.inspect_spectra()

        # Assert
        assert isinstance(figures, dict)
        assert "raw_spectra" in figures
        assert "preprocessed_spectra" in figures
        assert all(isinstance(fig, plt.Figure) for fig in figures.values())

    def test_inspect_spectra_without_pipeline_raises_error(
        self, fitted_pca, dummy_data_loader
    ):
        """Test inspect_spectra without pipeline raises ValueError."""
        # Arrange
        X, y = dummy_data_loader
        inspector = PCAInspector(model=fitted_pca, X_train=X, y_train=y)

        # Assert
        with pytest.raises(
            ValueError, match="Spectra inspection requires a preprocessing pipeline"
        ):
            inspector.inspect_spectra()

    def test_inspect_spectra_with_color_by_y(
        self, fitted_pipeline_pca, dummy_data_loader
    ):
        """Test inspect_spectra with color by y values."""
        # Arrange
        X, y = dummy_data_loader
        inspector = PCAInspector(model=fitted_pipeline_pca, X_train=X, y_train=y)

        # Act
        figures = inspector.inspect_spectra(color_by="y")

        # Assert
        assert len(figures) == 2

    def test_inspect_spectra_without_color_by_y(
        self, fitted_pipeline_pca, dummy_data_loader
    ):
        """Test inspect_spectra without color by y values."""
        # Arrange
        X, y = dummy_data_loader
        inspector = PCAInspector(model=fitted_pipeline_pca, X_train=X, y_train=y)

        # Act
        figures = inspector.inspect_spectra(color_by=None)

        # Assert
        assert len(figures) == 2

    def test_inspect_spectra_with_xlim(self, fitted_pipeline_pca, dummy_data_loader):
        """Test inspect_spectra with custom xlim."""
        # Arrange
        X, y = dummy_data_loader
        inspector = PCAInspector(model=fitted_pipeline_pca, X_train=X, y_train=y)

        # Act
        figures = inspector.inspect_spectra(xlim=(0, 2))

        # Assert
        assert len(figures) == 2

    def test_inspect_spectra_custom_figsize(
        self, fitted_pipeline_pca, dummy_data_loader
    ):
        """Test inspect_spectra with custom figure size."""
        # Arrange
        X, y = dummy_data_loader
        inspector = PCAInspector(model=fitted_pipeline_pca, X_train=X, y_train=y)

        # Act
        figures = inspector.inspect_spectra(figsize=(10, 4))

        # Assert
        assert figures["raw_spectra"].get_size_inches()[0] == pytest.approx(10, rel=0.1)

    def test_inspect_spectra_test_dataset(self, fitted_pipeline_pca, dummy_data_loader):
        """Test inspect_spectra with test dataset."""
        # Arrange
        X, y = dummy_data_loader
        X_train, X_test = X[:80], X[80:]
        y_train, y_test = y[:80], y[80:]
        inspector = PCAInspector(
            model=fitted_pipeline_pca,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
        )

        # Act
        figures = inspector.inspect_spectra(dataset="test")

        # Assert
        assert len(figures) == 2


class TestPCAInspectorDataCaching:
    """Test data caching behavior."""

    def test_scores_caching(self, fitted_pca, dummy_data_loader):
        """Test that scores are cached properly."""
        # Arrange
        X, _ = dummy_data_loader
        inspector = PCAInspector(model=fitted_pca, X_train=X)

        # Act
        scores1 = inspector.get_scores("train")
        scores2 = inspector.get_scores("train")

        # Assert - should return same cached object
        assert scores1 is scores2

    def test_preprocessed_data_caching(self, fitted_pipeline_pca, dummy_data_loader):
        """Test that preprocessed data is cached."""
        # Arrange
        X, _ = dummy_data_loader
        inspector = PCAInspector(model=fitted_pipeline_pca, X_train=X)

        # Act - access preprocessed data multiple times through scores
        scores1 = inspector.get_scores("train")
        scores2 = inspector.get_scores("train")

        # Assert - should be cached (same object)
        assert scores1 is scores2


class TestPCAInspectorEdgeCases:
    """Test edge cases and error handling."""

    def test_single_component_pca(self, dummy_data_loader):
        """Test with single component PCA."""
        # Arrange
        X, y = dummy_data_loader
        pca = PCA(n_components=1).fit(X)
        inspector = PCAInspector(model=pca, X_train=X, y_train=y)

        # Act & Assert - should not raise
        assert inspector.n_components == 1
        scores = inspector.get_scores("train")
        assert scores.shape == (100, 1)

    def test_high_dimensional_data(self):
        """Test with high-dimensional data."""
        # Arrange
        np.random.seed(42)
        X = np.random.randn(50, 100)  # 50 samples, 100 features
        pca = PCA(n_components=5).fit(X)
        inspector = PCAInspector(model=pca, X_train=X)

        # Act & Assert
        assert inspector.n_features == 100
        assert inspector.n_components == 5
        loadings = inspector.get_loadings()
        assert loadings.shape == (100, 5)

    def test_small_sample_size(self):
        """Test with small sample size."""
        # Arrange
        np.random.seed(42)
        X = np.random.randn(5, 3)  # Only 5 samples
        pca = PCA(n_components=2).fit(X)
        inspector = PCAInspector(model=pca, X_train=X)

        # Act & Assert
        assert inspector.n_samples["train"] == 5
        scores = inspector.get_scores("train")
        assert scores.shape == (5, 2)

    def test_inspector_with_list_input(self, fitted_pca):
        """Test inspector with list inputs instead of numpy arrays."""
        # Arrange
        X_list = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        y_list = [1, 2, 3]

        # Act
        inspector = PCAInspector(model=fitted_pca, X_train=X_list, y_train=y_list)

        # Assert - should convert to numpy arrays internally
        assert isinstance(inspector.datasets_["train"].X, np.ndarray)
        assert isinstance(inspector.datasets_["train"].y, np.ndarray)


class TestPCAInspectorIntegration:
    """Integration tests combining multiple features."""

    def test_full_workflow_with_pipeline(self, dummy_data_loader):
        """Test complete workflow with pipeline."""
        # Arrange
        X, y = dummy_data_loader
        X_train, X_test = X[:80], X[80:]
        y_train, y_test = y[:80], y[80:]

        pipeline = make_pipeline(StandardScaler(), PCA(n_components=2))
        pipeline.fit(X_train)

        # Act
        inspector = PCAInspector(
            model=pipeline,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            x_axis=np.arange(3),
        )

        # Summary
        inspector.summary()

        # Get data
        train_scores = inspector.get_scores("train")
        test_scores = inspector.get_scores("test")
        loadings = inspector.get_loadings([0, 1])
        var_ratio = inspector.get_explained_variance_ratio()

        # Create plots
        figures_train = inspector.inspect(
            dataset="train",
            components_scores=(0, 1),
            loadings_components=[0, 1],
        )
        figures_test = inspector.inspect(
            dataset="test",
            components_scores=(0, 1),
            loadings_components=[0, 1],
        )
        spectra_figs = inspector.inspect_spectra(dataset="train")

        # Assert
        assert train_scores.shape == (80, 2)
        assert test_scores.shape == (20, 2)
        assert loadings.shape == (3, 2)
        assert var_ratio.shape == (2,)
        assert len(figures_train) >= 3  # scores + loadings + variance (+ spectra)
        assert len(figures_test) >= 3
        assert len(spectra_figs) == 2

        # Cleanup
        for fig in {**figures_train, **figures_test, **spectra_figs}.values():
            plt.close(fig)

    def test_multiple_inspect_calls(self, fitted_pca, dummy_data_loader):
        """Test multiple calls to inspect method."""
        # Arrange
        X, y = dummy_data_loader
        inspector = PCAInspector(model=fitted_pca, X_train=X, y_train=y)

        # Act - call inspect multiple times
        figures1 = inspector.inspect(
            components_scores=(0, 1), loadings_components=[0, 1]
        )
        figures2 = inspector.inspect(
            components_scores=(0, 1), loadings_components=[0, 1]
        )

        # Assert - should create new figures each time
        assert figures1["scores_1"] is not figures2["scores_1"]

        # Cleanup
        for fig in {**figures1, **figures2}.values():
            plt.close(fig)


class TestPCAInspectorAdditionalCoverage:
    """Additional tests to improve code coverage."""

    def test_init_with_invalid_confidence_raises_error(
        self, fitted_pca, dummy_data_loader
    ):
        """Test that invalid confidence values raise ValueError."""
        # Arrange
        X, y = dummy_data_loader

        # Act & Assert - confidence = 0
        with pytest.raises(ValueError, match="confidence must be between 0 and 1"):
            PCAInspector(model=fitted_pca, X_train=X, y_train=y, confidence=0)

        # Act & Assert - confidence = 1
        with pytest.raises(ValueError, match="confidence must be between 0 and 1"):
            PCAInspector(model=fitted_pca, X_train=X, y_train=y, confidence=1)

        # Act & Assert - confidence > 1
        with pytest.raises(ValueError, match="confidence must be between 0 and 1"):
            PCAInspector(model=fitted_pca, X_train=X, y_train=y, confidence=1.5)

        # Act & Assert - confidence < 0
        with pytest.raises(ValueError, match="confidence must be between 0 and 1"):
            PCAInspector(model=fitted_pca, X_train=X, y_train=y, confidence=-0.1)

    def test_hotelling_t2_limit_caching(self, fitted_pca, dummy_data_loader):
        """Test that Hotelling T² limit is cached after first access."""
        # Arrange
        X, y = dummy_data_loader
        inspector = PCAInspector(
            model=fitted_pca, X_train=X, y_train=y, confidence=0.95
        )

        # Act - access limit twice
        limit1 = inspector.hotelling_t2_limit
        limit2 = inspector.hotelling_t2_limit

        # Assert - should be same value (cached)
        assert limit1 == limit2
        assert isinstance(limit1, (float, np.floating))
        assert limit1 > 0

    def test_q_residuals_limit_caching(self, fitted_pca, dummy_data_loader):
        """Test that Q residuals limit is cached after first access."""
        # Arrange
        X, y = dummy_data_loader
        inspector = PCAInspector(
            model=fitted_pca, X_train=X, y_train=y, confidence=0.95
        )

        # Act - access limit twice
        limit1 = inspector.q_residuals_limit
        limit2 = inspector.q_residuals_limit

        # Assert - should be same value (cached)
        assert limit1 == limit2
        assert isinstance(limit1, (float, np.floating))
        assert limit1 > 0

    def test_summary_pc_variances_with_different_components(self, dummy_data_loader):
        """Test summary pc_variances dict adjusts to number of components."""
        # Arrange
        X, y = dummy_data_loader

        # Act & Assert - 1 component
        pca_one = PCA(n_components=1).fit(X)
        inspector_one = PCAInspector(model=pca_one, X_train=X, y_train=y)
        summary_one = inspector_one.summary()
        assert "PC1" in summary_one.pc_variances
        assert "PC2" not in summary_one.pc_variances
        assert "PC3" not in summary_one.pc_variances

        # Act & Assert - 3+ components
        pca_three = PCA(n_components=3).fit(X)
        inspector_three = PCAInspector(model=pca_three, X_train=X, y_train=y)
        summary_three = inspector_three.summary()
        assert "PC1" in summary_three.pc_variances
        assert "PC2" in summary_three.pc_variances
        assert "PC3" in summary_three.pc_variances

    def test_summary_preprocessing_steps_empty_for_plain_model(
        self, fitted_pca, dummy_data_loader
    ):
        """Test that preprocessing_steps is empty list for plain PCA model."""
        # Arrange
        X, y = dummy_data_loader
        inspector = PCAInspector(model=fitted_pca, X_train=X, y_train=y)

        # Act
        summary = inspector.summary()

        # Assert
        assert summary.preprocessing_steps == []
        assert summary.has_preprocessing is False

    def test_inspect_spectra_with_multiple_datasets(
        self, fitted_pipeline_pca, dummy_data_loader
    ):
        """Test inspect_spectra with multiple datasets (multi-dataset path)."""
        # Arrange
        X, y = dummy_data_loader
        X_train, X_test = X[:80], X[80:]
        y_train, y_test = y[:80], y[80:]
        inspector = PCAInspector(
            model=fitted_pipeline_pca,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
        )

        # Act
        figures = inspector.inspect_spectra(dataset=["train", "test"])

        # Assert - should have both raw and preprocessed figures
        assert isinstance(figures, dict)
        assert "raw_spectra" in figures
        assert "preprocessed_spectra" in figures

        # Cleanup
        for fig in figures.values():
            plt.close(fig)
