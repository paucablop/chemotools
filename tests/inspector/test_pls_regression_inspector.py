import numpy as np
import pytest
import matplotlib.pyplot as plt
from sklearn.base import is_regressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from chemotools.inspector import PLSRegressionInspector
from chemotools.inspector.core.summaries import InspectorSummary


@pytest.fixture
def regression_data():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(90, 6))
    y = X @ np.array([1.2, -0.7, 0.5, 0.0, 0.3, -1.1]) + rng.normal(scale=0.1, size=90)
    y = y.reshape(-1, 1)
    X_train, X_test, X_val = X[:50], X[50:70], X[70:]
    y_train, y_test, y_val = y[:50], y[50:70], y[70:]
    return {
        "train": (X_train, y_train),
        "test": (X_test, y_test),
        "val": (X_val, y_val),
    }


@pytest.fixture
def fitted_pls(regression_data):
    X_train, y_train = regression_data["train"]
    model = PLSRegression(n_components=3)
    model.fit(X_train, y_train)
    return model


@pytest.fixture
def fitted_pipeline(regression_data):
    X_train, y_train = regression_data["train"]
    model = make_pipeline(StandardScaler(), PLSRegression(n_components=2))
    model.fit(X_train, y_train)
    return model


@pytest.fixture
def multi_target_regression_data():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(90, 6))
    coef = np.array(
        [
            [1.0, -0.4],
            [-0.6, 0.8],
            [0.2, 0.5],
            [0.0, -0.3],
            [0.7, 0.1],
            [-1.1, 0.9],
        ]
    )
    y = X @ coef + rng.normal(scale=0.1, size=(90, 2))
    X_train, X_test, X_val = X[:50], X[50:70], X[70:]
    y_train, y_test, y_val = y[:50], y[50:70], y[70:]
    return {
        "train": (X_train, y_train),
        "test": (X_test, y_test),
        "val": (X_val, y_val),
    }


@pytest.fixture
def fitted_pls_multi(multi_target_regression_data):
    X_train, y_train = multi_target_regression_data["train"]
    model = PLSRegression(n_components=2)
    model.fit(X_train, y_train)
    return model


class TestInitialization:
    def test_init_with_all_datasets(self, fitted_pls, regression_data):
        """Test PLSRegressionInspector initialization with train, test, and validation datasets."""
        # Arrange
        X_train, y_train = regression_data["train"]
        X_test, y_test = regression_data["test"]
        X_val, y_val = regression_data["val"]

        # Act
        inspector = PLSRegressionInspector(
            fitted_pls,
            X_train,
            y_train,
            X_test=X_test,
            y_test=y_test,
            X_val=X_val,
            y_val=y_val,
        )

        # Assert
        assert inspector.n_samples == {"train": 50, "test": 20, "val": 20}
        assert inspector.n_components == 3
        assert inspector.transformer is None

    def test_init_with_pipeline(self, fitted_pipeline, regression_data):
        """Test PLSRegressionInspector initialization with a scikit-learn Pipeline."""
        # Arrange
        X_train, y_train = regression_data["train"]

        # Act
        inspector = PLSRegressionInspector(fitted_pipeline, X_train, y_train)

        # Assert
        assert inspector.transformer is not None
        assert is_regressor(inspector.estimator)

    def test_missing_targets_raises(self, fitted_pls, regression_data):
        """Test that missing target variables raise appropriate errors."""
        # Arrange
        X_train, _ = regression_data["train"]
        X_test, y_test = regression_data["test"]

        # Act & Assert - missing y_train
        with pytest.raises(ValueError, match="y_train required"):
            PLSRegressionInspector(fitted_pls, X_train, y_train=None)

        # Act & Assert - missing y_test
        with pytest.raises(ValueError, match="y_test required"):
            PLSRegressionInspector(
                fitted_pls,
                X_train,
                y_train=np.ones((50, 1)),
                X_test=X_test,
                y_test=None,
            )


class TestSummary:
    def test_summary_contains_metrics(self, fitted_pls, regression_data):
        """Test that summary contains expected regression metrics and model information."""
        # Arrange
        X_train, y_train = regression_data["train"]
        inspector = PLSRegressionInspector(fitted_pls, X_train, y_train)

        # Act
        summary = inspector.summary()

        # Assert
        assert isinstance(summary, InspectorSummary)
        assert summary.train is not None
        assert summary.train.rmse is not None
        assert summary.train.r2 is not None
        assert summary.model_type is not None
        assert summary.model_type.startswith("PLS")

    def test_summary_with_pipeline(self, fitted_pipeline, regression_data):
        """Test that summary correctly identifies preprocessing steps in a pipeline."""
        # Arrange
        X_train, y_train = regression_data["train"]
        inspector = PLSRegressionInspector(fitted_pipeline, X_train, y_train)

        # Act
        summary = inspector.summary()

        # Assert
        assert summary.has_preprocessing is True
        assert len(summary.preprocessing_steps) == 1

    def test_summary_variance_coverage(self, fitted_pls, regression_data):
        """Test summary method when variance ratios are available."""
        X_train, y_train = regression_data["train"]

        # Manually add variance ratios to simulate a model that has them
        fitted_pls.explained_x_variance_ratio_ = np.array([0.4, 0.3, 0.1])
        fitted_pls.explained_y_variance_ratio_ = np.array([0.5, 0.4, 0.05])

        inspector = PLSRegressionInspector(fitted_pls, X_train, y_train)

        summary = inspector.summary()

        assert summary.explained_x_variance_ratio is not None
        assert summary.total_x_variance is not None
        assert summary.explained_y_variance_ratio is not None
        assert summary.total_y_variance is not None
        assert summary.total_x_variance == pytest.approx(80.0)
        assert summary.total_y_variance == pytest.approx(95.0)

    def test_summary_includes_latent_info(self, fitted_pls, regression_data):
        """Test that summary includes latent variable information."""
        # Arrange
        X_train, y_train = regression_data["train"]
        inspector = PLSRegressionInspector(fitted_pls, X_train, y_train)

        # Act
        summary = inspector.summary()

        # Assert
        assert summary.n_components == 3
        assert isinstance(summary.hotelling_t2_limit, float)
        assert isinstance(summary.q_residuals_limit, float)


class TestInspectFigures:
    def test_inspect_single_dataset(self, fitted_pls, regression_data):
        """Test that inspect() generates all expected diagnostic figures for a single dataset."""
        # Arrange
        X_train, y_train = regression_data["train"]
        inspector = PLSRegressionInspector(fitted_pls, X_train, y_train)

        # Act
        figures = inspector.inspect(
            dataset="train", components_scores=(0, 1), loadings_components=[0, 1]
        )

        # Assert
        expected_keys = {
            "scores_1",
            "x_vs_y_scores_1",
            "loadings_x",
            "loadings_weights",
            "loadings_rotations",
            "regression_coefficients",
            "distances_hotelling_q",
            "distances_q_y_residuals",
            "distances_leverage_studentized",
            "predicted_vs_actual",
            "residuals",
            "qq_plot",
            "residual_distribution",
        }
        assert expected_keys.issubset(figures.keys())

        for fig in figures.values():
            fig.canvas.draw_idle()

        # Cleanup
        for fig in figures.values():
            plt.close(fig)

    def test_inspect_multi_dataset(self, fitted_pls, regression_data):
        """Test that inspect() handles multiple datasets (train and test) correctly."""
        # Arrange
        X_train, y_train = regression_data["train"]
        X_test, y_test = regression_data["test"]
        inspector = PLSRegressionInspector(
            fitted_pls,
            X_train,
            y_train,
            X_test=X_test,
            y_test=y_test,
        )

        # Act
        figures = inspector.inspect(
            dataset=["train", "test"], components_scores=((0, 1),)
        )

        # Assert
        assert "scores_1" in figures
        assert "distances_leverage_studentized" in figures
        assert "predicted_vs_actual" in figures
        assert "residuals" in figures

        # Cleanup
        for fig in figures.values():
            plt.close(fig)

    def test_inspect_includes_q_vs_y_residuals_plot(self, fitted_pls, regression_data):
        """Test that inspect() includes the Q vs Y residuals plot in output."""
        # Arrange
        X_train, y_train = regression_data["train"]
        inspector = PLSRegressionInspector(fitted_pls, X_train, y_train)

        # Act
        figures = inspector.inspect(dataset="train")

        # Assert
        assert "distances_q_y_residuals" in figures
        fig = figures["distances_q_y_residuals"]
        assert fig is not None
        ax = fig.axes[0]
        assert ax.get_xlabel() == "Y Residuals (Prediction Error)"
        assert ax.get_ylabel() == "Q Residuals (SPE)"

        # Cleanup
        for fig in figures.values():
            plt.close(fig)

    def test_q_vs_y_residuals_single_dataset(self, fitted_pls, regression_data):
        """Test Q vs Y residuals plot for single dataset in inspector."""
        # Arrange
        X_train, y_train = regression_data["train"]
        inspector = PLSRegressionInspector(fitted_pls, X_train, y_train)

        # Act
        figures = inspector.inspect(dataset="train")

        # Assert
        fig = figures["distances_q_y_residuals"]
        ax = fig.axes[0]
        assert "Train" in ax.get_title()

        # Cleanup
        for fig in figures.values():
            plt.close(fig)

    def test_q_vs_y_residuals_multi_dataset(self, fitted_pls, regression_data):
        """Test Q vs Y residuals plot for multiple datasets in inspector."""
        # Arrange
        X_train, y_train = regression_data["train"]
        X_test, y_test = regression_data["test"]
        inspector = PLSRegressionInspector(
            fitted_pls,
            X_train,
            y_train,
            X_test=X_test,
            y_test=y_test,
        )

        # Act
        figures = inspector.inspect(dataset=["train", "test"])

        # Assert
        fig = figures["distances_q_y_residuals"]
        ax = fig.axes[0]
        assert ax.get_legend() is not None
        legend_labels = [t.get_text() for t in ax.get_legend().get_texts()]
        assert "Train" in legend_labels
        assert "Test" in legend_labels

        # Cleanup
        for fig in figures.values():
            plt.close(fig)

    def test_inspect_spectra_requires_pipeline(self, fitted_pls, regression_data):
        """Test that inspect_spectra() raises an error when model is not a pipeline."""
        # Arrange
        X_train, y_train = regression_data["train"]
        inspector = PLSRegressionInspector(fitted_pls, X_train, y_train)

        # Act & Assert
        with pytest.raises(ValueError, match="requires a preprocessing"):
            inspector.inspect_spectra()

    def test_inspect_spectra_pipeline_single_dataset(
        self, fitted_pipeline, regression_data
    ):
        """Test inspect_spectra() with a pipeline model for a single dataset."""
        # Arrange
        X_train, y_train = regression_data["train"]
        inspector = PLSRegressionInspector(fitted_pipeline, X_train, y_train)

        # Act
        figures = inspector.inspect_spectra(dataset="train")

        # Assert
        assert set(figures.keys()) == {"raw_spectra", "preprocessed_spectra"}

        # Cleanup
        for fig in figures.values():
            plt.close(fig)

    def test_inspect_spectra_pipeline_multi_dataset(
        self, fitted_pipeline, regression_data
    ):
        """Test inspect_spectra() with a pipeline model for multiple datasets."""
        # Arrange
        X_train, y_train = regression_data["train"]
        X_test, y_test = regression_data["test"]
        inspector = PLSRegressionInspector(
            fitted_pipeline,
            X_train,
            y_train,
            X_test=X_test,
            y_test=y_test,
        )

        # Act
        figures = inspector.inspect_spectra(dataset=["train", "test"])

        # Assert
        assert set(figures.keys()) == {"raw_spectra", "preprocessed_spectra"}

        # Cleanup
        for fig in figures.values():
            plt.close(fig)


class TestRegressionDiagnostics:
    def test_regression_metrics_cached(self, fitted_pls, regression_data):
        """Test that regression metrics (RMSE, R2) are computed once and cached."""
        # Arrange
        X_train, y_train = regression_data["train"]
        inspector = PLSRegressionInspector(fitted_pls, X_train, y_train)

        # Act
        rmse_first = inspector.RMSE_train
        rmse_second = inspector.RMSE_train
        r2_first = inspector.R2_train
        r2_second = inspector.R2_train

        # Assert
        assert rmse_first == pytest.approx(rmse_second)
        assert r2_first == pytest.approx(r2_second)

    def test_regression_distances_keys(self, fitted_pls, regression_data):
        """Test that regression diagnostics plot has correct axis labels."""
        # Arrange
        X_train, y_train = regression_data["train"]
        inspector = PLSRegressionInspector(fitted_pls, X_train, y_train)

        # Act
        figures = inspector.inspect(
            dataset="train",
            components_scores=(0, 1),
            loadings_components=[0, 1],
            color_by_y=False,
        )

        # Assert
        assert "distances_leverage_studentized" in figures
        fig = figures["distances_leverage_studentized"]
        ax = fig.axes[0]
        assert ax.get_xlabel() == "Leverage"
        assert ax.get_ylabel() == "Studentized Residuals"

        # Cleanup
        for fig in figures.values():
            plt.close(fig)


class TestPLSSpecificDiagnostics:
    """Test PLS-specific diagnostics: leverage and studentized residuals."""

    def test_leverage_detector_cached(self, fitted_pls, regression_data):
        """Test that leverage detector is created once and cached."""
        # Arrange
        X_train, y_train = regression_data["train"]
        inspector = PLSRegressionInspector(fitted_pls, X_train, y_train)

        # Act
        detector1 = inspector.leverage_detector
        detector2 = inspector.leverage_detector

        # Assert
        assert detector1 is detector2

    def test_studentized_detector_cached(self, fitted_pls, regression_data):
        """Test that studentized residuals detector is created once and cached."""
        # Arrange
        X_train, y_train = regression_data["train"]
        inspector = PLSRegressionInspector(fitted_pls, X_train, y_train)

        # Act
        detector1 = inspector.studentized_detector
        detector2 = inspector.studentized_detector

        # Assert
        assert detector1 is detector2

    def test_leverage_detector_fitted_on_training_data(
        self, fitted_pls, regression_data
    ):
        """Test that leverage detector is fitted on training data."""
        # Arrange
        X_train, y_train = regression_data["train"]
        inspector = PLSRegressionInspector(fitted_pls, X_train, y_train)

        # Act
        detector = inspector.leverage_detector
        leverages = detector.predict_residuals(X_train)

        # Assert
        assert len(leverages) == len(X_train)
        assert all(lev >= 0 for lev in leverages)  # Leverage is non-negative

    def test_studentized_detector_fitted_on_training_data(
        self, fitted_pls, regression_data
    ):
        """Test that studentized residuals detector is fitted on training data."""
        # Arrange
        X_train, y_train = regression_data["train"]
        inspector = PLSRegressionInspector(fitted_pls, X_train, y_train)

        # Act
        detector = inspector.studentized_detector
        studentized = detector.predict_residuals(X_train, y_train)

        # Assert
        assert len(studentized) == len(X_train)

    def test_get_regression_stats_returns_expected_keys(
        self, fitted_pls, regression_data
    ):
        """Test that _get_regression_stats returns all expected keys."""
        # Arrange
        X_train, y_train = regression_data["train"]
        inspector = PLSRegressionInspector(fitted_pls, X_train, y_train)
        leverage_detector = inspector.leverage_detector

        # Act
        stats = inspector._get_regression_stats("train", 0, leverage_detector)

        # Assert
        expected_keys = {"X", "y_true", "y_pred", "studentized", "leverages"}
        assert set(stats.keys()) == expected_keys

    def test_get_regression_stats_values_have_correct_shapes(
        self, fitted_pls, regression_data
    ):
        """Test that _get_regression_stats returns arrays with correct shapes."""
        # Arrange
        X_train, y_train = regression_data["train"]
        inspector = PLSRegressionInspector(fitted_pls, X_train, y_train)
        leverage_detector = inspector.leverage_detector
        n_samples = len(X_train)

        # Act
        stats = inspector._get_regression_stats("train", 0, leverage_detector)

        # Assert
        assert stats["X"].shape[0] == n_samples
        assert len(stats["y_true"]) == n_samples
        assert len(stats["y_pred"]) == n_samples
        assert len(stats["studentized"]) == n_samples
        assert len(stats["leverages"]) == n_samples

    def test_get_regression_stats_for_test_dataset(self, fitted_pls, regression_data):
        """Test that _get_regression_stats works for test dataset."""
        # Arrange
        X_train, y_train = regression_data["train"]
        X_test, y_test = regression_data["test"]
        inspector = PLSRegressionInspector(
            fitted_pls, X_train, y_train, X_test=X_test, y_test=y_test
        )
        leverage_detector = inspector.leverage_detector

        # Act
        stats = inspector._get_regression_stats("test", 0, leverage_detector)

        # Assert
        assert stats["X"].shape[0] == len(X_test)
        assert len(stats["y_true"]) == len(y_test)


class TestPLSRegressionInspectorFigureCleanup:
    """Test automatic figure cleanup in PLSRegressionInspector."""

    def test_inspect_tracks_figures(self, fitted_pls, regression_data):
        """Test that inspect() tracks created figures."""
        # Arrange
        X_train, y_train = regression_data["train"]
        inspector = PLSRegressionInspector(fitted_pls, X_train, y_train)

        # Act
        figures = inspector.inspect(
            dataset="train", components_scores=(0, 1), loadings_components=[0, 1]
        )

        # Assert
        assert len(inspector._tracked_figures) == len(figures)
        for fig in figures.values():
            assert fig in inspector._tracked_figures

    def test_inspect_cleans_up_previous_figures(self, fitted_pls, regression_data):
        """Test that calling inspect() twice cleans up previous figures."""
        # Arrange
        X_train, y_train = regression_data["train"]
        inspector = PLSRegressionInspector(fitted_pls, X_train, y_train)

        # Act - first call
        figures1 = inspector.inspect(
            dataset="train", components_scores=(0, 1), loadings_components=[0, 1]
        )
        first_call_figures = list(figures1.values())
        num_first_call = len(figures1)

        # Act - second call (should cleanup first figures)
        figures2 = inspector.inspect(
            dataset="train", components_scores=(0, 1), loadings_components=[0, 1]
        )

        # Assert - only second call figures are tracked (not accumulated)
        assert len(inspector._tracked_figures) == len(figures2)
        assert len(inspector._tracked_figures) == num_first_call  # Same number

        # Assert - tracked figures are the new ones, not the old ones
        for fig in figures2.values():
            assert fig in inspector._tracked_figures
        for fig in first_call_figures:
            assert fig not in inspector._tracked_figures

    def test_close_figures_clears_tracked(self, fitted_pls, regression_data):
        """Test that close_figures() properly clears tracked figures."""
        # Arrange
        X_train, y_train = regression_data["train"]
        inspector = PLSRegressionInspector(fitted_pls, X_train, y_train)
        figures = inspector.inspect(
            dataset="train", components_scores=(0, 1), loadings_components=[0, 1]
        )
        fig_nums = [fig.number for fig in figures.values()]

        # Act
        inspector.close_figures()

        # Assert
        assert inspector._tracked_figures == []
        for fig_num in fig_nums:
            assert fig_num not in plt.get_fignums()


class TestAdditionalCoverage:
    def test_detector_limits_cached(self, fitted_pls, regression_data, monkeypatch):
        """Test that Hotelling T2 and Q-residuals limits are computed once and cached."""
        # Arrange
        X_train, y_train = regression_data["train"]
        hot_calls = []
        q_calls = []

        class _DummyHotelling:
            def __init__(self, model, confidence):
                self.model = model
                self.confidence = confidence
                self.critical_value_ = 0.0

            def fit(self, X):
                hot_calls.append(X.copy())
                self.critical_value_ = 1.23
                return self

        class _DummyQ:
            def __init__(self, model, confidence):
                self.model = model
                self.confidence = confidence
                self.critical_value_ = 0.0

            def fit(self, X):
                q_calls.append(X.copy())
                self.critical_value_ = 4.56
                return self

        monkeypatch.setattr(
            "chemotools.inspector.core.latent.HotellingT2",
            _DummyHotelling,
        )
        monkeypatch.setattr(
            "chemotools.inspector.core.latent.QResiduals",
            _DummyQ,
        )

        # Act
        inspector = PLSRegressionInspector(fitted_pls, X_train, y_train)
        hotelling_1 = inspector.hotelling_t2_limit
        hotelling_2 = inspector.hotelling_t2_limit
        q_1 = inspector.q_residuals_limit
        q_2 = inspector.q_residuals_limit

        # Assert
        assert hotelling_1 == pytest.approx(1.23)
        assert hotelling_2 == pytest.approx(1.23)
        assert len(hot_calls) == 1
        assert q_1 == pytest.approx(4.56)
        assert q_2 == pytest.approx(4.56)
        assert len(q_calls) == 1

    def test_component_selection_helpers(self, fitted_pls, regression_data):
        """Test component selection methods for loadings, weights, and rotations."""
        # Arrange
        X_train, y_train = regression_data["train"]
        inspector = PLSRegressionInspector(fitted_pls, X_train, y_train)

        # Act
        all_loadings = inspector.get_x_loadings()
        single_loading = inspector.get_x_loadings(0)
        multi_loading = inspector.get_x_loadings([0, 1])
        single_weight = inspector.get_x_weights(1)
        list_weights = inspector.get_x_weights([0, 2])
        single_rotation = inspector.get_x_rotations(1)
        list_rotations = inspector.get_x_rotations([0, 2])

        # Assert
        assert single_loading.shape[1] == 1
        assert multi_loading.shape[1] == 2
        assert np.allclose(single_loading.squeeze(), all_loadings[:, 0])
        assert single_weight.shape[1] == 1
        assert list_weights.shape[1] == 2
        assert single_rotation.shape[1] == 1
        assert list_rotations.shape[1] == 2

    def test_regression_coefficients_multitarget_and_legend(
        self, fitted_pls_multi, multi_target_regression_data, monkeypatch
    ):
        """Test regression coefficients plot with multiple targets includes legend."""
        # Arrange
        X_train, y_train = multi_target_regression_data["train"]
        inspector = PLSRegressionInspector(fitted_pls_multi, X_train, y_train)

        def _dummy_figure(*args, **kwargs):
            fig, _ = plt.subplots()
            return fig

        monkeypatch.setattr(
            "chemotools.inspector._pls_regression_inspector.create_regression_distances_plot",
            _dummy_figure,
        )
        monkeypatch.setattr(
            "chemotools.inspector._pls_regression_inspector.create_predicted_vs_actual_plot",
            _dummy_figure,
        )
        monkeypatch.setattr(
            "chemotools.inspector._pls_regression_inspector.create_y_residual_plot",
            _dummy_figure,
        )
        monkeypatch.setattr(
            "chemotools.inspector._pls_regression_inspector.create_qq_plot",
            _dummy_figure,
        )
        monkeypatch.setattr(
            "chemotools.inspector._pls_regression_inspector.create_residual_distribution_plot",
            _dummy_figure,
        )

        # Act
        coef = inspector.get_regression_coefficients()
        figures = inspector.inspect(
            dataset="train",
            components_scores=(0, 1),
            loadings_components=[0, 1],
            color_by=None,
        )

        # Assert
        assert coef.shape[1] == 2
        coef_fig = figures["regression_coefficients"]
        legend = coef_fig.axes[0].legend_
        assert legend is not None
        legend_labels = [text.get_text() for text in legend.get_texts()]
        assert legend_labels == ["Target 1", "Target 2"]

        # Cleanup
        for fig in figures.values():
            plt.close(fig)

    def test_create_latent_scores_missing_train_fallback(
        self, fitted_pls, regression_data, monkeypatch
    ):
        """Test latent scores creation with fallback when train dataset is missing."""
        # Arrange
        X_train, y_train = regression_data["train"]
        X_test, y_test = regression_data["test"]
        inspector = PLSRegressionInspector(
            fitted_pls,
            X_train,
            y_train,
            X_test=X_test,
            y_test=y_test,
        )
        original_get = inspector.get_x_scores
        first_call = {"trigger": True}

        def _patched_get(dataset="train"):
            if dataset == "train" and first_call["trigger"]:
                first_call["trigger"] = False
                raise KeyError("train missing")
            return original_get(dataset)

        monkeypatch.setattr(inspector, "get_x_scores", _patched_get)

        # Act
        figures = inspector.create_latent_scores_figures(
            dataset="test",
            components=(0, 1),
            color_by=None,
            annotate_by=None,
            figsize=(3, 3),
        )

        # Assert
        assert "scores_1" in figures

        # Cleanup
        for fig in figures.values():
            plt.close(fig)

    def test_create_latent_scores_multi_dataset_combined(
        self, fitted_pls, regression_data
    ):
        """Test latent scores creation for multiple datasets combined."""
        # Arrange
        X_train, y_train = regression_data["train"]
        X_test, y_test = regression_data["test"]
        inspector = PLSRegressionInspector(
            fitted_pls,
            X_train,
            y_train,
            X_test=X_test,
            y_test=y_test,
        )

        # Act
        figures = inspector.create_latent_scores_figures(
            dataset=["train", "test"],
            components=(0, 1),
            color_by=None,
            annotate_by=None,
            figsize=(3, 3),
        )

        # Assert
        assert set(figures.keys()) == {"scores_1"}

        # Cleanup
        for fig in figures.values():
            plt.close(fig)

    def test_create_x_vs_y_scores_mixed_components(self, fitted_pls, regression_data):
        """Test X vs Y scores creation with mixed component specifications."""
        # Arrange
        X_train, y_train = regression_data["train"]
        inspector = PLSRegressionInspector(fitted_pls, X_train, y_train)
        annotate_by = {"train": np.arange(X_train.shape[0])}

        # Act
        figures = inspector.inspect(
            components_scores=[(0, 1), 2],
            color_by="y",
            annotate_by=annotate_by,
            scores_figsize=(3, 3),
        )

        # Assert
        assert "x_vs_y_scores_1" in figures
        assert "x_vs_y_scores_2" not in figures

        # Cleanup
        for fig in figures.values():
            plt.close(fig)

        # Cleanup
        for fig in figures.values():
            plt.close(fig)


class TestValidationPropagation:
    def test_y_length_mismatch_raises(self, fitted_pls, regression_data):
        """Test that mismatched X and y sample counts raise appropriate errors."""
        # Arrange
        X_train, y_train = regression_data["train"]
        X_test, y_test = regression_data["test"]

        # Act & Assert - train dataset mismatch
        with pytest.raises(ValueError, match="same number of samples"):
            PLSRegressionInspector(
                fitted_pls,
                X_train,
                y_train[:-1],
            )

        # Act & Assert - test dataset mismatch
        with pytest.raises(ValueError, match="same number of samples"):
            PLSRegressionInspector(
                fitted_pls,
                X_train,
                y_train,
                X_test=X_test,
                y_test=y_test[:-1],
            )

    def test_inspect_color_by_default(self, fitted_pls, regression_data):
        """Test inspect method default color_by logic."""
        # Arrange
        X_train, y_train = regression_data["train"]
        inspector = PLSRegressionInspector(fitted_pls, X_train, y_train)

        # Act
        figures = inspector.inspect(dataset="train")

        # Assert
        assert len(figures) > 0

        # Cleanup
        for fig in figures.values():
            plt.close(fig)

    def test_inspect_y_variance_plot(self, fitted_pls, regression_data):
        """Test inspect method creates Y-variance plot when available."""
        # Arrange
        X_train, y_train = regression_data["train"]
        # Manually add variance ratios
        fitted_pls.explained_y_variance_ratio_ = np.array([0.5, 0.4, 0.05])
        inspector = PLSRegressionInspector(fitted_pls, X_train, y_train)

        # Act
        figures = inspector.inspect(dataset="train")

        # Assert
        assert "variance_y" in figures

        # Cleanup
        for fig in figures.values():
            plt.close(fig)


def test_inspect_single_dataset_raw_array_arguments(fitted_pls, regression_data):
    """Test that raw arrays can be passed for color_by/annotate_by when inspecting a single dataset."""
    # Arrange
    X_train, y_train = regression_data["train"]
    inspector = PLSRegressionInspector(fitted_pls, X_train, y_train)

    # Create dummy color/annotation arrays matching sample count
    n_samples = X_train.shape[0]
    color_array = np.random.rand(n_samples)
    annotate_array = np.arange(n_samples)

    # Act
    figs = inspector.inspect(
        dataset="train", color_by=color_array, annotate_by=annotate_array
    )

    # Assert
    assert len(figs) > 0
    for fig in figs.values():
        plt.close(fig)

    # Act & Assert - Error cases
    with pytest.raises(ValueError, match="When inspecting multiple datasets"):
        inspector.inspect(dataset=["train", "train"], color_by=color_array)

    with pytest.raises(ValueError, match="When inspecting multiple datasets"):
        inspector.inspect(dataset=["train", "train"], annotate_by=annotate_array)
