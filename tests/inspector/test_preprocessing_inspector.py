"""Comprehensive test suite for PreprocessingInspector."""

import warnings

import matplotlib
import numpy as np
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import NMF, PCA, FastICA, KernelPCA
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVR

from chemotools.inspector import PreprocessingInspector
from chemotools.inspector.core.summaries import PreprocessingSummary

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def spectral_data():
    """Generate realistic spectral data for testing."""
    rng = np.random.RandomState(42)
    n_samples, n_features = 50, 200
    x_axis = np.linspace(400, 4000, n_features)
    X = rng.randn(n_samples, n_features) + np.sin(x_axis / 500)
    y = np.sum(X[:, :10], axis=1) + rng.randn(n_samples) * 0.1
    return X, y, x_axis


@pytest.fixture
def split_data(spectral_data):
    """Split spectral data into train / test / val."""
    X, y, x_axis = spectral_data
    return {
        "X_train": X[:30],
        "y_train": y[:30],
        "X_test": X[30:40],
        "y_test": y[30:40],
        "X_val": X[40:],
        "y_val": y[40:],
        "x_axis": x_axis,
    }


@pytest.fixture
def fitted_preprocessing_pipeline(spectral_data):
    """Pipeline with only preprocessing steps (no estimator)."""
    X, _, _ = spectral_data
    pipe = make_pipeline(StandardScaler(), MinMaxScaler())
    pipe.fit(X)
    return pipe


@pytest.fixture
def fitted_pipeline_with_pca(spectral_data):
    """Pipeline with preprocessing + PCA at the end."""
    X, _, _ = spectral_data
    pipe = make_pipeline(StandardScaler(), MinMaxScaler(), PCA(n_components=5))
    pipe.fit(X)
    return pipe


@pytest.fixture
def fitted_pipeline_with_svr(spectral_data):
    """Pipeline with preprocessing + regressor at the end."""
    X, y, _ = spectral_data
    pipe = make_pipeline(StandardScaler(), SVR())
    pipe.fit(X, y)
    return pipe


# ===========================================================================
# Initialization tests
# ===========================================================================


class TestPreprocessingInspectorInit:
    """Test initialization and validation."""

    def test_basic_init(self, fitted_preprocessing_pipeline, spectral_data):
        """Test basic initialisation with training data only."""
        # Arrange
        X, y, _ = spectral_data

        # Act
        inspector = PreprocessingInspector(fitted_preprocessing_pipeline, X, y)

        # Assert
        assert inspector.n_features == X.shape[1]
        assert inspector.n_samples == {"train": X.shape[0]}

    def test_init_with_x_axis(self, fitted_preprocessing_pipeline, spectral_data):
        """Test initialisation with explicit x_axis."""
        # Arrange
        X, y, x_axis = spectral_data

        # Act
        inspector = PreprocessingInspector(
            fitted_preprocessing_pipeline, X, y, x_axis=x_axis
        )

        # Assert
        np.testing.assert_array_equal(inspector.x_axis, x_axis)
        assert inspector.feature_names is not None

    def test_init_with_all_datasets(self, fitted_preprocessing_pipeline, split_data):
        """Test initialisation with train, test, and validation datasets."""
        # Arrange / Act
        inspector = PreprocessingInspector(
            fitted_preprocessing_pipeline,
            X_train=split_data["X_train"],
            y_train=split_data["y_train"],
            X_test=split_data["X_test"],
            y_test=split_data["y_test"],
            X_val=split_data["X_val"],
            y_val=split_data["y_val"],
            x_axis=split_data["x_axis"],
        )

        # Assert
        assert set(inspector.n_samples.keys()) == {"train", "test", "val"}
        assert inspector.n_samples["train"] == 30
        assert inspector.n_samples["test"] == 10
        assert inspector.n_samples["val"] == 10

    def test_init_without_y(self, fitted_preprocessing_pipeline, spectral_data):
        """Test initialisation without target values."""
        # Arrange
        X, _, _ = spectral_data

        # Act
        inspector = PreprocessingInspector(fitted_preprocessing_pipeline, X)

        # Assert
        assert inspector.n_features == X.shape[1]

    def test_not_a_pipeline_raises(self, spectral_data):
        """Test that passing a non-Pipeline model raises TypeError."""
        # Arrange
        X, _, _ = spectral_data
        model = StandardScaler().fit(X)

        # Act / Assert
        with pytest.raises(TypeError, match="Expected a sklearn Pipeline"):
            PreprocessingInspector(model, X)

    def test_unfitted_pipeline_raises(self, spectral_data):
        """Test that an unfitted pipeline raises RuntimeError."""
        # Arrange
        X, _, _ = spectral_data
        pipe = make_pipeline(StandardScaler(), MinMaxScaler())

        # Act / Assert
        with pytest.raises(RuntimeError, match="must be fitted"):
            PreprocessingInspector(pipe, X)

    def test_mismatched_x_axis_raises(
        self, fitted_preprocessing_pipeline, spectral_data
    ):
        """Test that mismatched x_axis length raises ValueError."""
        # Arrange
        X, y, _ = spectral_data
        wrong_x_axis = np.arange(5)

        # Act / Assert
        with pytest.raises(ValueError, match="x_axis length"):
            PreprocessingInspector(
                fitted_preprocessing_pipeline, X, y, x_axis=wrong_x_axis
            )

    def test_mismatched_test_features_raises(
        self, fitted_preprocessing_pipeline, spectral_data
    ):
        """Test that X_test with wrong feature count raises ValueError."""
        # Arrange
        X, y, _ = spectral_data
        X_test_wrong = np.ones((10, 3))

        # Act / Assert
        with pytest.raises(ValueError, match="same number of features"):
            PreprocessingInspector(
                fitted_preprocessing_pipeline, X, y, X_test=X_test_wrong
            )

    def test_mismatched_val_features_raises(
        self, fitted_preprocessing_pipeline, spectral_data
    ):
        """Test that X_val with wrong feature count raises ValueError."""
        # Arrange
        X, y, _ = spectral_data
        X_val_wrong = np.ones((10, 3))

        # Act / Assert
        with pytest.raises(ValueError, match="same number of features"):
            PreprocessingInspector(
                fitted_preprocessing_pipeline, X, y, X_val=X_val_wrong
            )

    def test_mismatched_y_samples_raises(
        self, fitted_preprocessing_pipeline, spectral_data
    ):
        """Test that y_train with wrong sample count raises ValueError."""
        # Arrange
        X, _, _ = spectral_data
        y_wrong = np.ones(5)

        # Act / Assert
        with pytest.raises(ValueError, match="samples"):
            PreprocessingInspector(fitted_preprocessing_pipeline, X, y_train=y_wrong)


# ===========================================================================
# Step detection tests
# ===========================================================================


class TestPreprocessingStepDetection:
    """Test that model steps are correctly excluded."""

    def test_all_preprocessing_steps(
        self, fitted_preprocessing_pipeline, spectral_data
    ):
        """Test that all transformer steps are kept."""
        # Arrange
        X, _, _ = spectral_data

        # Act
        inspector = PreprocessingInspector(fitted_preprocessing_pipeline, X)

        # Assert
        assert len(inspector.preprocessing_steps) == 2
        step_types = [type(s).__name__ for _, s in inspector.preprocessing_steps]
        assert step_types == ["StandardScaler", "MinMaxScaler"]

    def test_pca_excluded(self, fitted_pipeline_with_pca, spectral_data):
        """Test that PCA is automatically excluded from preprocessing steps."""
        # Arrange
        X, _, _ = spectral_data

        # Act
        inspector = PreprocessingInspector(fitted_pipeline_with_pca, X)

        # Assert
        assert len(inspector.preprocessing_steps) == 2
        step_types = [type(s).__name__ for _, s in inspector.preprocessing_steps]
        assert "PCA" not in step_types
        assert step_types == ["StandardScaler", "MinMaxScaler"]

    def test_regressor_excluded(self, fitted_pipeline_with_svr, spectral_data):
        """Test that a regressor (SVR) is excluded from preprocessing steps."""
        # Arrange
        X, y, _ = spectral_data

        # Act
        inspector = PreprocessingInspector(fitted_pipeline_with_svr, X, y)

        # Assert
        assert len(inspector.preprocessing_steps) == 1
        step_types = [type(s).__name__ for _, s in inspector.preprocessing_steps]
        assert "SVR" not in step_types
        assert step_types == ["StandardScaler"]

    def test_nmf_excluded(self, spectral_data):
        """Test that NMF is automatically excluded from preprocessing steps."""
        # Arrange
        X, _, _ = spectral_data
        X_pos = np.abs(X)  # NMF requires non-negative data
        pipe = make_pipeline(MinMaxScaler(), NMF(n_components=5))
        pipe.fit(X_pos)

        # Act
        inspector = PreprocessingInspector(pipe, X_pos)

        # Assert
        assert len(inspector.preprocessing_steps) == 1
        step_types = [type(s).__name__ for _, s in inspector.preprocessing_steps]
        assert "NMF" not in step_types

    def test_fast_ica_excluded(self, spectral_data):
        """Test that FastICA is automatically excluded from preprocessing steps."""
        # Arrange
        X, _, _ = spectral_data
        pipe = make_pipeline(StandardScaler(), FastICA(n_components=5))
        pipe.fit(X)

        # Act
        inspector = PreprocessingInspector(pipe, X)

        # Assert
        assert len(inspector.preprocessing_steps) == 1
        step_types = [type(s).__name__ for _, s in inspector.preprocessing_steps]
        assert "FastICA" not in step_types

    def test_kernel_pca_excluded(self, spectral_data):
        """Test that KernelPCA is automatically excluded."""
        # Arrange
        X, _, _ = spectral_data
        pipe = make_pipeline(StandardScaler(), KernelPCA(n_components=5))
        pipe.fit(X)

        # Act
        inspector = PreprocessingInspector(pipe, X)

        # Assert
        assert len(inspector.preprocessing_steps) == 1
        step_types = [type(s).__name__ for _, s in inspector.preprocessing_steps]
        assert "KernelPCA" not in step_types

    def test_passthrough_step_skipped(self, spectral_data):
        """Test that 'passthrough' pipeline steps are correctly ignored."""
        # Arrange
        X, _, _ = spectral_data
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("noop", "passthrough"),
                ("minmax", MinMaxScaler()),
            ]
        )
        pipe.fit(X)

        # Act
        inspector = PreprocessingInspector(pipe, X)

        # Assert — passthrough is excluded, only real transformers remain
        assert len(inspector.preprocessing_steps) == 2
        step_types = [type(s).__name__ for _, s in inspector.preprocessing_steps]
        assert step_types == ["StandardScaler", "MinMaxScaler"]

    def test_all_model_steps_raises(self, spectral_data):
        """Test that a pipeline with only model steps raises ValueError."""
        # Arrange
        X, _, _ = spectral_data
        pipe = make_pipeline(PCA(n_components=3))
        pipe.fit(X)

        # Act / Assert
        with pytest.raises(ValueError, match="does not contain any preprocessing"):
            PreprocessingInspector(pipe, X)


# ===========================================================================
# Properties and data access
# ===========================================================================


class TestProperties:
    """Test property accessors."""

    def test_pipeline_property(self, fitted_preprocessing_pipeline, spectral_data):
        """Test that the pipeline property returns the original pipeline."""
        # Arrange
        X, _, _ = spectral_data

        # Act
        inspector = PreprocessingInspector(fitted_preprocessing_pipeline, X)

        # Assert
        assert inspector.pipeline is fitted_preprocessing_pipeline

    def test_model_property(self, fitted_preprocessing_pipeline, spectral_data):
        """Test that the model property is an alias for pipeline."""
        # Arrange
        X, _, _ = spectral_data

        # Act
        inspector = PreprocessingInspector(fitted_preprocessing_pipeline, X)

        # Assert
        assert inspector.model is inspector.pipeline
        assert inspector.model is fitted_preprocessing_pipeline

    def test_transformer_property(self, fitted_preprocessing_pipeline, spectral_data):
        """Test that the transformer property returns a Pipeline."""
        # Arrange
        X, _, _ = spectral_data

        # Act
        inspector = PreprocessingInspector(fitted_preprocessing_pipeline, X)
        transformer = inspector.transformer

        # Assert
        assert transformer is not None
        assert isinstance(transformer, Pipeline)
        assert len(transformer.steps) == 2

    def test_transformer_excludes_model_steps(
        self, fitted_pipeline_with_pca, spectral_data
    ):
        """Test that the transformer property excludes PCA."""
        # Arrange
        X, _, _ = spectral_data

        # Act
        inspector = PreprocessingInspector(fitted_pipeline_with_pca, X)
        transformer = inspector.transformer

        # Assert
        assert transformer is not None
        step_types = [type(s).__name__ for _, s in transformer.steps]
        assert "PCA" not in step_types

    def test_feature_names_none_by_default(
        self, fitted_preprocessing_pipeline, spectral_data
    ):
        """Test that feature_names is None when no x_axis is provided."""
        # Arrange
        X, _, _ = spectral_data

        # Act
        inspector = PreprocessingInspector(fitted_preprocessing_pipeline, X)

        # Assert
        assert inspector.feature_names is None

    def test_feature_names_with_x_axis(
        self, fitted_preprocessing_pipeline, spectral_data
    ):
        """Test that feature_names is set when x_axis is provided."""
        # Arrange
        X, _, x_axis = spectral_data

        # Act
        inspector = PreprocessingInspector(
            fitted_preprocessing_pipeline, X, x_axis=x_axis
        )

        # Assert
        np.testing.assert_array_equal(inspector.feature_names, x_axis)


# ===========================================================================
# SpectraMixin protocol compliance
# ===========================================================================


class TestSpectraMixinProtocol:
    """Test that SpectraMixin methods work correctly."""

    def test_get_raw_data(self, fitted_preprocessing_pipeline, spectral_data):
        """Test _get_raw_data returns the original arrays."""
        # Arrange
        X, y, _ = spectral_data
        inspector = PreprocessingInspector(fitted_preprocessing_pipeline, X, y)

        # Act
        X_raw, y_raw = inspector._get_raw_data("train")

        # Assert
        np.testing.assert_array_equal(X_raw, X)
        np.testing.assert_array_equal(y_raw, y)

    def test_get_preprocessed_data(self, fitted_preprocessing_pipeline, spectral_data):
        """Test _get_preprocessed_data transforms the data."""
        # Arrange
        X, _, _ = spectral_data
        inspector = PreprocessingInspector(fitted_preprocessing_pipeline, X)

        # Act
        X_prep = inspector._get_preprocessed_data("train")

        # Assert
        assert X_prep.shape == X.shape

    def test_preprocessed_data_cached(
        self, fitted_preprocessing_pipeline, spectral_data
    ):
        """Test that preprocessed data is cached across calls."""
        # Arrange
        X, _, _ = spectral_data
        inspector = PreprocessingInspector(fitted_preprocessing_pipeline, X)

        # Act
        X_prep1 = inspector._get_preprocessed_data("train")
        X_prep2 = inspector._get_preprocessed_data("train")

        # Assert
        assert X_prep1 is X_prep2

    def test_get_preprocessed_x_axis_same_features(
        self, fitted_preprocessing_pipeline, spectral_data
    ):
        """Test preprocessed x_axis when feature count is unchanged."""
        # Arrange
        X, _, x_axis = spectral_data
        inspector = PreprocessingInspector(
            fitted_preprocessing_pipeline, X, x_axis=x_axis
        )

        # Act
        prep_x = inspector._get_preprocessed_x_axis()

        # Assert
        np.testing.assert_array_equal(prep_x, x_axis)

    def test_invalid_dataset_raises(self, fitted_preprocessing_pipeline, spectral_data):
        """Test that requesting a missing dataset raises ValueError."""
        # Arrange
        X, _, _ = spectral_data
        inspector = PreprocessingInspector(fitted_preprocessing_pipeline, X)

        # Act / Assert
        with pytest.raises(ValueError, match="Test data not provided"):
            inspector._get_dataset("test")

    def test_inspect_spectra(self, fitted_preprocessing_pipeline, spectral_data):
        """Test inspect_spectra returns raw and preprocessed figures."""
        # Arrange
        X, y, _ = spectral_data
        inspector = PreprocessingInspector(fitted_preprocessing_pipeline, X, y)

        # Act
        figures = inspector.inspect_spectra()

        # Assert
        assert "raw_spectra" in figures
        assert "preprocessed_spectra" in figures
        assert len(figures) == 2


# ===========================================================================
# inspect() visualisation
# ===========================================================================


class TestInspect:
    """Test the main inspect() method."""

    def test_returns_correct_number_of_figures(
        self, fitted_preprocessing_pipeline, spectral_data
    ):
        """Test that inspect() returns one raw + one per preprocessing step."""
        # Arrange
        X, y, _ = spectral_data
        inspector = PreprocessingInspector(fitted_preprocessing_pipeline, X, y)

        # Act
        figures = inspector.inspect()

        # Assert — raw + 2 preprocessing steps
        assert len(figures) == 3
        assert "raw" in figures
        assert "step_1_standardscaler" in figures
        assert "step_2_minmaxscaler" in figures

    def test_pipeline_with_pca_excludes_pca(
        self, fitted_pipeline_with_pca, spectral_data
    ):
        """Test that PCA steps are excluded from generated figures."""
        # Arrange
        X, y, _ = spectral_data
        inspector = PreprocessingInspector(fitted_pipeline_with_pca, X, y)

        # Act
        figures = inspector.inspect()

        # Assert — raw + 2 preprocessing steps (PCA excluded)
        assert len(figures) == 3
        assert "raw" in figures
        for key in figures:
            assert "pca" not in key.lower() or key == "raw"

    def test_figures_are_matplotlib_figures(
        self, fitted_preprocessing_pipeline, spectral_data
    ):
        """Test that all returned values are matplotlib Figure objects."""
        # Arrange
        X, y, _ = spectral_data
        inspector = PreprocessingInspector(fitted_preprocessing_pipeline, X, y)

        # Act
        figures = inspector.inspect()

        # Assert
        for fig in figures.values():
            assert isinstance(fig, matplotlib.figure.Figure)

    def test_single_dataset_default(self, fitted_preprocessing_pipeline, spectral_data):
        """Test inspect() with explicit single dataset."""
        # Arrange
        X, y, _ = spectral_data
        inspector = PreprocessingInspector(fitted_preprocessing_pipeline, X, y)

        # Act
        figures = inspector.inspect(dataset="train")

        # Assert
        assert len(figures) == 3

    def test_multi_dataset(self, fitted_preprocessing_pipeline, split_data):
        """Test inspect() overlays multiple datasets on the same figure."""
        # Arrange
        inspector = PreprocessingInspector(
            fitted_preprocessing_pipeline,
            X_train=split_data["X_train"],
            y_train=split_data["y_train"],
            X_test=split_data["X_test"],
            y_test=split_data["y_test"],
        )

        # Act
        figures = inspector.inspect(dataset=["train", "test"])

        # Assert — raw + 2 steps
        assert len(figures) == 3

    def test_color_by_y(self, fitted_preprocessing_pipeline, spectral_data):
        """Test inspect() with color_by='y'."""
        # Arrange
        X, y, _ = spectral_data
        inspector = PreprocessingInspector(fitted_preprocessing_pipeline, X, y)

        # Act
        figures = inspector.inspect(color_by="y")

        # Assert
        assert len(figures) == 3

    def test_color_by_sample_index(self, fitted_preprocessing_pipeline, spectral_data):
        """Test inspect() with color_by='sample_index'."""
        # Arrange
        X, y, _ = spectral_data
        inspector = PreprocessingInspector(fitted_preprocessing_pipeline, X, y)

        # Act
        figures = inspector.inspect(color_by="sample_index")

        # Assert
        assert len(figures) == 3

    def test_color_by_none(self, fitted_preprocessing_pipeline, spectral_data):
        """Test inspect() without any colouring."""
        # Arrange
        X, y, _ = spectral_data
        inspector = PreprocessingInspector(fitted_preprocessing_pipeline, X, y)

        # Act
        figures = inspector.inspect(color_by=None)

        # Assert
        assert len(figures) == 3

    def test_color_by_dict(self, fitted_preprocessing_pipeline, spectral_data):
        """Test inspect() with a custom color_by dictionary."""
        # Arrange
        X, y, _ = spectral_data
        inspector = PreprocessingInspector(fitted_preprocessing_pipeline, X, y)
        custom_colors = {"train": np.arange(X.shape[0])}

        # Act
        figures = inspector.inspect(color_by=custom_colors)

        # Assert
        assert len(figures) == 3

    def test_xlim(self, fitted_preprocessing_pipeline, spectral_data):
        """Test inspect() with x-axis limits for zooming."""
        # Arrange
        X, y, x_axis = spectral_data
        inspector = PreprocessingInspector(
            fitted_preprocessing_pipeline, X, y, x_axis=x_axis
        )

        # Act
        figures = inspector.inspect(xlim=(1000, 2000))

        # Assert
        assert len(figures) == 3

    def test_figsize(self, fitted_preprocessing_pipeline, spectral_data):
        """Test that custom figsize is applied to all figures."""
        # Arrange
        X, y, _ = spectral_data
        inspector = PreprocessingInspector(fitted_preprocessing_pipeline, X, y)
        expected_width, expected_height = 8, 4

        # Act
        figures = inspector.inspect(figsize=(expected_width, expected_height))

        # Assert
        for fig in figures.values():
            w, h = fig.get_size_inches()
            assert abs(w - expected_width) < 0.5
            assert abs(h - expected_height) < 0.5

    def test_color_mode_categorical(self, fitted_preprocessing_pipeline, spectral_data):
        """Test categorical colour mode."""
        # Arrange
        X, _, _ = spectral_data
        y_cat = np.array(["A", "B"] * 25)
        inspector = PreprocessingInspector(fitted_preprocessing_pipeline, X, y_cat)

        # Act
        figures = inspector.inspect(color_by="y", color_mode="categorical")

        # Assert
        assert len(figures) == 3

    def test_color_mode_continuous(self, fitted_preprocessing_pipeline, spectral_data):
        """Test continuous colour mode."""
        # Arrange
        X, y, _ = spectral_data
        inspector = PreprocessingInspector(fitted_preprocessing_pipeline, X, y)

        # Act
        figures = inspector.inspect(color_by="y", color_mode="continuous")

        # Assert
        assert len(figures) == 3

    def test_without_y_no_color(self, fitted_preprocessing_pipeline, spectral_data):
        """Test inspect() with no y and no colouring."""
        # Arrange
        X, _, _ = spectral_data
        inspector = PreprocessingInspector(fitted_preprocessing_pipeline, X)

        # Act
        figures = inspector.inspect(color_by=None)

        # Assert
        assert len(figures) == 3

    def test_plot_title_contains_latest_step(
        self, fitted_preprocessing_pipeline, spectral_data
    ):
        """Test that each step's plot title highlights the latest step name."""
        # Arrange
        X, y, _ = spectral_data
        inspector = PreprocessingInspector(fitted_preprocessing_pipeline, X, y)

        # Act
        figures = inspector.inspect()
        step1_fig = figures["step_1_standardscaler"]
        step2_fig = figures["step_2_minmaxscaler"]

        # Assert
        step1_title = step1_fig.axes[0].get_title()
        step2_title = step2_fig.axes[0].get_title()
        assert "StandardScaler" in step1_title
        assert "MinMaxScaler" in step2_title


# ===========================================================================
# Summary
# ===========================================================================


class TestSummary:
    """Test the summary() method."""

    def test_summary_returns_dataclass(
        self, fitted_preprocessing_pipeline, spectral_data
    ):
        """Test that summary() returns a PreprocessingSummary dataclass."""
        # Arrange
        X, _, _ = spectral_data
        inspector = PreprocessingInspector(fitted_preprocessing_pipeline, X)

        # Act
        summary = inspector.summary()

        # Assert
        assert isinstance(summary, PreprocessingSummary)
        assert summary.n_preprocessing_steps == 2
        assert summary.n_excluded_steps == 0

    def test_summary_to_dict(self, fitted_preprocessing_pipeline, spectral_data):
        """Test that summary().to_dict() returns expected keys."""
        # Arrange
        X, _, _ = spectral_data
        inspector = PreprocessingInspector(fitted_preprocessing_pipeline, X)

        # Act
        d = inspector.summary().to_dict()

        # Assert
        assert isinstance(d, dict)
        assert d["n_preprocessing_steps"] == 2
        assert d["n_excluded_steps"] == 0
        assert "steps" in d

    def test_summary_with_excluded_step(self, fitted_pipeline_with_pca, spectral_data):
        """Test summary when a model step (PCA) is excluded."""
        # Arrange
        X, _, _ = spectral_data
        inspector = PreprocessingInspector(fitted_pipeline_with_pca, X)

        # Act
        summary = inspector.summary()

        # Assert
        assert summary.n_preprocessing_steps == 2
        assert summary.n_excluded_steps == 1
        assert summary.excluded[0]["type"] == "PCA"

    def test_summary_datasets(self, fitted_preprocessing_pipeline, split_data):
        """Test that summary reports all registered datasets."""
        # Arrange
        inspector = PreprocessingInspector(
            fitted_preprocessing_pipeline,
            X_train=split_data["X_train"],
            y_train=split_data["y_train"],
            X_test=split_data["X_test"],
        )

        # Act
        summary = inspector.summary()

        # Assert
        assert "train" in summary.n_samples
        assert "test" in summary.n_samples

    def test_summary_repr(self, fitted_preprocessing_pipeline, spectral_data):
        """Test that summary repr contains expected information."""
        # Arrange
        X, _, _ = spectral_data
        inspector = PreprocessingInspector(fitted_preprocessing_pipeline, X)

        # Act
        text = repr(inspector.summary())

        # Assert
        assert "Preprocessing Inspector Summary" in text
        assert "StandardScaler" in text
        assert "MinMaxScaler" in text


# ===========================================================================
# Figure management
# ===========================================================================


class TestFigureManagement:
    """Test figure tracking and cleanup."""

    def test_close_figures(self, fitted_preprocessing_pipeline, spectral_data):
        """Test that close_figures clears all tracked figures."""
        # Arrange
        X, y, _ = spectral_data
        inspector = PreprocessingInspector(fitted_preprocessing_pipeline, X, y)
        inspector.inspect()

        # Act
        inspector.close_figures()

        # Assert
        assert len(inspector._tracked_figures) == 0

    def test_re_inspect_closes_previous(
        self, fitted_preprocessing_pipeline, spectral_data
    ):
        """Test that calling inspect() again closes previously tracked figures."""
        # Arrange
        X, y, _ = spectral_data
        inspector = PreprocessingInspector(fitted_preprocessing_pipeline, X, y)
        inspector.inspect()

        # Act
        figures2 = inspector.inspect()

        # Assert — only the new figures are tracked
        assert len(inspector._tracked_figures) == len(figures2)


# ===========================================================================
# Edge cases
# ===========================================================================


class TestEdgeCases:
    """Test edge cases and unusual configurations."""

    def test_single_step_pipeline(self, spectral_data):
        """Test a pipeline with a single preprocessing step."""
        # Arrange
        X, y, _ = spectral_data
        pipe = make_pipeline(StandardScaler())
        pipe.fit(X)

        # Act
        inspector = PreprocessingInspector(pipe, X, y)
        figures = inspector.inspect()

        # Assert — raw + 1 step
        assert len(figures) == 2

    def test_many_steps_pipeline(self, spectral_data):
        """Test a pipeline with many preprocessing steps and PCA excluded."""
        # Arrange
        X, _, _ = spectral_data
        pipe = make_pipeline(
            StandardScaler(),
            MinMaxScaler(),
            StandardScaler(),
            MinMaxScaler(),
            PCA(n_components=5),
        )
        pipe.fit(X)

        # Act
        inspector = PreprocessingInspector(pipe, X)
        figures = inspector.inspect()

        # Assert — raw + 4 preprocessing steps (PCA excluded)
        assert len(figures) == 5

    def test_2d_y_gets_flattened(self, fitted_preprocessing_pipeline, spectral_data):
        """Test that a 2D y with one column is flattened to 1D."""
        # Arrange
        X, y, _ = spectral_data
        y_2d = y.reshape(-1, 1)

        # Act
        inspector = PreprocessingInspector(fitted_preprocessing_pipeline, X, y_2d)
        _, y_out = inspector._get_raw_data("train")

        # Assert
        assert y_out is not None
        assert y_out.ndim == 1


# ===========================================================================
# __repr__
# ===========================================================================


class TestRepr:
    """Test __repr__ output."""

    def test_repr_basic(self, fitted_preprocessing_pipeline, spectral_data):
        """Test that repr contains essential info."""
        # Arrange
        X, _, _ = spectral_data
        inspector = PreprocessingInspector(fitted_preprocessing_pipeline, X)

        # Act
        text = repr(inspector)

        # Assert
        assert "PreprocessingInspector" in text
        assert "steps=2" in text
        assert f"features={X.shape[1]}" in text
        assert "train" in text

    def test_repr_with_multiple_datasets(
        self, fitted_preprocessing_pipeline, split_data
    ):
        """Test that repr lists all datasets."""
        # Arrange
        inspector = PreprocessingInspector(
            fitted_preprocessing_pipeline,
            X_train=split_data["X_train"],
            y_train=split_data["y_train"],
            X_test=split_data["X_test"],
            y_test=split_data["y_test"],
        )

        # Act
        text = repr(inspector)

        # Assert
        assert "train" in text
        assert "test" in text


# ===========================================================================
# P3 — Coverage gap tests
# ===========================================================================


class TestFeatureSelectionStep:
    """Test that pipelines with feature-selection steps work correctly."""

    def test_inspect_with_feature_selection(self, spectral_data):
        """inspect() handles dimensionality change from feature selection."""
        # Arrange
        X, y, x_axis = spectral_data
        pipe = make_pipeline(StandardScaler(), SelectKBest(f_regression, k=50))
        pipe.fit(X, y)
        inspector = PreprocessingInspector(pipe, X, y, x_axis=x_axis)

        # Act
        figures = inspector.inspect()

        # Assert — raw + 2 steps
        assert len(figures) == 3
        assert "raw" in figures

    def test_resolve_step_x_axis_with_fewer_features(self, spectral_data):
        """_resolve_step_x_axis returns integer indices when features decrease."""
        # Arrange
        X, y, x_axis = spectral_data
        pipe = make_pipeline(StandardScaler(), SelectKBest(f_regression, k=50))
        pipe.fit(X, y)
        inspector = PreprocessingInspector(pipe, X, y, x_axis=x_axis)

        # Act — get preprocessed x_axis (50 features instead of 200)
        prep_x = inspector._get_preprocessed_x_axis()

        # Assert
        assert prep_x.shape[0] == 50

    def test_inspect_spectra_with_feature_selection(self, spectral_data):
        """inspect_spectra handles changed feature count in preprocessed data."""
        # Arrange
        X, y, x_axis = spectral_data
        pipe = make_pipeline(StandardScaler(), SelectKBest(f_regression, k=50))
        pipe.fit(X, y)
        inspector = PreprocessingInspector(pipe, X, y, x_axis=x_axis)

        # Act
        figures = inspector.inspect_spectra()

        # Assert
        assert "raw_spectra" in figures
        assert "preprocessed_spectra" in figures


class TestMultiDatasetSpectra:
    """Test inspect_spectra with multiple datasets."""

    def test_inspect_spectra_multi_dataset(
        self, fitted_preprocessing_pipeline, split_data
    ):
        """inspect_spectra overlays train and test on the same figure."""
        # Arrange
        inspector = PreprocessingInspector(
            fitted_preprocessing_pipeline,
            X_train=split_data["X_train"],
            y_train=split_data["y_train"],
            X_test=split_data["X_test"],
            y_test=split_data["y_test"],
        )

        # Act
        figures = inspector.inspect_spectra(dataset=["train", "test"])

        # Assert
        assert "raw_spectra" in figures
        assert "preprocessed_spectra" in figures
        assert isinstance(figures["raw_spectra"], matplotlib.figure.Figure)

    def test_inspect_spectra_three_datasets(
        self, fitted_preprocessing_pipeline, split_data
    ):
        """inspect_spectra with train, test, and val."""
        # Arrange
        inspector = PreprocessingInspector(
            fitted_preprocessing_pipeline,
            X_train=split_data["X_train"],
            y_train=split_data["y_train"],
            X_test=split_data["X_test"],
            y_test=split_data["y_test"],
            X_val=split_data["X_val"],
            y_val=split_data["y_val"],
        )

        # Act
        figures = inspector.inspect_spectra(dataset=["train", "test", "val"])

        # Assert
        assert "raw_spectra" in figures
        assert "preprocessed_spectra" in figures


class TestChemotoolsNativeTransformers:
    """Test pipelines with chemotools-native transformers."""

    def test_snv_in_pipeline(self, spectral_data):
        """Pipeline with StandardNormalVariate works correctly."""
        # Arrange
        from chemotools.scatter import StandardNormalVariate

        X, y, x_axis = spectral_data
        pipe = make_pipeline(StandardNormalVariate(), MinMaxScaler())
        pipe.fit(X)

        # Act
        inspector = PreprocessingInspector(pipe, X, y, x_axis=x_axis)
        figures = inspector.inspect()

        # Assert
        assert len(figures) == 3
        assert "raw" in figures
        step_types = [type(s).__name__ for _, s in inspector.preprocessing_steps]
        assert "StandardNormalVariate" in step_types

    def test_savitzky_golay_in_pipeline(self, spectral_data):
        """Pipeline with SavitzkyGolay derivative works correctly."""
        # Arrange
        from chemotools.derivative import SavitzkyGolay

        X, y, x_axis = spectral_data
        pipe = make_pipeline(SavitzkyGolay(window_size=11, polynomial_order=2))
        pipe.fit(X)

        # Act
        inspector = PreprocessingInspector(pipe, X, y, x_axis=x_axis)
        figures = inspector.inspect()

        # Assert
        assert len(figures) == 2  # raw + 1 step

    def test_mixed_chemotools_sklearn_pipeline(self, spectral_data):
        """Pipeline mixing chemotools and sklearn transformers."""
        # Arrange
        from chemotools.scatter import StandardNormalVariate

        X, y, x_axis = spectral_data
        pipe = make_pipeline(
            StandardNormalVariate(),
            StandardScaler(),
            MinMaxScaler(),
        )
        pipe.fit(X)

        # Act
        inspector = PreprocessingInspector(pipe, X, y, x_axis=x_axis)
        figures = inspector.inspect()

        # Assert
        assert len(figures) == 4  # raw + 3 steps


class TestLargePipeline:
    """Test with pipelines that have many steps."""

    def test_five_plus_steps(self, spectral_data):
        """Pipeline with 5+ preprocessing steps produces correct figure count."""
        # Arrange
        from chemotools.scatter import StandardNormalVariate

        X, y, x_axis = spectral_data
        pipe = make_pipeline(
            StandardNormalVariate(),
            StandardScaler(),
            MinMaxScaler(),
            StandardScaler(),
            MinMaxScaler(),
        )
        pipe.fit(X)

        # Act
        inspector = PreprocessingInspector(pipe, X, y, x_axis=x_axis)
        figures = inspector.inspect()

        # Assert — raw + 5 steps
        assert len(figures) == 6

    def test_five_plus_steps_with_pca_excluded(self, spectral_data):
        """Large pipeline with PCA at end still excludes PCA."""
        # Arrange
        X, _, x_axis = spectral_data
        pipe = make_pipeline(
            StandardScaler(),
            MinMaxScaler(),
            StandardScaler(),
            MinMaxScaler(),
            PCA(n_components=5),
        )
        pipe.fit(X)

        # Act
        inspector = PreprocessingInspector(pipe, X, x_axis=x_axis)
        figures = inspector.inspect()

        # Assert — raw + 4 preprocessing steps (PCA excluded)
        assert len(figures) == 5
        for key in figures:
            assert "pca" not in key.lower() or key == "raw"


class TestCrossInspectorConsistency:
    """Test that multiple inspectors can be created from the same pipeline."""

    def test_pca_and_preprocessing_inspector_same_pipeline(self, spectral_data):
        """PCAInspector and PreprocessingInspector from the same fitted pipeline."""
        # Arrange
        from chemotools.inspector import PCAInspector

        X, y, x_axis = spectral_data
        pipe = make_pipeline(StandardScaler(), MinMaxScaler(), PCA(n_components=5))
        pipe.fit(X)

        # Act
        pca_insp = PCAInspector(pipe, X, y, x_axis=x_axis)
        prep_insp = PreprocessingInspector(pipe, X, y, x_axis=x_axis)

        # Assert — raw data is the same
        X_raw_pca, _ = pca_insp._get_raw_data("train")
        X_raw_prep, _ = prep_insp._get_raw_data("train")
        np.testing.assert_array_equal(X_raw_pca, X_raw_prep)

        # Both inspectors identify 2 preprocessing steps
        assert len(prep_insp.preprocessing_steps) == 2

    def test_inspect_spectra_without_preprocessing(self, spectral_data):
        """PCAInspector.inspect_spectra without a pipeline returns raw only."""
        # Arrange
        from chemotools.inspector import PCAInspector

        X, y, _ = spectral_data
        pca = PCA(n_components=5)
        pca.fit(X)
        inspector = PCAInspector(pca, X, y)

        # Act
        figures = inspector.inspect_spectra()

        # Assert — only raw, no preprocessed
        assert "raw_spectra" in figures
        assert "preprocessed_spectra" not in figures


class TestNestedPipelineWarning:
    """Test that nested pipelines / ColumnTransformer produce a warning."""

    def test_nested_pipeline_warns(self, spectral_data):
        """A nested Pipeline step triggers a UserWarning."""
        # Arrange
        X, _, _ = spectral_data
        inner = make_pipeline(StandardScaler(), MinMaxScaler())
        outer = Pipeline([("inner", inner), ("scaler2", StandardScaler())])
        outer.fit(X)

        # Act / Assert
        with pytest.warns(UserWarning, match="treated as a single opaque step"):
            PreprocessingInspector(outer, X)

    def test_column_transformer_warns(self, spectral_data):
        """A ColumnTransformer step triggers a UserWarning."""
        # Arrange
        X, _, _ = spectral_data
        ct = ColumnTransformer(
            [
                ("scaler", StandardScaler(), slice(0, 100)),
                ("minmax", MinMaxScaler(), slice(100, 200)),
            ]
        )
        pipe = Pipeline([("ct", ct), ("out_scaler", StandardScaler())])
        pipe.fit(X)

        # Act / Assert
        with pytest.warns(UserWarning, match="ColumnTransformer"):
            PreprocessingInspector(pipe, X)

    def test_nested_pipeline_still_works(self, spectral_data):
        """Despite the warning, inspect() still succeeds with nested pipelines."""
        # Arrange
        X, _, _ = spectral_data
        inner = make_pipeline(StandardScaler(), MinMaxScaler())
        outer = Pipeline([("inner", inner), ("scaler2", StandardScaler())])
        outer.fit(X)

        # Act
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            inspector = PreprocessingInspector(outer, X)
            figures = inspector.inspect()

        # Assert — raw + 2 steps (inner pipeline is treated as one step)
        assert len(figures) == 3
