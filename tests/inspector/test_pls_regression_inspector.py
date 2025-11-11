import numpy as np
import pytest
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from chemotools.inspector import PLSRegressionInspector


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
        X_train, y_train = regression_data["train"]
        X_test, y_test = regression_data["test"]
        X_val, y_val = regression_data["val"]

        inspector = PLSRegressionInspector(
            fitted_pls,
            X_train,
            y_train,
            X_test=X_test,
            y_test=y_test,
            X_val=X_val,
            y_val=y_val,
        )

        assert inspector.nr_samples == {"train": 50, "test": 20, "val": 20}
        assert inspector.nr_components == 3
        assert inspector.transformer is None

    def test_init_with_pipeline(self, fitted_pipeline, regression_data):
        X_train, y_train = regression_data["train"]

        inspector = PLSRegressionInspector(fitted_pipeline, X_train, y_train)

        assert inspector.transformer is not None
        assert inspector.estimator._estimator_type == "regressor"

    def test_missing_targets_raises(self, fitted_pls, regression_data):
        X_train, _ = regression_data["train"]
        X_test, y_test = regression_data["test"]

        with pytest.raises(ValueError, match="y_train required"):
            PLSRegressionInspector(fitted_pls, X_train, y_train=None)

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
        X_train, y_train = regression_data["train"]
        inspector = PLSRegressionInspector(fitted_pls, X_train, y_train)

        summary = inspector.summary()

        assert "RMSE" in summary
        assert "R2" in summary
        assert "model_type" in summary
        assert summary["model_type"].startswith("PLS")

    def test_summary_with_pipeline(self, fitted_pipeline, regression_data):
        X_train, y_train = regression_data["train"]
        inspector = PLSRegressionInspector(fitted_pipeline, X_train, y_train)

        summary = inspector.summary()

        assert summary["has_preprocessing"] is True
        assert len(summary["preprocessing_steps"]) == 1


class TestInspectFigures:
    def test_inspect_single_dataset(self, fitted_pls, regression_data):
        X_train, y_train = regression_data["train"]
        inspector = PLSRegressionInspector(fitted_pls, X_train, y_train)

        figures = inspector.inspect(
            dataset="train", components_scores=(0, 1), loadings_components=[0, 1]
        )

        expected_keys = {
            "scores_1",
            "x_vs_y_scores_1",
            "loadings_x",
            "loadings_weights",
            "loadings_rotations",
            "regression_coefficients",
            "distances_hotelling_q",
            "distances_leverage_studentized",
            "predicted_vs_actual",
            "residuals",
            "qq_plot",
            "residual_distribution",
        }
        assert expected_keys.issubset(figures.keys())

        for fig in figures.values():
            fig.canvas.draw_idle()

    def test_inspect_multi_dataset(self, fitted_pls, regression_data):
        X_train, y_train = regression_data["train"]
        X_test, y_test = regression_data["test"]
        inspector = PLSRegressionInspector(
            fitted_pls,
            X_train,
            y_train,
            X_test=X_test,
            y_test=y_test,
        )

        figures = inspector.inspect(
            dataset=["train", "test"], components_scores=((0, 1),)
        )

        assert "scores_1" in figures
        assert "distances_leverage_studentized" in figures
        assert "predicted_vs_actual" in figures
        assert "residuals" in figures

    def test_inspect_spectra_requires_pipeline(self, fitted_pls, regression_data):
        X_train, y_train = regression_data["train"]
        inspector = PLSRegressionInspector(fitted_pls, X_train, y_train)

        with pytest.raises(ValueError, match="requires a preprocessing"):
            inspector.inspect_spectra()

    def test_inspect_spectra_pipeline_single_dataset(
        self, fitted_pipeline, regression_data
    ):
        X_train, y_train = regression_data["train"]
        inspector = PLSRegressionInspector(fitted_pipeline, X_train, y_train)

        figures = inspector.inspect_spectra(dataset="train")

        assert set(figures.keys()) == {"raw_spectra", "preprocessed_spectra"}

    def test_inspect_spectra_pipeline_multi_dataset(
        self, fitted_pipeline, regression_data
    ):
        X_train, y_train = regression_data["train"]
        X_test, y_test = regression_data["test"]
        inspector = PLSRegressionInspector(
            fitted_pipeline,
            X_train,
            y_train,
            X_test=X_test,
            y_test=y_test,
        )

        figures = inspector.inspect_spectra(dataset=["train", "test"])

        assert set(figures.keys()) == {"raw_spectra", "preprocessed_spectra"}


class TestRegressionDiagnostics:
    def test_regression_metrics_cached(self, fitted_pls, regression_data):
        X_train, y_train = regression_data["train"]
        inspector = PLSRegressionInspector(fitted_pls, X_train, y_train)

        rmse_first = inspector.RMSE_train
        rmse_second = inspector.RMSE_train
        assert rmse_first == pytest.approx(rmse_second)

        r2_first = inspector.R2_train
        r2_second = inspector.R2_train
        assert r2_first == pytest.approx(r2_second)

    def test_regression_distances_keys(self, fitted_pls, regression_data):
        X_train, y_train = regression_data["train"]
        inspector = PLSRegressionInspector(fitted_pls, X_train, y_train)

        figures = inspector.inspect(
            dataset="train",
            components_scores=(0, 1),
            loadings_components=[0, 1],
            color_by_y=False,
        )

        assert "distances_leverage_studentized" in figures
        fig = figures["distances_leverage_studentized"]
        ax = fig.axes[0]
        assert ax.get_xlabel() == "Leverage"
        assert ax.get_ylabel() == "Studentized Residuals"


class TestAdditionalCoverage:
    def test_detector_limits_cached(self, fitted_pls, regression_data, monkeypatch):
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
            "chemotools.inspector._pls_regression_inspector.HotellingT2",
            _DummyHotelling,
        )
        monkeypatch.setattr(
            "chemotools.inspector._pls_regression_inspector.QResiduals",
            _DummyQ,
        )

        inspector = PLSRegressionInspector(fitted_pls, X_train, y_train)

        assert inspector.hotelling_t2_limit == pytest.approx(1.23)
        assert inspector.hotelling_t2_limit == pytest.approx(1.23)
        assert len(hot_calls) == 1

        assert inspector.q_residuals_limit == pytest.approx(4.56)
        assert inspector.q_residuals_limit == pytest.approx(4.56)
        assert len(q_calls) == 1

    def test_component_selection_helpers(self, fitted_pls, regression_data):
        X_train, y_train = regression_data["train"]
        inspector = PLSRegressionInspector(fitted_pls, X_train, y_train)

        all_loadings = inspector.get_x_loadings()
        single_loading = inspector.get_x_loadings(0)
        multi_loading = inspector.get_x_loadings([0, 1])

        assert single_loading.shape[1] == 1
        assert multi_loading.shape[1] == 2
        assert np.allclose(single_loading.squeeze(), all_loadings[:, 0])

        single_weight = inspector.get_x_weights(1)
        list_weights = inspector.get_x_weights([0, 2])
        assert single_weight.shape[1] == 1
        assert list_weights.shape[1] == 2

        single_rotation = inspector.get_x_rotations(1)
        list_rotations = inspector.get_x_rotations([0, 2])
        assert single_rotation.shape[1] == 1
        assert list_rotations.shape[1] == 2

    def test_regression_coefficients_multitarget_and_legend(
        self, fitted_pls_multi, multi_target_regression_data, monkeypatch
    ):
        X_train, y_train = multi_target_regression_data["train"]
        inspector = PLSRegressionInspector(fitted_pls_multi, X_train, y_train)

        coef = inspector.get_regression_coefficients()
        assert coef.shape[1] == 2

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

        figures = inspector.inspect(
            dataset="train",
            components_scores=(0, 1),
            loadings_components=[0, 1],
            color_by_y=False,
        )
        coef_fig = figures["regression_coefficients"]
        legend = coef_fig.axes[0].legend_
        assert legend is not None
        legend_labels = [text.get_text() for text in legend.get_texts()]
        assert legend_labels == ["Target 1", "Target 2"]

        for fig in figures.values():
            plt.close(fig)

    def test_create_latent_scores_missing_train_fallback(
        self, fitted_pls, regression_data, monkeypatch
    ):
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

        figures = inspector.create_latent_scores_figures(
            dataset="test",
            components=(0, 1),
            color_by_y=False,
            annotate_by=None,
            figsize=(3, 3),
        )

        assert "scores_1" in figures
        for fig in figures.values():
            plt.close(fig)

    def test_create_latent_scores_multi_dataset_combined(
        self, fitted_pls, regression_data
    ):
        X_train, y_train = regression_data["train"]
        X_test, y_test = regression_data["test"]
        inspector = PLSRegressionInspector(
            fitted_pls,
            X_train,
            y_train,
            X_test=X_test,
            y_test=y_test,
        )

        figures = inspector.create_latent_scores_figures(
            dataset=["train", "test"],
            components=(0, 1),
            color_by_y=False,
            annotate_by=None,
            figsize=(3, 3),
        )

        assert set(figures.keys()) == {"scores_1"}

        for fig in figures.values():
            plt.close(fig)

    def test_create_x_vs_y_scores_mixed_components(self, fitted_pls, regression_data):
        X_train, y_train = regression_data["train"]
        inspector = PLSRegressionInspector(fitted_pls, X_train, y_train)

        annotate_by = {"train": np.arange(X_train.shape[0])}
        figures = inspector._create_x_vs_y_scores_figures(
            components=[(0, 1), 2],
            color_by_y=True,
            annotate_by=annotate_by,
            figsize=(3, 3),
        )

        assert set(figures.keys()) == {"x_vs_y_scores_1"}
        for fig in figures.values():
            plt.close(fig)


class TestValidationPropagation:
    def test_y_length_mismatch_raises(self, fitted_pls, regression_data):
        X_train, y_train = regression_data["train"]
        X_test, y_test = regression_data["test"]

        with pytest.raises(ValueError, match="same number of samples"):
            PLSRegressionInspector(
                fitted_pls,
                X_train,
                y_train[:-1],
            )

        with pytest.raises(ValueError, match="same number of samples"):
            PLSRegressionInspector(
                fitted_pls,
                X_train,
                y_train,
                X_test=X_test,
                y_test=y_test[:-1],
            )
