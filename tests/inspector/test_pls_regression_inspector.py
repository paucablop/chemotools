import numpy as np
import pytest
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

        figures = inspector.inspect(dataset="train", components_scores=(0, 1))

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

        figures = inspector.inspect(dataset="train", components_scores=(0, 1))

        assert "distances_leverage_studentized" in figures
        fig = figures["distances_leverage_studentized"]
        ax = fig.axes[0]
        assert ax.get_xlabel() == "Leverage"
        assert ax.get_ylabel() == "Studentized Residuals"


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
