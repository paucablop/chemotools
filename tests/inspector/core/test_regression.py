import numpy as np
import pytest
from unittest.mock import MagicMock
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from matplotlib.figure import Figure

from chemotools.inspector.core.regression import RegressionMixin


class _CountingModel:
    def __init__(self, estimator):
        self._estimator = estimator
        self.predict_calls = 0

    def predict(self, X):
        self.predict_calls += 1
        return self._estimator.predict(X)


class _DummyInspectorBase:
    def __init__(self, *, model, raw_data, estimator=None, confidence=0.95, **kwargs):
        self.model = model
        self.estimator = estimator
        self.confidence = confidence
        self._raw_store = raw_data
        self.datasets_ = {name: object() for name in raw_data}
        super().__init__(**kwargs)

    def _get_raw_data(self, dataset: str):
        return self._raw_store[dataset]

    def _get_preprocessed_data(self, dataset: str):
        return self._raw_store[dataset][0]


class DummyInspector(RegressionMixin, _DummyInspectorBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


@pytest.fixture
def regression_setup():
    X_train = np.array([[0.0], [1.0], [2.0], [3.0]])
    y_train = np.array([1.0, 2.8, 4.1, 6.2])
    X_test = np.array([[4.0], [5.0]])
    y_test = np.array([7.9, 9.7])

    estimator = LinearRegression().fit(X_train, y_train)
    # Mock transform for leverage calculation
    estimator.transform = MagicMock(return_value=X_train)
    estimator.n_components = 1
    model = _CountingModel(estimator)

    raw_data = {
        "train": (X_train, y_train),
        "test": (X_test, y_test),
    }

    inspector = DummyInspector(model=model, raw_data=raw_data, estimator=estimator)
    return inspector, raw_data, estimator


def test_regression_metrics_match_sklearn(regression_setup):
    """Test that regression metrics (RMSE, R2) match sklearn calculations."""
    # Arrange
    inspector, raw_data, estimator = regression_setup
    X_train, y_train = raw_data["train"]
    X_test, y_test = raw_data["test"]

    expected_train_rmse = np.sqrt(
        mean_squared_error(y_train, estimator.predict(X_train))
    )
    expected_test_rmse = np.sqrt(mean_squared_error(y_test, estimator.predict(X_test)))
    expected_train_r2 = r2_score(y_train, estimator.predict(X_train))
    expected_test_r2 = r2_score(y_test, estimator.predict(X_test))

    # Act
    rmse_train = inspector.RMSE_train
    rmse_test = inspector.RMSE_test
    r2_train = inspector.R2_train
    r2_test = inspector.R2_test
    rmse_val = inspector.RMSE_val
    r2_val = inspector.R2_val

    # Assert
    assert rmse_train == pytest.approx(expected_train_rmse)
    assert rmse_test == pytest.approx(expected_test_rmse)
    assert r2_train == pytest.approx(expected_train_r2)
    assert r2_test == pytest.approx(expected_test_r2)
    assert rmse_val is None
    assert r2_val is None


def test_predictions_are_cached(regression_setup):
    """Test that model predictions are computed once and cached."""
    # Arrange
    inspector, _, _ = regression_setup

    # Act
    inspector._get_predictions("train")
    first_call_count = inspector.model.predict_calls

    inspector._get_predictions("train")
    second_call_count = inspector.model.predict_calls

    # Assert
    assert first_call_count == 1
    assert second_call_count == 1


def test_get_regression_raw_data_raises_value_error_when_y_is_none():
    # Arrange
    X = np.array([[1.0]])
    raw_data = {"no_target": (X, None)}
    inspector = DummyInspector(model=None, raw_data=raw_data)

    # Act & Assert
    with pytest.raises(
        ValueError, match="Target values not available for dataset 'no_target'"
    ):
        inspector._get_regression_raw_data("no_target")


def test_calculate_bias_handles_shape_mismatch():
    # Arrange
    # y_true is 2D (N, 1)
    X_train = np.array([[0.0], [1.0]])
    y_train = np.array([[1.0], [2.8]])

    estimator = LinearRegression().fit(X_train, y_train)

    raw_data = {
        "train": (X_train, y_train),
    }

    inspector = DummyInspector(model=estimator, raw_data=raw_data)

    # Act
    # This will trigger _calculate_bias -> _get_predictions (returns 1D) -> mismatch with y_true (2D)
    bias = inspector.regression_bias("train")

    # Assert
    assert isinstance(bias, float)
    # Bias should be approx 0
    assert bias == pytest.approx(0.0, abs=1e-10)


def test_get_datasets_data_skips_missing_datasets(regression_setup):
    # Arrange
    inspector, _, _ = regression_setup

    # Act
    data = inspector._get_datasets_data(["train", "non_existent"])

    # Assert
    assert "train" in data
    assert "non_existent" not in data
    assert len(data) == 1


def test_plotting_methods_smoke_test(regression_setup):
    # Arrange
    inspector, _, _ = regression_setup

    # Act
    fig1 = inspector.create_predicted_vs_actual_plot()
    fig2 = inspector.create_residuals_plot()
    fig3 = inspector.create_qq_plot()
    fig4 = inspector.create_residual_distribution_plot()

    # Assert
    assert isinstance(fig1, Figure)
    assert isinstance(fig2, Figure)
    assert isinstance(fig3, Figure)
    assert isinstance(fig4, Figure)
