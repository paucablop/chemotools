import numpy as np
import pytest
from numpy.testing import assert_allclose
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

from chemotools.inspector.mixins import RegressionMixin


class _CountingModel:
    def __init__(self, estimator):
        self._estimator = estimator
        self.predict_calls = 0

    def predict(self, X):
        self.predict_calls += 1
        return self._estimator.predict(X)


class _DummyInspectorBase:
    def __init__(self, *, model, raw_data, confidence=0.95, **kwargs):
        self.model = model
        self.confidence = confidence
        self._raw_store = raw_data
        self.datasets_ = {name: object() for name in raw_data}
        super().__init__(**kwargs)

    def _get_raw_data(self, dataset: str):
        return self._raw_store[dataset]


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
    model = _CountingModel(estimator)

    raw_data = {
        "train": (X_train, y_train),
        "test": (X_test, y_test),
    }

    inspector = DummyInspector(model=model, raw_data=raw_data)
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


def test_detectors_are_cached(monkeypatch, regression_setup):
    """Test that outlier detectors (Leverage, StudentizedResiduals) are created once and cached."""
    # Arrange
    inspector, raw_data, _ = regression_setup
    X_train, y_train = raw_data["train"]

    class FakeLeverage:
        def __init__(self, model, confidence):
            self.model = model
            self.confidence = confidence
            self.fit_calls = 0
            self.fitted_with = None

        def fit(self, X, y):
            self.fit_calls += 1
            self.fitted_with = (X, y)
            return self

    class FakeStudentized:
        def __init__(self, model, confidence):
            self.model = model
            self.confidence = confidence
            self.fit_calls = 0
            self.fitted_with = None

        def fit(self, X, y):
            self.fit_calls += 1
            self.fitted_with = (X, y)
            return self

    monkeypatch.setattr(
        "chemotools.inspector.mixins._regression.Leverage",
        FakeLeverage,
    )
    monkeypatch.setattr(
        "chemotools.inspector.mixins._regression.StudentizedResiduals",
        FakeStudentized,
    )

    # Act
    leverage = inspector.leverage_detector
    student = inspector.studentized_detector
    leverage_again = inspector.leverage_detector
    student_again = inspector.studentized_detector

    # Assert
    assert leverage.fit_calls == 1
    assert student.fit_calls == 1
    assert_allclose(leverage.fitted_with[0], X_train)
    assert_allclose(leverage.fitted_with[1], y_train)
    assert_allclose(student.fitted_with[0], X_train)
    assert_allclose(student.fitted_with[1], y_train)
    assert leverage_again is leverage
    assert student_again is student
