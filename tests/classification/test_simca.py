"""Tests for the SIMCA classifier."""

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import SGDOneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.utils.estimator_checks import check_estimator

from chemotools.classification import SIMCA, OneClassPCA


def expected_failed_checks(estimator):
    return {
        "check_classifiers_train": (
            "Assumes 1D predictions; SIMCA returns a 2D membership matrix."
        ),
        "check_estimators_dtypes": (
            "The check supplies non-binary targets; SIMCA requires indicators."
        ),
        "check_classifier_data_not_an_array": (
            "The check supplies non-binary targets; SIMCA requires indicators."
        ),
        "check_classifiers_classes": (
            "The check assumes single-label targets and predictions."
        ),
        "check_fit2d_1feature": (
            "The check supplies non-binary targets; SIMCA requires indicators."
        ),
        "check_classifier_not_supporting_multiclass": (
            "Requires a binary-only error message; SIMCA instead requires "
            "a dense 2D multilabel indicator matrix."
        ),
    }


class TestSklearnCompatibility:
    @pytest.mark.parametrize(
        "estimator",
        [
            OneClassPCA(n_components=1),
            OneClassSVM(),
            SGDOneClassSVM(random_state=0),
            IsolationForest(random_state=0),
            LocalOutlierFactor(novelty=True),
            EllipticEnvelope(random_state=0),
        ],
    )
    def test_sklearn_compatibility(self, estimator):
        # Arrange
        estimator = SIMCA(estimator=estimator)

        # Act / Assert
        check_estimator(
            estimator,
            expected_failed_checks=expected_failed_checks(estimator),
        )


@pytest.fixture
def class_data():
    X = np.random.default_rng(0).normal(size=(91, 4))
    X[-1] = 1_000
    y = np.zeros((91, 2), dtype=int)
    y[:60, 0] = 1
    y[30:90, 1] = 1
    return X, y


class TestSIMCAFunctionality:
    @pytest.mark.parametrize("names", [None, [], ["A", "B"]])
    def test_class_wise_fitting(self, class_data, names):
        # Arrange
        X, y = class_data
        base = OneClassPCA(n_components=1)
        clf = SIMCA(base, class_names=names)
        references = [clone(base).fit(X[y[:, k] == 1]) for k in range(2)]

        # Act
        result = clf.fit(X, y)

        # Assert
        assert result is clf
        assert not hasattr(base, "pca_")
        assert clf.estimators_[0] is not clf.estimators_[1]
        assert all(model is not base for model in clf.estimators_)
        assert clf.n_classes_ == 2
        assert clf.n_features_in_ == X.shape[1]
        np.testing.assert_array_equal(clf.classes_, [0, 1])
        np.testing.assert_array_equal(clf.class_names_, names or [0, 1])
        np.testing.assert_array_equal(clf.class_count_, y.sum(axis=0))
        for model, reference in zip(clf.estimators_, references):
            np.testing.assert_allclose(model.mean_, reference.mean_)
            np.testing.assert_allclose(
                model.score_samples(X), reference.score_samples(X)
            )

    def test_predictions_and_scores(self, class_data):
        # Arrange
        X, y = class_data
        clf = SIMCA(OneClassPCA(n_components=1)).fit(X, y)
        probes = np.vstack([np.zeros(4), np.full(4, 1_000)])

        # Act
        predictions = clf.predict(probes)
        decisions = clf.decision_function(probes)
        scores = clf.score_samples(probes)

        # Assert
        assert predictions.shape == decisions.shape == scores.shape == (2, 2)
        assert np.issubdtype(predictions.dtype, np.integer)
        np.testing.assert_array_equal(predictions, [[1, 1], [0, 0]])
        for k, model in enumerate(clf.estimators_):
            np.testing.assert_array_equal(predictions[:, k], model.predict(probes) == 1)
            np.testing.assert_allclose(decisions[:, k], model.decision_function(probes))
            np.testing.assert_allclose(scores[:, k], model.score_samples(probes))
        for method in ("predict", "decision_function", "score_samples"):
            assert getattr(clf, method)(probes[:1]).shape == (1, 2)

    def test_label_helpers_and_lookup(self, class_data):
        # Arrange
        X, y = class_data
        clf = SIMCA(OneClassPCA(n_components=1), class_names=["A", "B"]).fit(X, y)
        fresh = clone(clf)
        probes = np.vstack([np.zeros(4), np.full(4, 1_000)])

        # Act
        labels = clf.predict_labels(probes)
        training_labels = fresh.fit_predict_labels(X, y)
        model = clf.get_class_model("B")

        # Assert
        assert labels == [("A", "B"), ()]
        assert model is clf.estimators_[1]
        assert training_labels == clf.predict_labels(X)
        assert training_labels == [
            tuple(clf.class_names_[row.astype(bool)]) for row in clf.predict(X)
        ]
        with pytest.raises(ValueError, match="not in model classes"):
            clf.get_class_model("missing")

    @pytest.mark.parametrize("representation", ["float32", "float64", "int64", "list"])
    def test_input_representations_and_single_class(self, class_data, representation):
        # Arrange
        X, _ = class_data
        dtype = "float64" if representation == "list" else representation
        X = (10 * X[:60]).astype(dtype)
        y = np.ones((len(X), 1), dtype=float)
        reference = SIMCA(OneClassPCA(n_components=1)).fit(X, y)
        clf = SIMCA(OneClassPCA(n_components=1))
        X_input = X.tolist() if representation == "list" else X
        y_input = y.tolist() if representation == "list" else y

        # Act
        clf.fit(X_input, y_input)
        predictions = clf.predict(X_input)

        # Assert
        assert predictions.shape == (len(X), 1)
        assert clf.n_classes_ == 1
        assert clf.get_class_model(0) is clf.estimators_[0]
        np.testing.assert_array_equal(predictions, reference.predict(X))
        np.testing.assert_allclose(
            clf.score_samples(X_input), reference.score_samples(X)
        )


class TestSIMCAValidation:
    @pytest.mark.parametrize(
        "case, message",
        [
            ("continuous", "Unknown label type|continuous"),
            ("continuous_multioutput", "Unknown label type|continuous"),
            ("labels", "indicator matrix"),
            ("nonbinary", "indicator matrix"),
            ("empty_class", "at least one training sample"),
            ("duplicate_names", "unique"),
            ("wrong_names", "one name per target column"),
            ("one_feature", r"1 feature\(s\)"),
            ("invalid_predict", "predict"),
        ],
    )
    def test_invalid_fit_inputs(self, class_data, case, message):
        # Arrange
        X, y = class_data
        clf = SIMCA(OneClassPCA(n_components=1))
        if case == "continuous":
            y = np.linspace(0.1, 0.9, len(X))
        elif case == "continuous_multioutput":
            y = y * 0.5
        elif case == "labels":
            y = np.arange(len(X)) % 3
        elif case == "nonbinary":
            y = y * 2
        elif case == "empty_class":
            y[:, 1] = 0
        elif case == "duplicate_names":
            clf.set_params(class_names=["A", "A"])
        elif case == "wrong_names":
            clf.set_params(class_names=["A"])
        elif case == "one_feature":
            X = X[:, :1]
        else:
            clf.estimator.predict = None

        # Act / Assert
        error = TypeError if case == "invalid_predict" else ValueError
        with pytest.raises(error, match=message):
            clf.fit(X, y)

    @pytest.mark.parametrize(
        "method", ["predict_labels", "get_class_model", "add_model"]
    )
    def test_unfitted_helpers(self, class_data, method):
        # Arrange
        X, _ = class_data
        clf = SIMCA(OneClassPCA(n_components=1))
        argument = 0 if method == "get_class_model" else X

        # Act / Assert
        with pytest.raises(NotFittedError):
            getattr(clf, method)(argument)


class TestSIMCAModelManagement:
    @pytest.mark.parametrize("name", [None, "", "C"])
    def test_add_class_and_refit(self, class_data, name):
        # Arrange
        X, y = class_data
        clf = SIMCA(OneClassPCA(n_components=1)).fit(X, y)
        old_models = clf.estimators_.copy()
        old_decisions = clf.decision_function(X)
        X_new = X[:60] + 5
        reference = OneClassPCA(n_components=1).fit(X_new)

        # Act
        result = clf.add_model(X_new, class_name=name)

        # Assert
        assert result is clf
        assert all(clf.estimators_[k] is model for k, model in enumerate(old_models))
        assert clf.n_classes_ == 3
        assert clf.class_names is None
        np.testing.assert_array_equal(clf.classes_, [0, 1, 2])
        np.testing.assert_array_equal(
            clf.class_names_,
            np.array([0, 1, 2 if name is None or name == "" else name], dtype=object),
        )
        np.testing.assert_array_equal(clf.class_count_, [60, 60, 60])
        np.testing.assert_allclose(clf.decision_function(X)[:, :2], old_decisions)
        np.testing.assert_allclose(
            clf.decision_function(X)[:, 2], reference.decision_function(X)
        )
        np.testing.assert_array_equal(clf.predict(X)[:, 2], reference.predict(X) == 1)
        np.testing.assert_allclose(
            clf.score_samples(X)[:, 2], reference.score_samples(X)
        )

        # Act
        clf.fit(X, y)

        # Assert
        assert clf.n_classes_ == len(clf.estimators_) == 2
        np.testing.assert_array_equal(clf.classes_, [0, 1])
        np.testing.assert_array_equal(clf.class_names_, [0, 1])
        np.testing.assert_array_equal(clf.class_count_, [60, 60])
        np.testing.assert_allclose(clf.decision_function(X), old_decisions)

    @pytest.mark.parametrize(
        "case",
        [
            "duplicate",
            "default_collision",
            "features",
            "empty_name",
        ],
    )
    def test_failed_addition_preserves_state(self, class_data, case):
        # Arrange
        X, y = class_data
        clf = SIMCA(OneClassPCA(n_components=1), class_names=[0, 2]).fit(X, y)
        old_models = clf.estimators_.copy()
        old_decisions = clf.decision_function(X)
        X_new = X[:60, :3] if case == "features" else X[:60]
        name = {
            "duplicate": 2,
            "default_collision": None,
            "features": "C",
            "empty_name": "",
        }[case]

        # Act / Assert
        message = "features" if case == "features" else "already exists"
        with pytest.raises(ValueError, match=message):
            clf.add_model(X_new, class_name=name)

        assert clf.n_classes_ == len(clf.estimators_) == 2
        assert all(clf.estimators_[k] is model for k, model in enumerate(old_models))
        assert clf.n_features_in_ == X.shape[1]
        np.testing.assert_array_equal(clf.classes_, [0, 1])
        np.testing.assert_array_equal(clf.class_names_, [0, 2])
        np.testing.assert_array_equal(clf.class_count_, [60, 60])
        np.testing.assert_allclose(clf.decision_function(X), old_decisions)


class TestSIMCAIntegration:
    def test_parallel_equivalence(self, class_data):
        # Arrange
        X, y = class_data
        serial = SIMCA(OneClassPCA(n_components=1), n_jobs=1)
        parallel = SIMCA(OneClassPCA(n_components=1), n_jobs=2)

        # Act
        serial.fit(X, y)
        parallel.fit(X, y)

        # Assert
        np.testing.assert_array_equal(parallel.classes_, serial.classes_)
        np.testing.assert_array_equal(parallel.class_names_, serial.class_names_)
        np.testing.assert_array_equal(parallel.predict(X), serial.predict(X))
        np.testing.assert_allclose(
            parallel.decision_function(X), serial.decision_function(X)
        )
        np.testing.assert_allclose(parallel.score_samples(X), serial.score_samples(X))

    def test_pipeline_integration(self, class_data):
        # Arrange
        X, y = class_data
        pipeline = make_pipeline(StandardScaler(), SIMCA(OneClassPCA(n_components=1)))
        scaler = StandardScaler()
        manual = SIMCA(OneClassPCA(n_components=1))

        # Act
        pipeline.fit(X, y)
        X_scaled = scaler.fit_transform(X)
        manual.fit(X_scaled, y)

        # Assert
        np.testing.assert_array_equal(pipeline.predict(X), manual.predict(X_scaled))
        np.testing.assert_allclose(
            pipeline.decision_function(X), manual.decision_function(X_scaled)
        )
