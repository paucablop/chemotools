"""Tests for PLSDA classifier."""

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.estimator_checks import check_estimator

from chemotools.classification import PLSDA


def test_compliance_pls_da():
    check_estimator(PLSDA())


class TestPLSDAFunctionality:
    def test_binary_classification(self):
        X, y = make_classification(
            n_samples=100, n_features=20, n_classes=2, random_state=42
        )
        clf = PLSDA(n_components=3).fit(X, y)
        preds = clf.predict(X)
        assert set(np.unique(preds)) <= set(np.unique(y))
        assert clf.score(X, y) > 0.8

    def test_multiclass_classification(self):
        X, y = make_classification(
            n_samples=150,
            n_features=20,
            n_classes=3,
            n_informative=6,
            random_state=42,
        )
        clf = PLSDA(n_components=4).fit(X, y)
        assert clf.centroids_.shape == (3, 4)
        assert clf.score(X, y) > 0.7

    def test_predict_proba_sums_to_one(self):
        X, y = make_classification(
            n_samples=100, n_features=20, n_classes=2, random_state=0
        )
        clf = PLSDA(n_components=3).fit(X, y)
        proba = clf.predict_proba(X)
        np.testing.assert_array_almost_equal(proba.sum(axis=1), np.ones(len(X)))

    def test_predict_proba_matches_predict(self):
        X, y = make_classification(
            n_samples=100, n_features=20, n_classes=3, n_informative=6, random_state=0
        )
        clf = PLSDA(n_components=3).fit(X, y)
        proba = clf.predict_proba(X)
        preds_from_proba = clf.classes_[np.argmax(proba, axis=1)]
        np.testing.assert_array_equal(preds_from_proba, clf.predict(X))

    def test_decision_function_agrees_with_predict_binary(self):
        X, y = make_classification(
            n_samples=100, n_features=20, n_classes=2, random_state=0
        )
        clf = PLSDA(n_components=3).fit(X, y)
        scores = clf.decision_function(X)
        assert scores.shape == (100,)
        preds_from_scores = clf.classes_[(scores > 0).astype(int)]
        np.testing.assert_array_equal(preds_from_scores, clf.predict(X))

    def test_decision_function_agrees_with_predict_multiclass(self):
        X, y = make_classification(
            n_samples=150, n_features=20, n_classes=3, n_informative=6, random_state=0
        )
        clf = PLSDA(n_components=4).fit(X, y)
        scores = clf.decision_function(X)
        assert scores.shape == (150, 3)
        preds_from_scores = clf.classes_[np.argmax(scores, axis=1)]
        np.testing.assert_array_equal(preds_from_scores, clf.predict(X))

    def test_string_labels(self):
        X, y = make_classification(
            n_samples=90, n_features=20, n_classes=3, n_informative=6, random_state=1
        )
        string_labels = np.array(["brazil", "ethiopia", "vietnam"])[y]
        clf = PLSDA(n_components=3).fit(X, string_labels)
        preds = clf.predict(X)
        assert set(np.unique(preds)) <= {"brazil", "ethiopia", "vietnam"}

    def test_pipeline_integration(self):
        X, y = make_classification(
            n_samples=100, n_features=20, n_classes=2, random_state=42
        )
        pipe = Pipeline(
            [("scaler", StandardScaler()), ("plsda", PLSDA(n_components=3))]
        )
        pipe.fit(X, y)
        assert pipe.score(X, y) > 0.8

    def test_more_classes_than_components(self):
        # K=5 classes, only 2 latent components: must not crash, and must
        # still produce valid labels for every sample.
        X, y = make_classification(
            n_samples=200,
            n_features=30,
            n_classes=5,
            n_informative=15,
            n_clusters_per_class=1,
            random_state=7,
        )
        clf = PLSDA(n_components=2).fit(X, y)
        preds = clf.predict(X)
        assert set(np.unique(preds)) <= set(np.unique(y))
        assert clf.centroids_.shape == (5, 2)

    def test_imbalanced_classes(self):
        # 90 samples of class 0, 10 of class 1 -- nearest-centroid should
        # still assign every sample to a valid class without error, and
        # should not collapse to predicting only the majority class.
        rng = np.random.RandomState(0)
        X_majority = rng.randn(90, 15) + np.array([0] * 15)
        X_minority = rng.randn(10, 15) + np.array([4] * 15)
        X = np.vstack([X_majority, X_minority])
        y = np.array([0] * 90 + [1] * 10)

        clf = PLSDA(n_components=3).fit(X, y)
        preds = clf.predict(X)
        assert set(np.unique(preds)) <= {0, 1}
        # With well-separated clusters, the minority class should still be
        # recoverable -- not swamped purely by majority-class centroid mass.
        assert (preds[90:] == 1).mean() > 0.7

    def test_single_sample_prediction(self):
        X, y = make_classification(
            n_samples=100, n_features=20, n_classes=2, random_state=42
        )
        clf = PLSDA(n_components=3).fit(X, y)
        single_pred = clf.predict(X[:1])
        single_proba = clf.predict_proba(X[:1])
        assert single_pred.shape == (1,)
        assert single_proba.shape == (1, 2)
        np.testing.assert_almost_equal(single_proba.sum(), 1.0)

    def test_fit_transform_matches_fit_then_transform(self):
        X, y = make_classification(
            n_samples=100, n_features=20, n_classes=2, random_state=42
        )
        scores_via_fit_transform = PLSDA(n_components=3).fit_transform(X, y)
        scores_via_two_step = PLSDA(n_components=3).fit(X, y).transform(X)
        np.testing.assert_array_almost_equal(
            scores_via_fit_transform, scores_via_two_step
        )

    def test_non_binary_integer_labels(self):
        # Class labels that are neither 0/1 nor a contiguous range -- makes
        # sure nothing secretly assumes labels are small sequential ints.
        X, y01 = make_classification(
            n_samples=100, n_features=20, n_classes=2, random_state=3
        )
        y = np.where(y01 == 0, 2, 7)
        clf = PLSDA(n_components=3).fit(X, y)
        preds = clf.predict(X)
        assert set(np.unique(preds)) <= {2, 7}

    def test_classes_sorted(self):
        X, y = make_classification(
            n_samples=90, n_features=20, n_classes=3, n_informative=6, random_state=1
        )
        string_labels = np.array(["vietnam", "brazil", "ethiopia"])[y]
        clf = PLSDA(n_components=3).fit(X, string_labels)
        np.testing.assert_array_equal(
            clf.classes_, np.array(["brazil", "ethiopia", "vietnam"])
        )

    def test_dataframe_feature_names_preserved(self):
        # Fitting on a DataFrame with string columns must record
        # feature_names_in_, so predicting on reordered columns is rejected
        # instead of silently producing wrong predictions.
        pd = pytest.importorskip("pandas")
        X, y = make_classification(
            n_samples=100, n_features=6, n_classes=2, random_state=42
        )
        columns = [f"feature_{i}" for i in range(6)]
        X_df = pd.DataFrame(X, columns=columns)

        clf = PLSDA(n_components=3).fit(X_df, y)
        np.testing.assert_array_equal(clf.feature_names_in_, columns)
        assert clf.predict(X_df).shape == (100,)

        with pytest.raises(ValueError, match="feature names"):
            clf.predict(X_df[columns[::-1]])
