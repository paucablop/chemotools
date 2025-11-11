import numpy as np

from chemotools.inspector.mixins import LatentVariableMixin


class _DummyDetector:
    def __init__(self, *, value, critical):
        self._value = value
        self.critical_value_ = critical
        self.fit_calls = []
        self.predict_calls = []

    def fit(self, X, y=None):
        self.fit_calls.append((X, y))
        return self

    def predict_residuals(self, X, y=None):
        self.predict_calls.append((X, y))
        return np.full(X.shape[0], self._value)


class DummyLatentInspector(LatentVariableMixin):
    component_label = "LV"

    def __init__(self):
        self.confidence = 0.95
        self.model = object()
        self._scores = {
            "train": np.array([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]]),
            "test": np.array([[0.5, 0.2], [0.3, 0.7], [0.6, 0.1]]),
        }
        self._explained = np.array([0.6, 0.3])
        self._raw = {
            "train": (np.ones((3, 3)), np.array([0, 1, 0])),
            "test": (np.zeros((3, 3)), np.array([1, 0, 1])),
        }
        self._feature_names = np.array([10, 20, 30])

    # LatentVariableMixin hooks
    def get_latent_scores(self, dataset: str) -> np.ndarray:
        return self._scores[dataset]

    def get_latent_explained_variance(self):
        return self._explained

    def get_latent_loadings(self) -> np.ndarray:
        return np.column_stack((np.array([1.0, 0.5, -0.1]), np.array([-0.2, 0.3, 0.9])))

    # Inspector bridge methods used internally
    def _get_preprocessed_feature_names(self):
        return self._feature_names

    def _get_raw_data(self, dataset):
        return self._raw[dataset]


def test_create_latent_scores_single_dataset():
    inspector = DummyLatentInspector()

    figures = inspector.create_latent_scores_figures(
        dataset="train",
        components=(0, 1),
        color_by_y=True,
        annotate_by="sample_index",
        figsize=(4, 4),
    )

    assert set(figures.keys()) == {"scores_1"}
    fig = figures["scores_1"]
    ax = fig.axes[0]
    assert ax.get_xlabel().startswith("LV1")


def test_create_latent_scores_multi_dataset():
    inspector = DummyLatentInspector()

    figures = inspector.create_latent_scores_figures(
        dataset=["train", "test"],
        components=((0, 1),),
        color_by_y=False,
        annotate_by=None,
        figsize=(4, 4),
    )

    assert "scores_1" in figures
    assert "scores_1_train" in figures
    assert "scores_1_test" in figures
    multi_ax = figures["scores_1"].axes[0]
    # Confidence ellipse from training data leaves at least one patch
    assert len(multi_ax.patches) >= 1


def test_create_latent_distance_runs_with_monkeypatched_detectors(monkeypatch):
    inspector = DummyLatentInspector()

    hot_instances = []
    q_instances = []

    def hot_factory(model, confidence):
        detector = _DummyDetector(value=0.5, critical=1.2)
        hot_instances.append(detector)
        return detector

    def q_factory(model, confidence):
        detector = _DummyDetector(value=0.7, critical=1.5)
        q_instances.append(detector)
        return detector

    monkeypatch.setattr(
        "chemotools.inspector.mixins._latent.HotellingT2",
        hot_factory,
    )
    monkeypatch.setattr(
        "chemotools.inspector.mixins._latent.QResiduals",
        q_factory,
    )

    fig = inspector.create_latent_distance_figure(
        dataset=["train", "test"],
        color_by_y=False,
        figsize=(4, 4),
    )

    assert fig is not None
    assert len(hot_instances) == 1
    assert len(q_instances) == 1
    hot = hot_instances[0]
    q_det = q_instances[0]
    assert len(hot.fit_calls) == 1
    assert len(q_det.fit_calls) == 1
    # Two datasets produce two predict calls each
    assert len(hot.predict_calls) == 2
    assert len(q_det.predict_calls) == 2
