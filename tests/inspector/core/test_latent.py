import numpy as np
from sklearn.decomposition import PCA

from chemotools.inspector.core.latent import LatentVariableMixin


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

        # Create a valid fitted model for HotellingT2/QResiduals
        self.model = PCA(n_components=2)
        X_train = np.random.rand(10, 3)
        self.model.fit(X_train)

        X_test = np.random.rand(5, 3)

        self._scores = {
            "train": self.model.transform(X_train),
            "test": self.model.transform(X_test),
        }
        self._explained = self.model.explained_variance_ratio_
        self._raw = {
            "train": (X_train, np.zeros(10)),
            "test": (X_test, np.zeros(5)),
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
    """Test latent scores figure creation for a single dataset."""
    # Arrange
    inspector = DummyLatentInspector()

    # Act
    figures = inspector.create_latent_scores_figures(
        dataset="train",
        components=(0, 1),
        color_by="y",
        annotate_by="sample_index",
        figsize=(4, 4),
    )

    # Assert
    assert set(figures.keys()) == {"scores_1"}
    fig = figures["scores_1"]
    ax = fig.axes[0]
    assert ax.get_xlabel().startswith("LV1")

    # Cleanup
    import matplotlib.pyplot as plt

    for fig in figures.values():
        plt.close(fig)


def test_create_latent_scores_multi_dataset():
    """Test latent scores figure creation for multiple datasets with confidence ellipse."""
    # Arrange
    inspector = DummyLatentInspector()

    # Act
    figures = inspector.create_latent_scores_figures(
        dataset=["train", "test"],
        components=((0, 1),),
        color_by=None,
        annotate_by=None,
        figsize=(4, 4),
    )

    # Assert
    # Multi-dataset mode only creates combined plots, not individual dataset plots
    assert "scores_1" in figures
    assert (
        "scores_1_train" not in figures
    )  # Individual plots not created in multi-dataset mode
    assert (
        "scores_1_test" not in figures
    )  # Individual plots not created in multi-dataset mode
    multi_ax = figures["scores_1"].axes[0]
    # Confidence ellipse from training data leaves at least one patch
    assert len(multi_ax.patches) >= 1

    # Cleanup
    import matplotlib.pyplot as plt

    for fig in figures.values():
        plt.close(fig)


def test_create_latent_distance_runs_with_monkeypatched_detectors(monkeypatch):
    """Test latent distance figure with mocked Hotelling T2 and Q-residuals detectors."""
    # Arrange
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
        "chemotools.inspector.core.latent.HotellingT2",
        hot_factory,
    )
    monkeypatch.setattr(
        "chemotools.inspector.core.latent.QResiduals",
        q_factory,
    )

    # Act
    fig = inspector.create_latent_distance_figure(
        dataset=["train", "test"],
        color_by=None,
        figsize=(4, 4),
    )

    # Assert
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

    # Cleanup
    import matplotlib.pyplot as plt

    plt.close(fig)


def test_latent_summary():
    """Test latent summary generation."""
    # Arrange
    inspector = DummyLatentInspector()
    # Mock the n_components_ attribute that would normally come from _BaseInspector
    inspector.n_components_ = 2

    # Act
    summary = inspector.latent_summary()

    # Assert
    from chemotools.inspector.core.latent import LatentSummary

    assert isinstance(summary, LatentSummary)
    assert summary.n_components == 2
    assert isinstance(summary.hotelling_t2_limit, float)
    assert isinstance(summary.q_residuals_limit, float)
