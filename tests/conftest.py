import os
from typing import Tuple

import numpy as np
import pytest

from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


from chemotools.outliers._base import _ModelResidualsBase


test_directory = os.path.dirname(os.path.abspath(__file__))

path_to_resources = os.path.join(test_directory, "resources")


class _DummyModelResiduals(_ModelResidualsBase):
    """Dummy class to test _ModelResidualsBase"""

    def __init__(self, model, confidence):
        super().__init__(model, confidence)

    def predict_residuals(self):
        return np.zeros(10)

    def _calculate_critical_value(self):
        return 1.96


@pytest.fixture
def dummy_data_loader(
    scale: float = 0.2, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    np.random.seed(seed)
    x1 = np.linspace(0, 100, 100) + np.random.normal(size=100) * scale
    x2 = np.linspace(0, 200, 100) + np.random.normal(size=100) * scale
    x3 = np.linspace(0, 300, 100) + np.random.normal(size=100) * scale
    X = np.column_stack((x1, x2, x3))
    y = np.sum(X, axis=1) + np.random.normal(size=100) * scale
    return X, y


@pytest.fixture
def fitted_pca(dummy_data_loader):
    X, _ = dummy_data_loader
    return PCA(n_components=2).fit(X)


@pytest.fixture
def unfitted_pca():
    return PCA(n_components=2)


@pytest.fixture
def fitted_pls(dummy_data_loader):
    X, y = dummy_data_loader
    return PLSRegression(n_components=2).fit(X, y)


@pytest.fixture
def unfitted_pls():
    return PLSRegression(n_components=2)


@pytest.fixture
def fitted_pipeline_pca(dummy_data_loader):
    X, _ = dummy_data_loader
    return make_pipeline(StandardScaler(), PCA(n_components=2)).fit(X)


@pytest.fixture
def unfitted_pipeline():
    return make_pipeline(StandardScaler(with_std=False), PCA(n_components=2))


@pytest.fixture
def invalid_pipeline(dummy_data_loader):
    X, _ = dummy_data_loader
    return make_pipeline(StandardScaler(), StandardScaler())


@pytest.fixture
def invalid_model():
    return SVR()


@pytest.fixture
def pipeline_with_invalid_model():
    return make_pipeline(StandardScaler(), SVR())


@pytest.fixture
def spectrum() -> np.ndarray:
    return np.loadtxt(
        os.path.join(path_to_resources, "spectrum.csv"), delimiter=","
    ).reshape(1, -1)


@pytest.fixture
def spectrum_arpls() -> np.ndarray:
    return np.loadtxt(
        os.path.join(path_to_resources, "spectrum_arpls.csv"), delimiter=","
    ).reshape(1, -1)


@pytest.fixture
def reference_airpls() -> np.ndarray:
    return np.loadtxt(
        os.path.join(path_to_resources, "reference_airpls.csv"), delimiter=","
    ).reshape(1, -1)


@pytest.fixture
def reference_arpls() -> np.ndarray:
    return np.loadtxt(
        os.path.join(path_to_resources, "reference_arpls.csv"), delimiter=","
    ).reshape(1, -1)


@pytest.fixture
def reference_asls() -> np.ndarray:
    return np.loadtxt(
        os.path.join(path_to_resources, "reference_asls.csv"), delimiter=","
    ).reshape(1, -1)


@pytest.fixture
def reference_msc_mean() -> np.ndarray:
    return np.loadtxt(
        os.path.join(path_to_resources, "reference_msc_mean.csv"), delimiter=","
    ).reshape(1, -1)


@pytest.fixture
def reference_msc_median() -> np.ndarray:
    return np.loadtxt(
        os.path.join(path_to_resources, "reference_msc_median.csv"), delimiter=","
    ).reshape(1, -1)


@pytest.fixture
def reference_sg_15_2() -> np.ndarray:
    return np.loadtxt(
        os.path.join(path_to_resources, "reference_sg_15_2.csv"), delimiter=","
    ).reshape(1, -1)


@pytest.fixture
def reference_snv() -> np.ndarray:
    return np.loadtxt(
        os.path.join(path_to_resources, "reference_snv.csv"), delimiter=","
    ).reshape(1, -1)


@pytest.fixture
def reference_whittaker() -> np.ndarray:
    return np.loadtxt(
        os.path.join(path_to_resources, "reference_whittaker.csv"), delimiter=","
    ).reshape(1, -1)
