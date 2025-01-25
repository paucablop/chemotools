import numpy as np
import os
import pytest


test_directory = os.path.dirname(os.path.abspath(__file__))

path_to_resources = os.path.join(test_directory, "resources")


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
def reference_whitakker() -> np.ndarray:
    return np.loadtxt(
        os.path.join(path_to_resources, "reference_whitakker.csv"), delimiter=","
    ).reshape(1, -1)
