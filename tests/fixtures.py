import numpy as np
import os
import pytest


test_directory = os.path.dirname(os.path.abspath(__file__))

path_to_resources = os.path.join(test_directory, "resources")


@pytest.fixture
def spectrum() -> np.ndarray:
    return [np.loadtxt(os.path.join(path_to_resources, "spectrum.csv"), delimiter=",").tolist()]


@pytest.fixture
def reference_airpls() -> np.ndarray:
    return [
        np.loadtxt(
            os.path.join(path_to_resources, "reference_airpls.csv"), delimiter=","
        ).tolist()
    ]


@pytest.fixture
def reference_whitakker() -> np.ndarray:
    return [
        np.loadtxt(
            os.path.join(path_to_resources, "reference_whitakker.csv"), delimiter=","
        ).tolist()
    ]
