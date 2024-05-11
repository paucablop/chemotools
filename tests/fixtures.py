import os
from typing import List

import numpy as np
import pytest

test_directory = os.path.dirname(os.path.abspath(__file__))

path_to_resources = os.path.join(test_directory, "resources")


@pytest.fixture
def spectrum() -> List[np.ndarray]:
    return [
        np.loadtxt(
            os.path.join(path_to_resources, "spectrum.csv"),
            delimiter=",",
        )
    ]


@pytest.fixture
def spectrum_arpls() -> List[np.ndarray]:
    return [
        np.loadtxt(
            os.path.join(path_to_resources, "spectrum_arpls.csv"),
            delimiter=",",
        )
    ]


@pytest.fixture
def reference_airpls() -> List[np.ndarray]:
    return [
        np.loadtxt(
            os.path.join(path_to_resources, "reference_airpls.csv"),
            delimiter=",",
        )
    ]


@pytest.fixture
def reference_arpls() -> List[np.ndarray]:
    return [
        np.loadtxt(
            os.path.join(path_to_resources, "reference_arpls.csv"),
            delimiter=",",
        )
    ]


@pytest.fixture
def reference_msc_mean() -> List[np.ndarray]:
    return [
        np.loadtxt(
            os.path.join(path_to_resources, "reference_msc_mean.csv"),
            delimiter=",",
        )
    ]


@pytest.fixture
def reference_msc_median() -> List[np.ndarray]:
    return [
        np.loadtxt(
            os.path.join(path_to_resources, "reference_msc_median.csv"),
            delimiter=",",
        )
    ]


@pytest.fixture
def reference_sg_15_2() -> List[np.ndarray]:
    return [
        np.loadtxt(
            os.path.join(path_to_resources, "reference_sg_15_2.csv"),
            delimiter=",",
        )
    ]


@pytest.fixture
def reference_snv() -> List[np.ndarray]:
    return [
        np.loadtxt(
            os.path.join(path_to_resources, "reference_snv.csv"),
            delimiter=",",
        )
    ]


@pytest.fixture
def reference_whittaker() -> List[np.ndarray]:
    return [
        np.loadtxt(
            os.path.join(path_to_resources, "reference_whittaker.csv"),
            delimiter=",",
        )
    ]


@pytest.fixture
def spectrum_whittaker_auto_lambda() -> np.ndarray:
    spectral_data = np.loadtxt(
        os.path.join(path_to_resources, "spectrum_whittaker_auto_lambda.csv"),
        delimiter=",",
        skiprows=1,
    )

    return spectral_data[::, 1]


@pytest.fixture
def noise_level_whittaker_auto_lambda() -> np.ndarray:
    spectral_data = np.loadtxt(
        os.path.join(path_to_resources, "spectrum_whittaker_auto_lambda.csv"),
        delimiter=",",
        skiprows=1,
    )

    return spectral_data[::, 2]


@pytest.fixture
def reference_finite_differences() -> List[tuple[int, int, np.ndarray]]:
    fin_diff_table = np.genfromtxt(
        os.path.join(path_to_resources, "reference_finite_differences.csv"),
        skip_header=2,
        delimiter=",",
        missing_values="#N/A",
        filling_values=np.nan,
        dtype=np.float64,
    )
    fin_diff_ordered_coeffs = []
    for row_idx in range(0, fin_diff_table.shape[0]):
        # the first column is the difference order, the second column is the accuracy,
        # and the remaining columns are the coefficients where the trailing NaNs are
        # removed
        row = fin_diff_table[row_idx, ::]
        fin_diff_ordered_coeffs.append(
            (
                int(row[0]),
                int(row[1]),
                row[2:][~np.isnan(row[2:])],
            )
        )

    return fin_diff_ordered_coeffs
