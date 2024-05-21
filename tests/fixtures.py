### Imports ###

import os
from typing import List

import numpy as np
import pytest

from tests.test_for_utils.utils_models import (
    NoiseEstimationReference,
    RefDifferenceKernel,
)

### Constants ###

test_directory = os.path.dirname(os.path.abspath(__file__))

path_to_resources = os.path.join(test_directory, "resources")

### Fixtures ###


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

    return spectral_data[::, 2]


@pytest.fixture
def noise_level_whittaker_auto_lambda() -> np.ndarray:
    spectral_data = np.loadtxt(
        os.path.join(path_to_resources, "spectrum_whittaker_auto_lambda.csv"),
        delimiter=",",
        skiprows=1,
    )

    return spectral_data[::, 3]


@pytest.fixture
def reference_finite_differences(kind: str) -> List[RefDifferenceKernel]:
    fpath = os.path.join(
        path_to_resources,
        f"./finite_differences/reference_{kind}_differences.csv",
    )
    fin_diff_table = np.genfromtxt(
        fpath,
        skip_header=2,
        delimiter=",",
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
            RefDifferenceKernel(
                differences=round(row[0]),
                accuracy=round(row[1]),
                kernel=row[2:][~np.isnan(row[2:])],
            )
        )

    return fin_diff_ordered_coeffs


@pytest.fixture
def noise_level_estimation_signal() -> np.ndarray:
    fpath = os.path.join(
        path_to_resources,
        "noise_level_estimation/noise_estimation_refs.csv",
    )
    data = np.genfromtxt(
        fpath,
        delimiter=",",
        skip_header=1,
        filling_values=np.nan,
        dtype=np.float64,
    )

    # the original signal is indicated by the first 4 columns with metadata being NaN
    metadata = data[::, 0:4]
    signal_idx = np.where(np.isnan(metadata).all(axis=1))[0][0]

    return data[signal_idx, 4:]


@pytest.fixture
def noise_level_estimation_refs() -> List[NoiseEstimationReference]:
    fpath = os.path.join(
        path_to_resources,
        "noise_level_estimation/noise_estimation_refs.csv",
    )
    data = np.genfromtxt(
        fpath,
        delimiter=",",
        skip_header=1,
        filling_values=np.nan,
        dtype=np.float64,
    )

    # the original signal is indicated by the first 4 columns with metadata being NaN
    # it has to be excluded from the references
    metadata = data[::, 0:4]
    signal_idx = np.where(np.isnan(metadata).all(axis=1))[0][0]
    data = np.delete(data, obj=signal_idx, axis=0)

    # then, all the references are extracted
    noise_level_refs = []
    for row_idx in range(0, data.shape[0]):
        row = data[row_idx, ::]
        # if the window size is 0, it is set to None because this indicates that the
        # global noise level is to be estimated rather than a local one
        window_size = int(row[0])
        window_size = window_size if window_size > 0 else None
        noise_level_refs.append(
            NoiseEstimationReference(
                window_size=window_size,
                min_noise_level=row[1],
                differences=round(row[2]),
                accuracy=round(row[3]),
                noise_level=row[4:],
            )
        )

    return noise_level_refs
