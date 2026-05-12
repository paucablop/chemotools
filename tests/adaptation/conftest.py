"""Shared fixtures for adaptation tests."""

import numpy as np
import pytest


@pytest.fixture
def sample_data():
    """Generate sample data for adaptation tests.

    Returns:
        tuple: (X_target, X_source) where:
            - X_source is the source/reference instrument data (base)
            - X_target is the target instrument data to be calibrated (scaled version)
    """
    rng = np.random.default_rng(17)
    X_source = rng.normal(size=(100, 20))
    X_target = X_source * 2 - rng.normal(size=(100, 20)) * 0.02
    return X_target, X_source


def data_diff(dataset_ref, dataset_test):
    """Calculate normalized difference between two datasets.

    Args:
        dataset_ref: Reference dataset
        dataset_test: Test dataset

    Returns:
        float: Normalized difference (||ref - test|| / ||ref||)
    """
    diff_norm = np.linalg.norm(dataset_ref - dataset_test)
    ref_norm = np.linalg.norm(dataset_ref)
    difference = diff_norm / ref_norm
    return difference
