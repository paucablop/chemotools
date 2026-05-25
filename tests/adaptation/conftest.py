"""Shared fixtures for adaptation tests."""

import numpy as np
import pytest


def _make_spectral_data(
    rng: np.random.Generator,
    n_samples: int,
    n_features: int = 50,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate paired (source, target) spectral data using a Beer-Lambert model.

    Spectra are formed as a linear mixture of Gaussian peaks, giving correlated
    features with a positive (non-zero) mean — like real absorbance data.  The
    target instrument adds a smooth multiplicative gain and additive offset so
    that centering bugs and window-structure issues are detectable, unlike the
    mean-zero i.i.d. data that would mask them.

    Parameters
    ----------
    rng : np.random.Generator
    n_samples : int
    n_features : int, default=50

    Returns
    -------
    X_source : ndarray of shape (n_samples, n_features)
    X_target : ndarray of shape (n_samples, n_features)
    C : ndarray of shape (n_samples, 4) — concentration matrix
    """
    n_components = 4
    wavenumbers = np.linspace(0, 1, n_features)

    # Smooth Gaussian spectral basis — correlated features, positive mean
    peaks = [0.15, 0.40, 0.65, 0.85]
    widths = [0.07, 0.08, 0.06, 0.07]
    basis = np.stack(
        [np.exp(-0.5 * ((wavenumbers - p) / w) ** 2) for p, w in zip(peaks, widths)]
    )  # (n_components, n_features)

    # Positive concentrations → spectra have positive mean (Beer-Lambert law)
    C = rng.uniform(0.05, 1.0, size=(n_samples, n_components))
    X_ref = C @ basis  # (n_samples, n_features)

    # Smooth instrument effects: multiplicative gain + additive baseline + noise
    gain = 1.0 + 0.08 * np.sin(np.linspace(0, np.pi, n_features))
    offset = 0.04 * (1.0 + np.cos(np.linspace(0, 2 * np.pi, n_features)))
    noise_std = 0.003

    X_source = X_ref + rng.normal(0, noise_std, size=X_ref.shape)
    X_target = X_ref * gain + offset + rng.normal(0, noise_std, size=X_ref.shape)

    return X_source, X_target, C


@pytest.fixture
def sample_data():
    """Paired spectral fixture shared by all adaptation transformer tests.

    Returns Beer-Lambert spectra (positive mean, smooth feature correlation)
    with a realistic instrument transfer: smooth multiplicative gain plus
    additive baseline offset plus noise.  The non-zero mean and correlated
    features make centering bugs and window-structure issues detectable, unlike
    mean-zero i.i.d. data.

    Returns
    -------
    tuple : (X_target, X_source)
        X_target : ndarray of shape (100, 50) — target instrument spectra
        X_source : ndarray of shape (100, 50) — source (reference) spectra
    """
    rng = np.random.default_rng(17)
    X_source, X_target, _ = _make_spectral_data(rng, n_samples=100)
    return X_target, X_source


@pytest.fixture
def labeled_sample_data():
    """Paired spectral fixture with concentration labels for end-to-end tests.

    Use this fixture to verify that a calibration transfer improves downstream
    prediction accuracy, not just spectral distance.  ``y`` is the concentration
    of the first spectral component, which has a known linear relationship to
    the spectra.

    Returns
    -------
    tuple : (X_target_train, X_source_train, y_train, X_target_test, y_test)
        All arrays have n_features=50.  Train split: 80 samples; test: 20 samples.
    """
    rng = np.random.default_rng(42)
    X_source, X_target, C = _make_spectral_data(rng, n_samples=100)
    y = C[:, 0]
    return X_target[:80], X_source[:80], y[:80], X_target[80:], y[80:]


def data_diff(dataset_ref, dataset_test) -> float:
    """Normalized Frobenius distance: ||ref - test|| / ||ref||."""
    return np.linalg.norm(dataset_ref - dataset_test) / np.linalg.norm(dataset_ref)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error."""
    return float(np.sqrt(np.mean((y_true - y_pred.ravel()) ** 2)))
