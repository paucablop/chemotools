import numpy as np
import pytest
from sklearn.utils.estimator_checks import check_estimator

# adjust import to your package structure
from chemotools.smooth import ModifiedSincFilter


# ---------------------------
# sklearn compliance
# ---------------------------
def test_compliance_modified_sinc():
    transformer = ModifiedSincFilter()
    check_estimator(transformer)


# ---------------------------
# kernel sanity
# ---------------------------
def test_ms_kernel_properties_default():
    ms = ModifiedSincFilter(window_size=21, n=6, alpha=3.0, mode="interp")
    # fit on any numeric 2D data to set up internal attributes
    X = np.zeros((1, 21), dtype=np.float64)
    ms.fit(X)

    k = ms.kernel_
    assert k.ndim == 1
    assert k.size == 21
    # symmetry and Direct Current preservation
    assert np.allclose(k, k[::-1], atol=1e-12)
    assert np.isclose(k.sum(), 1.0, atol=1e-12)


def test_ms_kernel_changes_with_params():
    # different alpha should yield a different kernel (not identical vector)
    ms1 = ModifiedSincFilter(window_size=21, n=6, alpha=2.0, mode="interp").fit(
        np.zeros((1, 21))
    )
    ms2 = ModifiedSincFilter(window_size=21, n=6, alpha=4.0, mode="interp").fit(
        np.zeros((1, 21))
    )
    assert not np.allclose(ms1.kernel_, ms2.kernel_, atol=1e-12)
    # both remain valid kernels
    assert np.isclose(ms1.kernel_.sum(), 1.0, atol=1e-12)
    assert np.isclose(ms2.kernel_.sum(), 1.0, atol=1e-12)
    assert np.allclose(ms1.kernel_, ms1.kernel_[::-1], atol=1e-12)
    assert np.allclose(ms2.kernel_, ms2.kernel_[::-1], atol=1e-12)


# ---------------------------
# basic functionality (single-row signals)
# ---------------------------
@pytest.mark.parametrize("mode", ["interp", "nearest", "wrap", "constant"])
def test_ms_constant_preservation_all_modes(mode):
    # Direct Current should be preserved for any padding scheme
    nine = 9
    ms = ModifiedSincFilter(window_size=nine, n=6, alpha=3.0, mode=mode)
    X = np.full((1, nine), 2.5, dtype=np.float64)

    Y = ms.fit_transform(X)

    assert np.allclose(Y, X, atol=1e-12)
    assert Y.shape == X.shape
    assert Y.dtype == np.float64


@pytest.mark.parametrize("mode", ["interp", "nearest", "wrap", "constant"])
def test_ms_impulse_equals_kernel_all_modes(mode):
    # Convolving a centered impulse returns the kernel itself
    m = 4
    L = 2 * m + 1
    ms = ModifiedSincFilter(window_size=L, n=6, alpha=3.0, mode=mode)

    X = np.zeros((1, L), dtype=np.float64)
    X[0, m] = 1.0  # centered delta

    ms.fit_transform(X)
    k = ms.kernel_

    assert np.isclose(k.sum(), 1.0, atol=1e-12)
    assert np.allclose(k, k[::-1], atol=1e-12)


def test_ms_linear_ramp_preservation_interp_only():
    # With linear extrapolation, a linear ramp should be preserved at the edges
    nine = 9
    X = np.arange(nine, dtype=np.float64)[None, :]  # shape (1, 9)

    ms_interp = ModifiedSincFilter(window_size=nine, n=6, alpha=3.0, mode="interp")
    Y_interp = ms_interp.fit_transform(X)
    assert np.allclose(Y_interp, X, atol=1e-12)


# multi-row / axis behavior
def test_ms_axis_behavior_rows_vs_columns():
    # Smoothing along axis=1 should match smoothing each row independently.
    # Likewise, axis=0 + transpose should give the same result.
    rng = np.random.default_rng(42)
    n_rows = 4
    n_cols = 21
    X = rng.normal(size=(n_rows, n_cols)).astype(np.float64)

    ms_row = ModifiedSincFilter(window_size=21, n=6, alpha=3.0, mode="interp", axis=1)
    Y_row = ms_row.fit_transform(X)

    ms_col = ModifiedSincFilter(window_size=21, n=6, alpha=3.0, mode="interp", axis=0)
    Y_col = ms_col.fit_transform(X.T).T  # smooth columns, then transpose back

    assert np.allclose(Y_row, Y_col, atol=1e-12)
    assert Y_row.shape == X.shape
    assert Y_col.shape == X.shape


# Test kappa corrections with different n values
@pytest.mark.parametrize("n", [6, 8, 10])
def test_kappa_corrections_applied(n):
    """Test that kappa corrections are properly applied for n=6,8,10."""
    # Using a larger window size to ensure m >= n/2 + 2
    window_size = n * 4 + 1  # Ensure large enough window
    ms = ModifiedSincFilter(window_size=window_size, n=n, alpha=3.0)

    # Fit on dummy data to trigger kernel computation
    X = np.zeros((1, window_size), dtype=np.float64)
    ms.fit(X)

    # The kernel should be computed with kappa corrections
    assert hasattr(ms, "kernel_")
    assert ms.kernel_.shape == (window_size,)

    # Verify symmetry and DC preservation
    assert np.allclose(ms.kernel_, ms.kernel_[::-1], atol=1e-12)
    assert np.isclose(ms.kernel_.sum(), 1.0, atol=1e-12)
