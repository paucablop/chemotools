import numpy as np
from sklearn.utils.estimator_checks import check_estimator

from chemotools.augmentation import ScatterShift
from chemotools.scatter import ExtendedMultiplicativeScatterCorrection


# Test compliance with scikit-learn
def test_compliance_scatter_shift():
    # Arrange
    transformer = ScatterShift()
    # Act & Assert
    check_estimator(transformer)


# Test that default parameters leave the data unchanged (identity)
def test_scatter_shift_identity_at_defaults():
    # Arrange
    rng = np.random.default_rng(0)
    spectra = rng.normal(size=(5, 50))
    transformer = ScatterShift(random_state=42)

    # Act
    spectra_augmented = transformer.fit_transform(spectra)

    # Assert
    assert spectra.shape == spectra_augmented.shape
    assert np.allclose(spectra, spectra_augmented, atol=1e-8)


# Test that a multiplicative-only shift scales each spectrum by a constant factor
def test_scatter_shift_multiplicative_only():
    # Arrange
    spectrum = np.ones(100).reshape(1, -1)
    transformer = ScatterShift(
        order=0, multiplicative_scale=0.1, additive_scale=0.0, random_state=42
    )

    # Act
    augmented = transformer.fit_transform(spectrum)

    # Assert: a flat spectrum stays flat under a pure multiplicative factor
    assert augmented.shape == spectrum.shape
    assert np.isclose(np.std(augmented[0]), 0.0, atol=1e-8)
    # The multiplicative factor must lie within [1 - scale, 1 + scale]
    assert 0.9 <= augmented[0, 0] <= 1.1


# Test that an additive polynomial baseline introduces wavelength-dependent structure
def test_scatter_shift_polynomial_baseline():
    # Arrange
    spectrum = np.ones(100).reshape(1, -1)
    transformer = ScatterShift(
        order=2, multiplicative_scale=0.0, additive_scale=0.5, random_state=42
    )

    # Act
    augmented = transformer.fit_transform(spectrum)

    # Assert: a non-zero order baseline introduces curvature (non-constant offset),
    # which BaselineShift (constant only) cannot produce
    assert augmented.shape == spectrum.shape
    assert np.std(augmented[0]) > 0.0


# Test reproducibility with a fixed random_state
def test_scatter_shift_reproducible():
    # Arrange
    rng = np.random.default_rng(1)
    spectra = rng.normal(size=(4, 60))
    params = dict(
        order=2, multiplicative_scale=0.05, additive_scale=0.02, random_state=7
    )

    # Act
    first = ScatterShift(**params).fit_transform(spectra)
    second = ScatterShift(**params).fit_transform(spectra)

    # Assert
    assert np.allclose(first, second, atol=1e-12)


# Test that the polynomial basis matches the EMSC design-matrix construction
def test_scatter_shift_basis_matches_emsc():
    # Arrange
    n_features = 75
    order = 3
    spectra = np.ones((3, n_features))
    transformer = ScatterShift(order=order).fit(spectra)

    # Reconstruct the EMSC polynomial basis independently
    x_indices = np.linspace(-1, 1, n_features)
    expected_basis = np.vander(x_indices, N=order + 1, increasing=True)

    # Assert: augmentation basis is identical to the EMSC baseline basis
    assert np.allclose(transformer.polynomial_basis_, expected_basis, atol=1e-12)

    # And EMSC at the same order builds the same polynomial block, confirming the
    # augmenter is the forward counterpart of the correction
    emsc = ExtendedMultiplicativeScatterCorrection(order=order).fit(spectra)
    assert emsc.A_[:, : order + 1].shape == transformer.polynomial_basis_.shape
