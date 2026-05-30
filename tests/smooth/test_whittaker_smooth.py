import numpy as np
from sklearn.utils.estimator_checks import check_estimator

from chemotools.smooth import WhittakerSmooth


# Test compliance with scikit-learn
def test_compliance_whittaker_smooth():
    # Arrange
    transformer = WhittakerSmooth()
    # Act & Assert
    check_estimator(transformer)


def test_compliance_whittaker_smooth_sparse():
    # Arrange
    transformer = WhittakerSmooth(solver_type="sparse")
    # Act & Assert
    check_estimator(transformer)


# Test functionality
def test_whittaker_smooth_banded(spectrum, reference_whittaker):
    # Arrange
    whittaker_smooth = WhittakerSmooth()

    # Act
    spectrum_corrected = whittaker_smooth.fit_transform(spectrum)

    # Assert
    assert np.allclose(spectrum_corrected[0], reference_whittaker[0], atol=1e-8)


def test_whittaker_smooth_sparse(spectrum, reference_whittaker):
    # Arrange
    whittaker_smooth = WhittakerSmooth(solver_type="sparse")

    # Act
    spectrum_corrected = whittaker_smooth.fit_transform(spectrum)

    # Assert
    assert np.allclose(spectrum_corrected[0], reference_whittaker[0], atol=1e-8)


def test_whittaker_smooth_banded_multi_row_matches_single_row():
    # Arrange: stack multiple independent copies of the same spectrum with distinct
    # row values so that a row/column transposition would produce wrong results
    rng = np.random.default_rng(42)
    n_samples, n_features = 20, 200
    X = rng.normal(size=(n_samples, n_features))

    ws = WhittakerSmooth()
    ws.fit(X)

    # Act: transform all rows at once (batch path)
    X_batch = ws.transform(X)

    # Reference: transform each row independently (single-row path)
    X_sequential = np.vstack([ws.transform(X[[i]]) for i in range(n_samples)])

    # Assert: every row must match the single-row result
    assert np.allclose(X_batch, X_sequential, atol=1e-10)
