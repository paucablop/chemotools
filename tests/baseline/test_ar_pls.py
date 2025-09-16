import numpy as np
from sklearn.utils.estimator_checks import check_estimator

from chemotools.baseline import ArPls


# Test compliance with scikit-learn
def test_compliance_ar_pls():
    # Arrange
    transformer = ArPls()
    # Act & Assert
    check_estimator(transformer)


def test_compliance_ar_pls_sparse():
    # Arrange
    transformer = ArPls(solver_type="sparse")
    # Act & Assert
    check_estimator(transformer)


# Test functionality
def test_ar_pls_banded(spectrum_arpls, reference_arpls):
    # Arrange
    arpls = ArPls(1e2, 0.0001, solver_type="banded")
    reference = np.array(spectrum_arpls) - np.array(reference_arpls)

    # Act
    spectrum_corrected = arpls.fit_transform(spectrum_arpls)

    # Assert
    assert np.allclose(spectrum_corrected[0], reference[0], atol=1e-4)


def test_ar_pls_sparse(spectrum_arpls, reference_arpls):
    # Arrange
    arpls = ArPls(1e2, 0.0001, solver_type="sparse")
    reference = np.array(spectrum_arpls) - np.array(reference_arpls)

    # Act
    spectrum_corrected = arpls.fit_transform(spectrum_arpls)

    # Assert
    assert np.allclose(spectrum_corrected[0], reference[0], atol=1e-4)
