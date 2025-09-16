import numpy as np
from sklearn.utils.estimator_checks import check_estimator

from chemotools.baseline import AsLs


# Test compliance with scikit-learn
def test_compliance_as_ls():
    # Arrange
    transformer = AsLs()
    # Act & Assert
    check_estimator(transformer)


def test_compliance_as_ls_sparse():
    # Arrange
    transformer = AsLs(solver_type="sparse")
    # Act & Assert
    check_estimator(transformer)


# Test functionality
def test_as_ls_banded(spectrum, reference_asls):
    # Arrange
    as_ls = AsLs(solver_type="banded")

    # Act
    spectrum_corrected = as_ls.fit_transform(spectrum)

    # Assert
    assert np.allclose(spectrum_corrected[0], reference_asls[0], atol=1e-4)


def test_as_ls_sparse(spectrum, reference_asls):
    # Arrange
    as_ls = AsLs(solver_type="sparse")

    # Act
    spectrum_corrected = as_ls.fit_transform(spectrum)

    # Assert
    assert np.allclose(spectrum_corrected[0], reference_asls[0], atol=1e-4)
