import pytest

import numpy as np
from sklearn.utils.estimator_checks import check_estimator

from chemotools.scatter import StandardNormalVariate


# Test compliance with scikit-learn
def test_compliance_standard_normal_variate():
    # Arrange
    transformer = StandardNormalVariate()
    # Act & Assert
    check_estimator(transformer)


# Test functionality
def test_standard_normal_variate(spectrum, reference_snv):
    # Arrange
    snv = StandardNormalVariate()

    # Act
    spectrum_corrected = snv.fit_transform(spectrum)

    # Assert
    assert np.allclose(spectrum_corrected[0], reference_snv[0], atol=1e-2)


def test_snv_flat_signal_warns():
    # Arrange
    X = np.array([[5.0, 5.0, 5.0]])  # flat signal
    snv = StandardNormalVariate().fit(X)

    # Act & Assert
    with pytest.warns(
        UserWarning,
        match="Standard deviation is zero in SNV. This indicates a flat signal and will result in NaNs.",
    ):
        snv.transform(X)
