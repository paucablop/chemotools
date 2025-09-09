import numpy as np
from sklearn.utils.estimator_checks import check_estimator

from chemotools.smooth import WhittakerSmooth


# Test compliance with scikit-learn
def test_compliance_whittaker_smooth():
    # Arrange
    transformer = WhittakerSmooth()
    # Act & Assert
    check_estimator(transformer)


# Test functionality
def test_whittaker_smooth(spectrum, reference_whittaker):
    # Arrange
    whittaker_smooth = WhittakerSmooth()

    # Act
    spectrum_corrected = whittaker_smooth.fit_transform(spectrum)

    # Assert
    assert np.allclose(spectrum_corrected[0], reference_whittaker[0], atol=1e-8)
