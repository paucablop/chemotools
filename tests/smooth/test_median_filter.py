import numpy as np
from sklearn.utils.estimator_checks import check_estimator

from chemotools.smooth import MedianFilter


# Test compliance with scikit-learn
def test_compliance_median_filter():
    # Arrange
    transformer = MedianFilter()
    # Act & Assert
    check_estimator(transformer)


# Test functionality
def test_median_filter():
    # Arrange
    array = np.array([[1.0, 2.0, 30.0, 4.0, 5.0]])
    mean_filter = MedianFilter(window_size=3)

    # Act
    array_corrected = mean_filter.fit_transform(array)

    # Assert
    assert np.allclose(array_corrected[0], [1, 2.0, 4.0, 5.0, 5.0], atol=1e-8)
