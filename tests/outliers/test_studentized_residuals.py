import numpy as np
from sklearn.cross_decomposition import PLSRegression

from chemotools.outliers import StudentizedResiduals


# Test functionality
def test_studentized_residuals(dummy_data_loader):
    # Arrange
    X, y = dummy_data_loader
    pls = PLSRegression(n_components=1).fit(X, y)

    # Act
    stu_residuals = StudentizedResiduals(pls, confidence=0.95).fit(X, y)
    studentized_residuals = stu_residuals.predict_residuals(X, y)

    # Assert
    assert np.isclose(
        np.mean(studentized_residuals), 0, atol=0.001
    ), "Mean of studentized residuals should be 0"
    assert np.isclose(
        np.std(studentized_residuals), 1, atol=0.001
    ), "Standard deviation of studentized residuals should be 1"
    assert np.isclose(
        studentized_residuals[0], -1.16998195, atol=0.001
    ), "Studentized residual value mismatch"
    assert np.isclose(
        studentized_residuals[-1], 1.17827456, atol=0.001
    ), "Studentized residual value mismatch"
    assert studentized_residuals.shape == (100,), "Studentized residuals shape mismatch"
