"""Tests for enhanced PCR."""

from chemotools.models._principal_component_regression import (
    PrincipalComponentRegression,
)
from sklearn.utils.estimator_checks import check_estimator


# Test compliance with scikit-learn
def test_compliance_pls_regression():
    # Arrange
    transformer = PrincipalComponentRegression()
    # Act & Assert
    check_estimator(transformer)
