"""
Test for SpectralSpaceTransform
"""

# Authors: Ruggero Guerrini
# License: MIT

from sklearn.utils.estimator_checks import check_estimator

from chemotools.classification import DDSIMCA

class TestSklearnCompliance:

    """Tests for sklearn estimator API compliance."""

    def test_compliance_DDSIMCA(self):
        
        """Verifies that DDSIMCA passes all sklearn estimator
        checks."""
        # Arrange
        transformer = DDSIMCA()
        # Act & Assert
        check_estimator(transformer)