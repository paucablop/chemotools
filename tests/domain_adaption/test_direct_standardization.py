"""
Test for DirectStandardization
"""

# Authors: Ruggero Guerrini
# License: MIT
import numpy as np
from chemotools.domain_adaption._direct_standardization import (
   DirectStandardization,
)


def data_diff(dataset_ref, dataset_test):
   diff_norm = np.linalg.norm(dataset_ref - dataset_test)
   ref_norm = np.linalg.norm(dataset_ref)
   difference = diff_norm / ref_norm
   return difference


class Test_Direct_Standardization:
   """
   Test that enhanced Direct Standardization maintains sklearn API compatibility
   """

   def test_shape_consistency_and_improvement(self):
      # Arrange - I create a slave linked to my mster
      np.random.seed(17)
      X_master = np.random.rand(100, 50)
      X_slave = 1.2 * X_master + 0.01 * np.random.randn(100, 50)

      # Fit model
      model = DirectStandardization().fit(X_slave, X_master)

      # Act
      X_transformed = model.transform(X_slave)

      # Assert
      assert X_transformed.shape == X_slave.shape
      assert X_transformed.shape == X_master.shape
      assert model.T.shape == (X_slave.shape[1], X_master.shape[1])
      # Test to verify that the difference is smaller with the transfer model
      before = data_diff(X_master, X_slave)
      after = data_diff(X_master, X_transformed)
      assert before > after
