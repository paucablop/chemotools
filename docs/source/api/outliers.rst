chemotools.outliers
===================

.. currentmodule:: chemotools.outliers

Outlier detection methods to identify unusual samples and improve model diagnostics.

**Import from this module:**

.. code-block:: python

   from chemotools.outliers import (
       DModX,
       HotellingT2,
       Leverage,
       QResiduals,
       StudentizedResiduals,
   )

Available Classes
-----------------

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Class
     - Description
   * - :doc:`DModX </methods/generated/chemotools.outliers.DModX>`
     - Distance to model in X-space
   * - :doc:`HotellingT2 </methods/generated/chemotools.outliers.HotellingT2>`
     - Hotelling's T² statistic for multivariate outliers
   * - :doc:`Leverage </methods/generated/chemotools.outliers.Leverage>`
     - Leverage values for influential point detection
   * - :doc:`QResiduals </methods/generated/chemotools.outliers.QResiduals>`
     - Q residuals (squared prediction error)
   * - :doc:`StudentizedResiduals </methods/generated/chemotools.outliers.StudentizedResiduals>`
     - Studentized residuals for regression outliers

See Also
--------

:doc:`Outlier Detection Methods Overview </methods/outliers>` - Complete documentation with examples and visual guides
