chemotools.inspector
====================

.. currentmodule:: chemotools.inspector

Inspector classes for comprehensive model diagnostics and visualization. Inspectors provide a unified interface for creating multiple diagnostic plots for PCA and PLS models.

.. note::

   The inspector module is experimental and under active development. The API may change in future versions.

**Import from this module:**

.. code-block:: python

   from chemotools.inspector import (
       PCAInspector,
       PLSRegressionInspector,
   )

Available Classes
-----------------

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Class
     - Description
   * - :doc:`PCAInspector </methods/generated/chemotools.inspector.PCAInspector>`
     - Inspector for PCA model diagnostics (scores, loadings, variance, outliers)
   * - :doc:`PLSRegressionInspector </methods/generated/chemotools.inspector.PLSRegressionInspector>`
     - Inspector for PLS regression diagnostics (includes regression metrics)

See Also
--------

:doc:`Inspector Methods Overview </methods/inspector>` - Complete documentation with examples and visual guides
