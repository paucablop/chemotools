chemotools.feature_selection
=============================

.. currentmodule:: chemotools.feature_selection

Feature selection methods to identify the most chemically relevant wavelengths or variables for your models.

**Import from this module:**

.. code-block:: python

   from chemotools.feature_selection import (
       IndexSelector,
       RangeCut,
       SRSelector,
       VIPSelector,
   )

Available Classes
-----------------

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Class
     - Description
   * - :doc:`IndexSelector </methods/generated/chemotools.feature_selection.IndexSelector>`
     - Select features by specific indices
   * - :doc:`RangeCut </methods/generated/chemotools.feature_selection.RangeCut>`
     - Select features within a specified range
   * - :doc:`SRSelector </methods/generated/chemotools.feature_selection.SRSelector>`
     - Selectivity ratio based feature selection
   * - :doc:`VIPSelector </methods/generated/chemotools.feature_selection.VIPSelector>`
     - Variable importance in projection (VIP) selector

See Also
--------

:doc:`Feature Selection Methods Overview </methods/feature_selection>` - Complete documentation with examples and visual guides
