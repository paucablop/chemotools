chemotools.scale
================

.. currentmodule:: chemotools.scale

Scaling methods to normalize spectral intensity and improve model performance.

**Import from this module:**

.. code-block:: python

   from chemotools.scale import (
       BandScaler,
       MinMaxScaler,
       NormScaler,
       ParetoScaler,
       PointScaler,
   )

Available Classes
-----------------

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Class
     - Description
   * - :doc:`BandScaler </methods/generated/chemotools.scale.BandScaler>`
     - Scale by the average intensity or area of a spectral band
   * - :doc:`MinMaxScaler </methods/generated/chemotools.scale.MinMaxScaler>`
     - Scale features to a given range (min-max normalization)
   * - :doc:`NormScaler </methods/generated/chemotools.scale.NormScaler>`
     - Scale by vector norm (L1, L2, or max)
   * - :doc:`ParetoScaler </methods/generated/chemotools.scale.ParetoScaler>`
     - Scale by a generalized power of the standard deviation (Pareto scaling)
   * - :doc:`PointScaler </methods/generated/chemotools.scale.PointScaler>`
     - Scale relative to a specific point in the spectrum

See Also
--------

:doc:`Scale Methods Overview </methods/scale>` - Complete documentation with examples and visual guides
