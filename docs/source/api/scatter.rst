chemotools.scatter
==================

.. currentmodule:: chemotools.scatter

Scatter correction methods to remove multiplicative scatter effects and physical variations in spectral data.

**Import from this module:**

.. code-block:: python

   from chemotools.scatter import (
       ExtendedMultiplicativeScatterCorrection,
       MultiplicativeScatterCorrection,
       RobustNormalVariate,
       StandardNormalVariate,
   )

Available Classes
-----------------

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Class
     - Description
   * - :doc:`ExtendedMultiplicativeScatterCorrection </methods/generated/chemotools.scatter.ExtendedMultiplicativeScatterCorrection>`
     - EMSC with polynomial baseline correction
   * - :doc:`MultiplicativeScatterCorrection </methods/generated/chemotools.scatter.MultiplicativeScatterCorrection>`
     - Classic MSC algorithm
   * - :doc:`RobustNormalVariate </methods/generated/chemotools.scatter.RobustNormalVariate>`
     - RNV using median and MAD for robustness
   * - :doc:`StandardNormalVariate </methods/generated/chemotools.scatter.StandardNormalVariate>`
     - SNV using mean and standard deviation

See Also
--------

:doc:`Scatter Correction Methods Overview </methods/scatter>` - Complete documentation with examples and visual guides
