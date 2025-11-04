chemotools.smooth
=================

.. currentmodule:: chemotools.smooth

Smoothing methods to reduce noise while preserving spectral features.

**Import from this module:**

.. code-block:: python

   from chemotools.smooth import (
       MeanFilter,
       MedianFilter,
       ModifiedSincFilter,
       SavitzkyGolayFilter,
       WhittakerSmooth,
   )

Available Classes
-----------------

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Class
     - Description
   * - :doc:`MeanFilter </methods/generated/chemotools.smooth.MeanFilter>`
     - Moving average smoothing
   * - :doc:`MedianFilter </methods/generated/chemotools.smooth.MedianFilter>`
     - Median filter for noise reduction
   * - :doc:`ModifiedSincFilter </methods/generated/chemotools.smooth.ModifiedSincFilter>`
     - Modified Sinc smoothing filter
   * - :doc:`SavitzkyGolayFilter </methods/generated/chemotools.smooth.SavitzkyGolayFilter>`
     - Savitzky-Golay smoothing filter
   * - :doc:`WhittakerSmooth </methods/generated/chemotools.smooth.WhittakerSmooth>`
     - Whittaker smoothing with penalized least squares

See Also
--------

:doc:`Smoothing Methods Overview </methods/smooth>` - Complete documentation with examples and visual guides
