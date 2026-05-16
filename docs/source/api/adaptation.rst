chemotools.adaptation
=====================

.. currentmodule:: chemotools.adaptation

Adaptation methods for calibration transfer between instruments. These transformers correct for differences in spectral axes or instrument response, allowing models trained on one instrument to remain valid on another.

**Import from this module:**

.. code-block:: python

   from chemotools.adaptation import (
       DirectStandardization,
       PiecewiseDirectStandardization,
       XAxisInterpolator,
   )

Available Classes
-----------------

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Class
     - Description
   * - :doc:`DirectStandardization </methods/generated/chemotools.adaptation.DirectStandardization>`
     - Linear calibration transfer via least-squares mapping between instruments
   * - :doc:`PiecewiseDirectStandardization </methods/generated/chemotools.adaptation.PiecewiseDirectStandardization>`
     - Piecewise calibration transfer using local least-squares windows
   * - :doc:`XAxisInterpolator </methods/generated/chemotools.adaptation.XAxisInterpolator>`
     - Resample spectra onto a shared x-axis grid via interpolation

See Also
--------

:doc:`Adaptation Methods Overview </methods/adaptation>` - Complete documentation with examples and visual guides
