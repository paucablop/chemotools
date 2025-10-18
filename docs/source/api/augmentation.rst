chemotools.augmentation
=======================

.. currentmodule:: chemotools.augmentation

Data augmentation methods for spectral analysis. These methods add controlled variation to your data to simulate different real-world scenarios and improve model robustness.

**Import from this module:**

.. code-block:: python

   from chemotools.augmentation import (
       AddNoise,
       BaselineShift,
       FractionalShift,
       GaussianBroadening,
       IndexShift,
       SpectrumScale,
   )

Available Classes
-----------------

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Class
     - Description
   * - :doc:`AddNoise </methods/generated/chemotools.augmentation.AddNoise>`
     - Add random noise to spectra
   * - :doc:`BaselineShift </methods/generated/chemotools.augmentation.BaselineShift>`
     - Shift baseline by a constant value
   * - :doc:`FractionalShift </methods/generated/chemotools.augmentation.FractionalShift>`
     - Apply fractional wavelength shifts
   * - :doc:`GaussianBroadening </methods/generated/chemotools.augmentation.GaussianBroadening>`
     - Broaden peaks with Gaussian kernel
   * - :doc:`IndexShift </methods/generated/chemotools.augmentation.IndexShift>`
     - Shift spectra by index positions
   * - :doc:`SpectrumScale </methods/generated/chemotools.augmentation.SpectrumScale>`
     - Scale spectrum intensity

See Also
--------

:doc:`Augmentation Methods Overview </methods/augmentation>` - Complete documentation with examples and visual guides
