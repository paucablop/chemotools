chemotools.adaptation
=====================

.. currentmodule:: chemotools.adaptation

Adaptation methods for calibration transfer between instruments, together with
metadata-aware transformation and validation utilities.

**Import from this module:**

.. code-block:: python

   from chemotools.adaptation import (
       DirectStandardization,
       MetadataFunctionTransformer,
       PiecewiseDirectStandardization,
       SpectralSpaceTransform,
       XAxisInterpolator,
   )

   from chemotools.adaptation.functions import subtract_reference
   from chemotools.adaptation.validation import check_metadata_function

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
   * - :doc:`SpectralSpaceTransform </methods/generated/chemotools.adaptation.SpectralSpaceTransform>`
     - SVD-based shared latent space domain adaptation
   * - :doc:`XAxisInterpolator </methods/generated/chemotools.adaptation.XAxisInterpolator>`
     - Resample spectra onto a shared x-axis grid via interpolation
   * - :doc:`MetadataFunctionTransformer </methods/generated/chemotools.adaptation.MetadataFunctionTransformer>`
     - Apply a callable with auxiliary inputs supplied through metadata routing

Available Functions
-------------------

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Function
     - Description
   * - :doc:`check_metadata_function </methods/generated/chemotools.adaptation.validation.check_metadata_function>`
     - Validate a metadata-aware callable on representative data
   * - :doc:`subtract_reference </methods/generated/chemotools.adaptation.functions.subtract_reference>`
     - Subtract a shared or per-sample reference from spectra
   * - :doc:`divide_by_reference </methods/generated/chemotools.adaptation.functions.divide_by_reference>`
     - Divide spectra by a shared or per-sample reference
   * - :doc:`scale_by_factor </methods/generated/chemotools.adaptation.functions.scale_by_factor>`
     - Scale spectra by a scalar or array factor
   * - :doc:`add_offset </methods/generated/chemotools.adaptation.functions.add_offset>`
     - Add a shared or per-sample offset to spectra

See Also
--------

:doc:`Adaptation Methods Overview </methods/adaptation>` - Complete documentation with examples and visual guides
