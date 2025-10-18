chemotools.baseline
===================

.. currentmodule:: chemotools.baseline

Baseline correction methods for spectral preprocessing. These methods remove baseline drift and offset to isolate the chemical signal.

**Import from this module:**

.. code-block:: python

   from chemotools.baseline import (
       AirPls,
       ArPls,
       AsLs,
       ConstantBaselineCorrection,
       CubicSplineCorrection,
       LinearCorrection,
       NonNegative,
       PolynomialCorrection,
       SubtractReference,
   )

Available Classes
-----------------

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Class
     - Description
   * - :doc:`AirPls </methods/generated/chemotools.baseline.AirPls>`
     - Adaptive iteratively reweighted penalized least squares
   * - :doc:`ArPls </methods/generated/chemotools.baseline.ArPls>`
     - Asymmetrically reweighted penalized least squares
   * - :doc:`AsLs </methods/generated/chemotools.baseline.AsLs>`
     - Asymmetric least squares smoothing
   * - :doc:`ConstantBaselineCorrection </methods/generated/chemotools.baseline.ConstantBaselineCorrection>`
     - Subtract a constant baseline value
   * - :doc:`CubicSplineCorrection </methods/generated/chemotools.baseline.CubicSplineCorrection>`
     - Fit and subtract cubic spline baseline
   * - :doc:`LinearCorrection </methods/generated/chemotools.baseline.LinearCorrection>`
     - Linear baseline correction
   * - :doc:`NonNegative </methods/generated/chemotools.baseline.NonNegative>`
     - Enforce non-negative values
   * - :doc:`PolynomialCorrection </methods/generated/chemotools.baseline.PolynomialCorrection>`
     - Polynomial baseline fitting and subtraction
   * - :doc:`SubtractReference </methods/generated/chemotools.baseline.SubtractReference>`
     - Subtract a reference spectrum

See Also
--------

:doc:`Baseline Methods Overview </methods/baseline>` - Complete documentation with examples and visual guides
