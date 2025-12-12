chemotools.plotting
===================

.. currentmodule:: chemotools.plotting

Plotting utilities for creating publication-quality visualizations of spectral data and model diagnostics. These classes follow a consistent API pattern and can be composed together for complex visualizations.

**Import from this module:**

.. code-block:: python

   from chemotools.plotting import (
       SpectraPlot,
       ScoresPlot,
       LoadingsPlot,
       DistancesPlot,
       ExplainedVariancePlot,
       FeatureSelectionPlot,
       PredictedVsActualPlot,
       YResidualsPlot,
       QQPlot,
       ResidualDistributionPlot,
   )

Available Classes
-----------------

**Spectral Visualization**

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Class
     - Description
   * - :doc:`SpectraPlot </methods/generated/chemotools.plotting.SpectraPlot>`
     - Plot spectral data with categorical or continuous coloring
   * - :doc:`FeatureSelectionPlot </methods/generated/chemotools.plotting.FeatureSelectionPlot>`
     - Visualize feature selection on spectral data

**Model Diagnostics**

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Class
     - Description
   * - :doc:`ScoresPlot </methods/generated/chemotools.plotting.ScoresPlot>`
     - Scatter plot of model scores (latent space projections)
   * - :doc:`LoadingsPlot </methods/generated/chemotools.plotting.LoadingsPlot>`
     - Line plot of model loadings (feature weights)
   * - :doc:`ExplainedVariancePlot </methods/generated/chemotools.plotting.ExplainedVariancePlot>`
     - Bar plot of explained variance by component
   * - :doc:`DistancesPlot </methods/generated/chemotools.plotting.DistancesPlot>`
     - Scatter plot for outlier detection (Q residuals, Hotelling's T²)

**Regression Diagnostics**

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Class
     - Description
   * - :doc:`PredictedVsActualPlot </methods/generated/chemotools.plotting.PredictedVsActualPlot>`
     - Scatter plot of predicted vs actual values
   * - :doc:`YResidualsPlot </methods/generated/chemotools.plotting.YResidualsPlot>`
     - Plot of Y residuals for homoscedasticity analysis
   * - :doc:`QQPlot </methods/generated/chemotools.plotting.QQPlot>`
     - Q-Q plot for assessing normality of residuals
   * - :doc:`ResidualDistributionPlot </methods/generated/chemotools.plotting.ResidualDistributionPlot>`
     - Histogram of residuals with normal distribution overlay

See Also
--------

:doc:`Plotting Methods Overview </methods/plotting>` - Complete documentation with examples and visual guides
