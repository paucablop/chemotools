API Reference
=============

The public Python API for ``chemotools``. This section provides a technical reference for all importable modules and classes.

For detailed documentation with examples and visual guides, see the :doc:`Methods </methods/index>` section.

.. toctree::
   :maxdepth: 1
   :caption: Core Modules

   Augmentation <augmentation>
   Baseline <baseline>
   Derivative <derivative>
   Feature Selection <feature_selection>
   Outliers <outliers>
   Scale <scale>
   Scatter <scatter>
   Smooth <smooth>
   Utilities <utils>

Quick Import Reference
----------------------

All classes can be imported directly from their respective modules:

.. code-block:: python

   from chemotools.baseline import AirPls, ArPls
   from chemotools.augmentation import AddNoise, BaselineShift
   from chemotools.scale import MinMaxScaler, NormScaler
   from chemotools.outliers import HotellingT2, Leverage
   # ... and so on



