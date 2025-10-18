**Explore our datasets**
==============================

Welcome to the world of data exploration! Our ``chemotools`` package provides useful datasets 
that help you test the package and learn. You can find these datasets in the ``chemotools.datasets`` 
module and access them using simple loading functions. Here's what we offer:

The Fermentation Dataset 🧪
-------------------------------------

This dataset contains spectra collected during a yeast fermentation process using attenuated total 
reflectance Fourier transform infrared spectroscopy (ATR-FTIR). The dataset includes both a 
training set and a test set.

For more information about the Fermentation Dataset, see these publications:

- Cabaneros Lopez, P., Abeykoon Udugama, I., Thomsen, S.T., et al. `Transforming data to information: A parallel hybrid model for real-time state estimation in lignocellulosic ethanol fermentation <https://doi.org/10.1002/bit.27586>`_.

- Cabaneros Lopez, P., Abeykoon Udugama, I., Thomsen, S.T., et al. `Towards a digital twin: a hybrid data-driven and mechanistic digital shadow to forecast the evolution of lignocellulosic fermentation <https://doi.org/10.1002/bbb.2108>`_.

- Cabaneros Lopez, P., Abeykoon Udugama, I., Thomsen, S.T., et al. `Promoting the co-utilisation of glucose and xylose in lignocellulosic ethanol fermentations using a data-driven feed-back controller <https://doi.org/10.1186/s13068-020-01829-2>`_.


The Train Set
~~~~~~~~~~~~~~~~

The train set contains 21 synthetic spectra with reference glucose concentrations, measured by high-performance 
liquid chromatography (HPLC). You can load the train set as a ``pandas.DataFrame`` 
or as a ``polars.DataFrame``:

**Load as pandas.DataFrame**:

.. code-block:: python

   from chemotools.datasets import load_fermentation_train

   X_train, y_train = load_fermentation_train()

**Load as polars.DataFrame**:

.. code-block:: python

   from chemotools.datasets import load_fermentation_train

   X_train, y_train = load_fermentation_train(set_output="polars")

.. note::
   Polars is supported in ``chemotools``>=0.1.5

.. note::
   To learn how to build a PLS model using the Fermentation Dataset, see our `Training Guide <https://chemotools.org/learn/pls_regression.html>`__.

The Test Set
~~~~~~~~~~~~~~~

The test set contains over 1000 spectra collected during a fermentation process. These spectra were 
captured every 1.25 minutes over several hours. It also includes 35 reference glucose concentrations 
measured hourly during the fermentation.

Load the test set using:

**Load as pandas.DataFrame**:

.. code-block:: python

   from chemotools.datasets import load_fermentation_test

   X_test, y_test = load_fermentation_test()

**Load as polars.DataFrame**:

.. code-block:: python

   from chemotools.datasets import load_fermentation_test

   X_test, y_test = load_fermentation_test(set_output="polars")

.. note::
   The wavenumbers are stored as column names in both the ``pandas.DataFrame`` and the ``polars.DataFrame``.
   In a ``pandas.DataFrame`` the column names can be of type ``float``, but in a ``polars.DataFrame`` the column 
   names must be of type ``str``.

The Coffee Dataset ☕
-------------------------------

The Coffee Dataset contains spectra collected from various coffee samples from different countries. 
These spectra were collected using attenuated total reflectance Fourier transform infrared 
spectroscopy (ATR-FTIR).

**Load as pandas.DataFrame**:

.. code-block:: python

   from chemotools.datasets import load_coffee

   spectra, labels = load_coffee()

**Load as polars.DataFrame**:

.. code-block:: python

   from chemotools.datasets import load_coffee

   spectra, labels = load_coffee(set_output="polars")

.. note::
   To learn how to build a PLS-DA classification model using the Coffee Dataset, 
   see our `Training Guide <https://chemotools.org/learn/pls_classification.html>`__.

We hope you enjoy exploring these datasets! 🚀