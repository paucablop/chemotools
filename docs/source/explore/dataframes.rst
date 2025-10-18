Working with DataFrames
=======================

For the ``pandas.DataFrame`` and ``polars.DataFrame`` lovers. By default, all ``scikit-learn`` and ``chemotools`` transformers output ``numpy.ndarray``. However, now it is possible to configure your ``chemotools`` preprocessing methods to produce either a ``pandas.DataFrame`` or a ``polars.DataFrame`` objects as output. This is possible after implementing the new ``set_output()`` API from ``scikit-learn`` (>= 1.2.2 for ``pandas`` and >= 1.4.0 for ``polars``) (`documentation <https://scikit-learn.org/stable/auto_examples/miscellaneous/plot_set_output.html>`_). The same API implemented in other ``scikit-learn`` preprocessing methods like the ``StandardScaler()`` is now available for the ``chemotools`` transformers.

.. note::
    From version 0.1.3, the ``set_output()`` is available for all ``chemotools`` functions!

Below there are two examples of how to use this new API:

**Example 1: Using the set_output() API with a single preprocessing method**
---------------------------------------------------------------------------------

1. Load your spectral data as a ``pandas.DataFrame``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

First load your spectral data. In this case, we assume a file called ``spectra.csv`` where each row represents a spectrum and each column represents wavenumbers.

.. code-block:: python

    import pandas as pd
    from chemotools.baseline import AirPls
    
    # Load your data as a pandas DataFrame
    spectra = pd.read_csv('data/spectra.csv', index_col=0)

The ``spectra`` variable is a ``pandas.DataFrame`` object with the indices representing the sample names and the columns representing the wavenumbers.

2. Create a ``chemotools`` preprocessing object and set the output to ``pandas``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Next, we create the ``AirPls`` object and set the output to ``pandas``.

.. code-block:: python

    # Create an AirPLS object and set the output to pandas
    airpls = AirPls().set_output(transform='pandas')

The ``set_output()`` method accepts the following arguments:

- ``transform``: The output format. Can be ``'pandas'`` or ``'default'`` (the default format will output a ``numpy.ndarray``).

.. hint::
    If you wanted to set the output to ``polars`` you would use ``transform='polars'`` in the ``set_output()`` method (``AirPLS().set_output(transform='polars')``).

3. Fit and transform the spectra
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # Fit and transform the spectra
    spectra_airpls = airpls.fit_transform(spectra)

The output of the ``fit_transform()`` method is now a ``pandas.DataFrame`` object.

.. hint::
    Notice that by default the indices and the columns of the input data are not maintained to the output, and the ``spectra_airpls`` DataFrame has default indices and columns.

**Example 2: Using the set_output() API with a pipeline**
-------------------------------------------------------------

Similarly, the ``set_output()`` API can be used with pipelines. The following code shows how to create a pipeline that performs:

- Multiplicative scatter correction
- Standard scaling

.. code-block:: python

    import pandas as pd
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from chemotools.scatter import MultiplicativeScatterCorrection
    
    # Make the pipeline
    pipeline = make_pipeline(MultiplicativeScatterCorrection(), StandardScaler())

    # Set the output to pandas
    pipeline.set_output(transform="pandas")
    
    # Fit the pipeline and transform the spectra
    output = pipeline.fit_transform(spectra)

.. hint::
    If you wanted to set the output to ``polars`` you would use ``transform='polars'`` in the ``set_output()`` method (``pipeline.set_output(transform='polars')``).

