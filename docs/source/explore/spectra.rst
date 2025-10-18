.. _spectra:

Working with spectra
====================

When working with spectroscopic data in ``chemotools`` and ``scikit-learn``, you often need to reshape single spectra to fit the expected data shapes. This guide explains how to reshape single spectra for preprocessing in ``scikit-learn`` and ``chemotools``.

Understanding Data Shapes
-------------------------

``chemotools`` and ``scikit-learn`` preprocessing techniques expect 2D arrays (matrices) where:

* Each row represents a sample
* Each column represents a feature

However, spectroscopic data often comes as single spectrum in 1D arrays (vectors). Here's an example of a single spectrum:

.. code-block:: python

    array([0.484434, 0.485629, 0.488754, 0.491942, 0.489923, 0.492869,
           0.497285, 0.501567, 0.500027, 0.50265])

To use ``chemotools`` and  ``scikit-learn`` with single spectra, you need to reshape the 1D array into a 2D array with one row.

Reshaping for Preprocessing
---------------------------

Here's how to reshape a 1D array into a 2D array with a single row:

.. code-block:: python

    import numpy as np

    spectra_2d = spectra_1d.reshape(1, -1)

The ``reshape(1, -1)`` method converts the 1D array ``spectra_1d`` into a 2D array with a single row. The result (``spectra_2d``) looks like this:

.. code-block:: python

    array([[0.484434, 0.485629, 0.488754, 0.491942, 0.489923, 0.492869,
            0.497285, 0.501567, 0.500027, 0.50265]])

.. note::
   The reshaped output is a 2D array with a single row - the format required by 
   ``scikit-learn`` and ``chemotools`` preprocessing techniques.

Now, you can use the reshaped single spectrum with ``chemotools`` and ``scikit-learn`` preprocessing techniques:

.. code-block:: python

    import numpy as np
    from chemotools.scatter import MultiplicativeScatterCorrection

    msc = MultiplicativeScatterCorrection()
    spectra_msc = msc.fit_transform(spectra_2d))

