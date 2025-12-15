.. _astartes_integration:

Sampling with Astartes
======================

If you've ever wondered whether a random train/test split is really the best approach for spectroscopic data, you're not alone. Random splits can leave gaps in your calibration set or, worse, put very similar samples in both sets—leading to overly optimistic error estimates.

`Astartes <https://github.com/JacksonBurns/astartes>`_ solves this problem. It's a Python library that works as a drop-in replacement for ``sklearn.model_selection.train_test_split``, but with access to sampling algorithms that actually make sense for chemometrics: **Kennard-Stone**, **SPXY**, and others.

Why does this matter?
---------------------

Kennard-Stone, for instance, picks samples that maximize the coverage of your spectral space. Instead of hoping that a random draw gives you a representative calibration set, you *guarantee* it. This reduces the risk of extrapolation when predicting new samples.

SPXY goes a step further by considering both the spectra (X) and the reference values (y) when selecting samples—useful when your y-values span a wide range.

Available Samplers
------------------

Astartes groups its samplers into two categories:

*   **Interpolative**: samples are chosen to ensure coverage *within* the data space. Good for building general-purpose models.
*   **Extrapolative**: samples are grouped into clusters, and entire clusters are held out. This tests how well the model generalizes to *unseen regions*.

Here are some of the most useful samplers for spectroscopic data:

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Sampler
     - Type
     - When to use it
   * - ``random``
     - Interpolative
     - Baseline comparison. Same as sklearn's default.
   * - ``kennard_stone``
     - Interpolative
     - The go-to for spectroscopy. Picks samples that maximize coverage of the X-space.
   * - ``spxy``
     - Interpolative
     - Like Kennard-Stone, but also considers the y-values. Great for calibration transfer.
   * - ``kmeans``
     - Extrapolative
     - Clusters samples, then holds out entire clusters. Tests generalization to new regions.
   * - ``sphere_exclusion``
     - Extrapolative
     - Excludes samples within a distance cutoff. Useful for testing robustness to gaps in the data.

For the full list (including molecule-specific samplers like Scaffold), check the `Astartes GitHub page <https://github.com/JacksonBurns/astartes#implemented-sampling-algorithms>`_.

Pairing with Chemotools
-----------------------

The two libraries complement each other nicely:

*   **Chemotools** handles the spectral preprocessing (smoothing, baseline correction, derivatives, etc.).
*   **Astartes** handles the sample selection before you even start modeling.

Together, they give you a clean, well-validated pipeline.

Example: Kennard-Stone Splitting
--------------------------------

Here's a quick example. We load some spectra, split them with Kennard-Stone, and then run them through a preprocessing pipeline:

.. code-block:: python

    from astartes import train_test_split
    from chemotools.datasets import load_fermentation_train
    from chemotools.baseline import AirPls
    from sklearn.pipeline import make_pipeline
    from sklearn.decomposition import PCA

    # Load the data (returns pandas DataFrames)
    X, y = load_fermentation_train()

    # Split with Kennard-Stone instead of random
    X_train, X_test, y_train, y_test = train_test_split(
        X.values,  # astartes expects numpy arrays
        y.values,
        train_size=0.75,
        sampler='kennard_stone'
    )

    # Preprocess and reduce dimensionality
    pipeline = make_pipeline(
        AirPls(),
        PCA(n_components=3)
    )

    pipeline.fit(X_train)
    scores = pipeline.transform(X_test)

If you want to include the y-values in the selection logic, just swap in ``sampler='spxy'``.

Installation
------------

Astartes is not a dependency of chemotools, so you'll need to install it separately:

.. code-block:: bash

    pip install astartes

For more details, check out the `Astartes documentation <https://jacksonburns.github.io/astartes/>`_.
