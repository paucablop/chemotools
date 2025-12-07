.. _astartes_integration:

Sampling with Astartes
======================

**Chemotools** focuses on preprocessing and modeling spectral data, but robust model validation starts with how you split your data. This is where **Astartes** comes in.

`Astartes <https://github.com/JacksonBurns/astartes>`_ is a Python library for rigorous dataset splitting (sampling). It serves as a drop-in replacement for ``sklearn.model_selection.train_test_split`` but offers "rational" sampling algorithms like **Kennard-Stone** and **SPXY**, which are essential in chemometrics.

Synergy with Chemotools
-----------------------

While ``chemotools`` provides the transformers to clean your spectra, ``astartes`` ensures your training set is statistically representative of your domain.

*   **Chemotools**: Preprocessing (Smoothing, Baseline correction, Derivatives).
*   **Astartes**: Calibration set selection (Kennard-Stone, SPXY).

Using them together allows you to build robust chemometric pipelines that are validated on rigorously selected data.

Example: Kennard-Stone Splitting
--------------------------------

Here is how you can use ``astartes`` to split your data using the Kennard-Stone algorithm before processing it with ``chemotools``.

.. code-block:: python

    from astartes import train_test_split
    from chemotools.datasets import load_fermentation_train
    from chemotools.baseline import AirPls
    from sklearn.pipeline import make_pipeline
    from sklearn.decomposition import PCA

    # 1. Load Data
    X, y = load_fermentation_train()

    # 2. Rational Split with Astartes (Kennard-Stone)
    # This ensures the training set covers the spectral diversity
    X_train, X_test, y_train, y_test = train_test_split(
        X.values, 
        y.values,
        train_size=0.75,
        sampler='kennard_stone'
    )

    # 3. Build Chemotools Pipeline
    pipeline = make_pipeline(
        AirPls(),
        PCA(n_components=3)
    )

    # 4. Fit and Evaluate
    pipeline.fit(X_train)
    scores = pipeline.transform(X_test)

Installation
------------

You can install astartes via pip:

.. code-block:: bash

    pip install astartes
