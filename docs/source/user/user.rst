Quick start
===========

Check the `installation guide <../user/install.html>`_ to get started.

.. hint::
    **New in chemometrics?** Learn through our step-by-step `tutorials page <../learn/index.html>`_.

.. hint::
    **Want to explore the tools at your fingertips?** Visit the `explore overview <../explore/index.html>`_.

Load data
---------

This loads a training set of mid infrared spectra (``X_train``) and associated reference values (``y_train``) from a fermentation process.

.. code-block:: python

    from chemotools.datasets import load_fermentation_train

    X_train, y_train = load_fermentation_train()

.. hint::
    **Looking for available datasets?** See the `datasets catalog <../learn/datasets.html>`_.

Preprocess spectra
------------------

Load, define and apply preprocessing steps to the training data. Here we use a Savitzky-Golay filter to compute the first derivative of the sample spectra.

.. code-block:: python

    from chemotools.derivate import SavitzkyGolay

    # Configure the preprocessing step
    sg = SavitzkyGolay(window_size=3, polynomial_order=1, derivate_order=1)

    # Fit and transform the training data
    X_train_preprocessed = sg.fit_transform(X_train)

.. hint::
    **Want to know more about preprocessing methods?** Read the `methods index <../methods/index.html>`_.

Build a pipeline
----------------

Combine preprocessing and modelling steps in a single pipeline. 

.. code-block:: python

    from chemotools.derivate import SavitzkyGolay
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.pipeline import make_pipeline
    
    # Define the pipeline
    pipeline = make_pipeline(
        SavitzkyGolay(window_size=3, polynomial_order=1, derivate_order=1),
        PLSRegression(n_components=2)
    )

    # Fit the pipeline to the training data
    pipeline.fit(X_train, y_train)

    # Prediction on the training data
    y_train_pred = pipeline.predict(X_train)

.. hint::
    **Want to know more about pipelines?** Open the `pipelines article <../explore/pipelines.html>`_.

.. hint::
    **Curious about model optimization?** Read the `optimization guide <../explore/optimize.html>`_.

**Congratulations!** You have just built your first chemotools workflow in Python 🎯