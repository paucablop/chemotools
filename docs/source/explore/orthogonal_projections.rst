.. _orthogonal_projections:

Orthogonal projections
======================

Spectroscopic measurements almost always contain variation that is irrelevant
to the property you want to predict — baseline drift, temperature effects,
scattering differences, instrument noise. **Orthogonal projection** methods
identify this unwanted systematic variation and remove it before calibration,
improving the interpretability and robustness of downstream models.

``chemotools`` provides three supervised orthogonal filtering methods in
:mod:`chemotools.projection`:

* :class:`~chemotools.projection.OrthogonalSignalCorrection` (OSC)
* :class:`~chemotools.projection.OrthogonalPLS` (OPLS)
* :class:`~chemotools.projection.DirectOrthogonalization` (DO)

All three are sklearn-compatible transformers: they accept ``(X, y)`` in
``fit`` and return a corrected ``X`` with the same number of features from
``transform``.

.. note::

   A fourth method — :class:`~chemotools.projection.ExternalParameterOrthogonalization`
   (EPO) — handles *unsupervised* removal of variation linked to known external
   factors (e.g., temperature, humidity). It is covered on its own page.

What does "orthogonal" mean here?
----------------------------------

After fitting a model on spectra ``X`` to predict ``y``, the total variance in
``X`` can be decomposed into:

* **Predictive variation** — correlated with ``y``, informative for the model.
* **Orthogonal variation** — uncorrelated with ``y``, pure noise or
  interference for the regression task.

Removing orthogonal variation before building the model reduces complexity,
can improve prediction accuracy, and often makes the resulting loadings easier
to interpret chemically.

.. image:: ../_static/images/explore/orthogonal_projections_concept.png
   :alt: Predictive vs orthogonal variation in a spectrum
   :align: center
   :width: 600

Orthogonal Signal Correction (OSC)
------------------------------------

:class:`~chemotools.projection.OrthogonalSignalCorrection` is the original
orthogonal filtering approach, introduced by Wold et al. (1998). It
iteratively finds components in ``X`` that have maximum variance *and* are
orthogonal to ``y``, then removes them by deflation.

Three algorithmic variants are available:

.. list-table::
   :widths: 15 55 30
   :header-rows: 1

   * - ``method``
     - Algorithm
     - Notes
   * - ``"wold"``
     - Iterative NIPALS-style. Alternates between score estimation and
       orthogonality constraint until convergence.
     - Original formulation. Can be slow for large datasets.
   * - ``"sjoblom"``
     - Modified iteration using the pseudo-inverse of ``y`` to enforce
       orthogonality directly.
     - Often converges faster than ``"wold"``.
   * - ``"fearn"``
     - Direct (non-iterative). Projects ``X`` onto the null space of ``y``
       and extracts dominant components via SVD.
     - No convergence issues; fast and deterministic.

.. code-block:: python

    import numpy as np
    from chemotools.projection import OrthogonalSignalCorrection

    rng = np.random.default_rng(42)
    X = rng.normal(size=(80, 200))
    y = rng.normal(size=80)

    # Remove 2 orthogonal components using the Fearn (non-iterative) method
    osc = OrthogonalSignalCorrection(n_components=2, method="fearn")
    X_corrected = osc.fit_transform(X, y)

    print(X_corrected.shape)  # (80, 200) — same shape, orthogonal variation removed

Orthogonal PLS (OPLS)
-----------------------

:class:`~chemotools.projection.OrthogonalPLS` extends PLS regression by
explicitly separating the predictive and orthogonal components of ``X``. At
each iteration it:

1. Estimates a predictive weight vector (maximising covariance with ``y``).
2. Computes the corresponding loading and decomposes it into a component
   aligned with the predictive direction and a component orthogonal to it.
3. Uses the orthogonal loading to define an orthogonal score, then deflates
   ``X`` by that component.

OPLS retains the predictive structure of ``X`` while filtering out orthogonal
variation, and works natively with multivariate targets.

.. code-block:: python

    from chemotools.projection import OrthogonalPLS

    opls = OrthogonalPLS(n_components=2)
    X_corrected = opls.fit_transform(X, y)

After fitting, diagnostic attributes are available:

.. code-block:: python

    print(opls.x_weights_orth_.shape)           # (200, 2) — orthogonal weights
    print(opls.retained_variance_ratio_)         # fraction of variance kept

In a pipeline, OPLS is placed before the regression step:

.. code-block:: python

    from sklearn.pipeline import Pipeline
    from sklearn.cross_decomposition import PLSRegression

    pipe = Pipeline([
        ("opls", OrthogonalPLS(n_components=1)),
        ("pls",  PLSRegression(n_components=3)),
    ])
    pipe.fit(X, y)

Direct Orthogonalization (DO)
------------------------------

:class:`~chemotools.projection.DirectOrthogonalization` is a simpler,
two-step method:

1. Orthogonalize ``X`` with respect to ``y`` (remove all linear relationship
   with ``y``).
2. Perform PCA on the orthogonalized matrix to estimate the dominant
   orthogonal directions.
3. Subtract those components from the original ``X``.

DO is computationally light and does not iterate, but it is less targeted
than OSC or OPLS because it first removes *all* correlation with ``y``
before extracting components.

.. code-block:: python

    from chemotools.projection import DirectOrthogonalization

    do = DirectOrthogonalization(n_components=2)
    X_corrected = do.fit_transform(X, y)

Choosing a method
------------------

Svensson, Kourti & MacGregor [2]_ evaluated these algorithms and identified
two groups based on how they interact with the downstream PLS calibration model:

* **Group 1 — efficient reduction**: OSC (all three variants). Removing a
  single orthogonal component is often sufficient to substantially reduce the
  number of PLS components needed in the calibration model.
* **Group 2 — one-for-one reduction**: OPLS and DO. Each orthogonal component
  removed reduces the complexity of the calibration model by exactly one PLS
  component.

Importantly, the study found that none of the algorithms provided a consistent
improvement in *prediction accuracy* over PLS on raw data. The primary benefit
is **interpretability**: the corrected data tends to be simpler to analyse and
the removed orthogonal variation can itself carry useful diagnostic information.

.. list-table::
   :widths: 20 15 15 20 30
   :header-rows: 1

   * - Method
     - Group [2]_
     - Iterative
     - Multivariate ``y``
     - Notes
   * - OSC (wold)
     - 1
     - Yes
     - No
     - Original formulation; can behave differently under non-linearities.
   * - OSC (sjoblom)
     - 1
     - Yes
     - No
     - Modified iteration; often converges faster than wold.
   * - OSC (fearn)
     - 1
     - No
     - No
     - Direct, non-iterative; deterministic and fast.
   * - OPLS
     - 2
     - Yes (deflation)
     - Yes
     - Explicitly separates predictive and orthogonal variation.
   * - DO
     - 2
     - No
     - No
     - Orthogonalizes X w.r.t. y first, then extracts components via PCA.

A practical rule of thumb:

* Use **OSC (fearn)** as the default — it is non-iterative, deterministic,
  and one component is often sufficient.
* Use **OPLS** when ``y`` is multivariate or when you want explicit separation
  of predictive and orthogonal scores for further analysis.
* Use **DO** as a simple, fast baseline to compare against.

.. [2] Svensson, O., Kourti, T., & MacGregor, J. F. (2002).
   An investigation of orthogonal signal correction algorithms and their
   characteristics. *Journal of Chemometrics*, 16(4), 176–188.
   https://doi.org/10.1002/cem.700

Fitting in a Pipeline
----------------------

All three transformers follow the standard sklearn API and compose freely with
other steps:

.. code-block:: python

    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.cross_decomposition import PLSRegression
    from chemotools.projection import OrthogonalSignalCorrection

    pipe = Pipeline([
        ("osc",    OrthogonalSignalCorrection(n_components=2, method="fearn")),
        ("scaler", StandardScaler(with_std=False)),
        ("pls",    PLSRegression(n_components=3)),
    ])

    pipe.fit(X, y)
    y_pred = pipe.predict(X)

.. seealso::

   * :class:`chemotools.projection.OrthogonalSignalCorrection`
   * :class:`chemotools.projection.OrthogonalPLS`
   * :class:`chemotools.projection.DirectOrthogonalization`
   * :class:`chemotools.projection.ExternalParameterOrthogonalization`
