.. _calibration_transfer:

Calibration transfer
====================

A model trained on one instrument does not always predict
accurately on a second instrument — even when measuring the same samples.
Differences in detector response, optical alignment, wavelength accuracy, and
environmental conditions all introduce systematic spectral shifts between
instruments.

**Calibration transfer** (also called *model transder* or *domain adaptation*) corrects
for these differences so that a model built on a *source* instrument can be
applied to spectra measured on a *target* instrument, without rebuilding the
model from scratch.

``chemotools`` implements two classical transfer methods in
:mod:`chemotools.adaptation`:

* :class:`~chemotools.adaptation.DirectStandardization` (DS) — a global linear
  transformation.
* :class:`~chemotools.adaptation.PiecewiseDirectStandardization` (PDS) — a
  local, window-based transformation using PLS regression.

Both follow the formulation introduced by Wang, Veltkamp & Kowalski (1991) [1]_.

The calibration transfer workflow
-----------------------------------

The general workflow is:

1. Measure a small set of **transfer samples** on both the source and target
   instruments.
2. Fit the standardization transformer using the paired spectra.
3. Apply the transformer to new target spectra before prediction.

.. code-block:: text

    Source instrument  ──►  calibration model (PLS, etc.)
                                   ▲
    Target instrument  ──►  standardization transformer  ──►  standardized spectra

After standardization, the corrected spectra can be fed directly into the
source-instrument model.

Direct Standardization (DS)
-----------------------------

:class:`~chemotools.adaptation.DirectStandardization` finds a global linear
map ``T`` that transforms target-instrument spectra to the source-instrument
space:

.. math::

   \hat{X}_{\text{source}} = X_{\text{target}} \cdot T

``T`` is estimated by ordinary least squares on the paired transfer spectra.
DS is simple, fast, and works well when the spectral differences between
instruments are globally smooth.

**Fitting DS**

Pass the target transfer spectra as ``X`` and the corresponding source
transfer spectra as ``X_source``:

.. code-block:: python

    import numpy as np
    from chemotools.adaptation import DirectStandardization

    rng = np.random.default_rng(42)
    n_transfer, n_features = 30, 100

    # Transfer samples measured on both instruments
    X_source = rng.normal(size=(n_transfer, n_features))
    X_target = X_source * 1.02 + 0.005 + rng.normal(size=(n_transfer, n_features)) * 0.002

    ds = DirectStandardization()
    ds.fit(X_target, X_source=X_source)

**Applying DS to new spectra**

.. code-block:: python

    # New spectra from the target instrument
    X_new_target = rng.normal(size=(10, n_features))

    # Transform to source-instrument space before prediction
    X_new_standardized = ds.transform(X_new_target)

**DS in a Pipeline**

.. code-block:: python

    from sklearn.pipeline import Pipeline
    from sklearn.cross_decomposition import PLSRegression

    pipe = Pipeline([
        ("ds",  DirectStandardization()),
        ("pls", PLSRegression(n_components=3)),
    ])

    # Fit the transfer step on transfer samples (X_target → X_source mapping)
    # Fit the regression step on source spectra and reference values
    # In practice these two fits are done separately; see the API reference.

.. note::

   When ``X_source`` is not provided, DS fits an identity transformation
   (i.e., ``transform`` returns ``X`` unchanged). This is useful as a
   no-op placeholder in pipelines during development.

Piecewise Direct Standardization (PDS)
-----------------------------------------

:class:`~chemotools.adaptation.PiecewiseDirectStandardization` extends DS by
building one *local* PLS model per output feature, using a small window of
neighbouring input features. This makes PDS more robust when:

* the spectral shift between instruments is non-linear or varies across the
  spectral range.
* there are wavelength registration differences (small offsets between the
  x-axis grids of the two instruments).

For each output feature :math:`j`, PDS uses the window
:math:`[j - w, \ldots, j + w]` of the target spectrum (where ``w =
window_length``) to predict the corresponding feature of the source spectrum
via PLS regression.

**Fitting PDS**

.. code-block:: python

    from chemotools.adaptation import PiecewiseDirectStandardization

    pds = PiecewiseDirectStandardization(
        window_length=3,    # half-width of the local spectral window
        n_components=2,     # PLS components per local model
    )
    pds.fit(X_target, X_source=X_source)

**Applying PDS**

.. code-block:: python

    X_new_standardized = pds.transform(X_new_target)

Choosing between DS and PDS
-----------------------------

.. list-table::
   :widths: 30 35 35
   :header-rows: 1

   * -
     - DS
     - PDS
   * - **Transformation**
     - Global linear map
     - Local PLS per feature
   * - **Number of transfer samples needed**
     - As many as features (can use regularization)
     - Few (local models have few variables)
   * - **Handles wavelength shifts**
     - Poorly
     - Well (windowed input)
   * - **Handles non-linear differences**
     - No
     - Partially
   * - **Computation time**
     - Fast
     - Moderate (one PLS per feature)
   * - **Best for**
     - Globally smooth instrument differences
     - Wavelength offsets, local non-linearities

A common strategy is to start with DS (fast, interpretable) and switch to PDS
if prediction accuracy on the target instrument is insufficient.

References
----------

.. [1] Wang, Y., Veltkamp, D. J., & Kowalski, B. R. (1991).
   Multivariate instrument standardization.
   *Analytical Chemistry*, 63(23), 2750–2756.
   https://doi.org/10.1021/ac00023a016

.. [2] Bouveresse, E., & Massart, D. L. (1996).
   Improvement of the piecewise direct standardisation procedure for the
   transfer of NIR spectra for multivariate calibration.
   *Chemometrics and Intelligent Laboratory Systems*, 32(2), 201–213.
   https://doi.org/10.1016/0169-7439(95)00074-7

.. seealso::

   * :doc:`DirectStandardization <../methods/generated/chemotools.adaptation.DirectStandardization>`
   * :doc:`PiecewiseDirectStandardization <../methods/generated/chemotools.adaptation.PiecewiseDirectStandardization>`
   * :doc:`XAxisInterpolator <../methods/generated/chemotools.adaptation.XAxisInterpolator>` — align spectra to a
     common x-axis grid before standardization.
