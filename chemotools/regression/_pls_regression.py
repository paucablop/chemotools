"""
The module :mod:`chemotools.regression._pls_regression` implements PLS regression
backed by the fast and numerically stable Improved Kernel PLS algorithms from the
`ikpls <https://github.com/sm00thix/ikpls>`_ package, with automatic explained
variance calculation for both X and Y spaces.
"""

# Authors: Pau Cabaneros, Ole-Christian Galbo Engstrøm
# License: MIT

from numbers import Integral
from typing import Tuple

import numpy as np
from ikpls.sklearn import PLS as _IkplsPLS
from sklearn.utils._param_validation import Interval, Options

from chemotools._doc_mixin import DocLinkMixin


class PLSRegression(DocLinkMixin, _IkplsPLS):
    """PLS regression via Improved Kernel PLS with automatic explained variance.

    This estimator wraps the fast, exact Improved Kernel PLS algorithms [1]_
    from the `ikpls <https://github.com/sm00thix/ikpls>`_ package [2]_ in a
    scikit-learn-conformant regressor + transformer, and automatically
    calculates explained variance ratios for both X-space and Y-space after
    fitting, making it easy to use with diagnostic plots and following the
    same API as PCA.

    Compared to ``sklearn.cross_decomposition.PLSRegression`` (NIPALS) [3]_ and [4]_,
    the Improved Kernel PLS algorithms are faster [5]_ while still being numerically
    stable [6]_. Predictions and regression coefficients agree with NIPALS;
    individual weight/score/loading vectors are identical up to a per-component sign,
    which is arbitrary in PLS.

    Additional capabilities inherited from ikpls:

    - :meth:`predict_all_components` returns predictions for every number of
      components ``1..n_components`` in a single call (shape
      ``(n_components, n_samples, n_targets)``), making component selection
      via (cross-)validation cheap.
    - Fine-grained preprocessing on the backend: the ikpls model supports
      independent ``center_X`` / ``center_Y`` / ``scale_X`` / ``scale_Y`` and a
      configurable ``ddof``. This wrapper exposes only a single ``scale`` flag
      (``X`` and ``Y`` are always mean-centered) and fixes ``ddof = 1``, to
      mirror the previous scikit-learn-backed ``PLSRegression``. A maintainer can
      enable the independent flags or a different ``ddof`` on the inner model,
      but they change the fitted model and the explained-variance computation.
    - Sample weights: the backend's ``fit`` accepts ``sample_weight`` (when
      given, the inner ``X`` / ``Y`` means and standard deviations become their
      weighted variants). This wrapper does not expose ``sample_weight``;
      whether to enable it is left to the maintainer.

    After fitting, two additional attributes are computed:

    - ``explained_x_variance_ratio_``: Variance explained in X-space (predictors)
    - ``explained_y_variance_ratio_``: Variance explained in Y-space (response)

    Following the PLSRegression implemented in scikit-learn [3]_, the
    explained variance calculation uses the x_scores_ (t) to asymmetrically
    deflate the Y matrix.

    In PLS, the latent score vector t (from X) is used to
    model Y via its loading vector c:
        Y_hat = t @ c.T

    Deflation removes the part of Y explained by the current component:
        Y_new = Y - Y_hat

    This process is repeated for each component, using the
    corresponding t and c vectors.
    Note: Unlike PCA, deflation in PLS is asymmetric—Y is
    deflated using t-scores derived from X.

    Parameters
    ----------
    n_components : int, default=2
        Number of components to keep. Should be in
        [1, min(n_samples, n_features)].

    scale : bool, default=True
        Whether to scale ``X`` and ``Y`` to unit standard deviation before
        fitting. ``X`` and ``Y`` are always mean-centered, matching the previous
        scikit-learn-backed ``PLSRegression``. Internally this maps onto the
        ikpls backend as ``center_X = center_Y = True`` and
        ``scale_X = scale_Y = scale``.

    algorithm : int, default=1
        Improved Kernel PLS algorithm to use, either 1 or 2. Algorithm 1 uses
        ``X`` directly, while algorithm 2 builds ``X.T @ X`` and is typically
        faster for tall matrices (many more samples than features). Both
        algorithms give the same results.

    copy : bool, default=True
        Whether to copy ``X`` and ``Y`` in fit before applying centering and
        potentially scaling.

    dtype : type, default=numpy.float64
        Floating point dtype used for the computations.

    Attributes
    ----------
    explained_x_variance_ratio_ : ndarray of shape (n_components,)
        Explained variance ratio in X-space (predictors) for each component.
        This measures how much variance in the predictor variables each latent
        variable captures. Automatically calculated after fitting.

    explained_y_variance_ratio_ : ndarray of shape (n_components,)
        Explained variance ratio in Y-space (response) for each component.
        This measures the prediction quality - how much variance in the response
        each latent variable explains. Automatically calculated after fitting.

        Both ratios are computed by sequential deflation in the centered
        (optionally scaled) space, which the public API always uses. For inputs
        with nonzero variance they are valid: non-negative, with the X ratios
        summing to 1 at full rank. Components beyond the numerical rank of ``X``
        carry no variance and are assigned zero.

        Degenerate inputs: if ``X`` (all features) or ``y`` is fully constant --
        i.e. has zero total variance -- the corresponding ratios are ``NaN`` (a
        ``0/0`` result), since there is no variance to apportion.

    x_scores_ : ndarray of shape (n_samples, n_components)
        The training X-scores ``T`` (the projection of the preprocessed
        training ``X`` onto ``x_rotations_``), equal to ``transform(X)`` on
        the training data. (There is no ``y_scores_`` attribute; the Y-scores
        are available on demand via ``transform(X, y)`` -- see :meth:`transform`
        for the caveat on how they relate to scikit-learn.)

    All other fitted attributes (``x_weights_``, ``y_weights_``,
    ``x_loadings_``, ``y_loadings_``, ``x_rotations_``, ``y_rotations_``,
    ``coef_``, ``intercept_``, ``n_features_in_``, ``feature_names_in_``) are
    inherited unchanged from the ikpls backend (``ikpls.sklearn.PLS``); see the
    ikpls documentation (https://ikpls.readthedocs.io/en/latest/) for their
    definitions.

    References
    ----------
    .. [1] Dayal, B. S., & MacGregor, J. F. (1997).
        Improved PLS algorithms.
        Journal of Chemometrics, 11(1), 73-85.
        https://doi.org/10.1002/(SICI)1099-128X(199701)11:1%3C73::AID-CEM435%3E3.0.CO;2-%23

    .. [2] Engstrøm, O.-C. G., Dreier, E. S., Jespersen, B. M., & Pedersen,
        K. S. (2024).
        IKPLS: Improved Kernel Partial Least Squares and Fast Cross-Validation
        Algorithms for Python with CPU and GPU Implementations Using NumPy and
        JAX.
        Journal of Open Source Software, 9(99), 6533.
        https://doi.org/10.21105/joss.06533

    .. [3] sklearn.cross_decomposition.PLSRegression
        https://scikit-learn.org/stable/modules/generated/sklearn.cross_decomposition.PLSRegression.html

    .. [4] Wegelin, J. A. (2000).
        A Survey of Partial Least Squares (PLS) Methods,
        with Emphasis on the Two-Block Case.
        Technical Report No. 371, Department of Statistics,
        University of Washington, Seattle, WA

    .. [5] Alin, A. (2009).
        Comparison of PLS algorithms when number of objects is much larger
        than number of variables.
        Statistical Papers, 50, 711-720.
        https://doi.org/10.1007/s00362-009-0251-7

    .. [6] Andersson, M. (2009).
        A comparison of nine PLS1 algorithms.
        Journal of Chemometrics, 23(10), 518-529.
        https://doi.org/10.1002/cem.1248

    .. [7] Abdi, H. (2003).
        Partial Least Squares (PLS) Regression.
        In Lewis-Beck M., Bryman A., Futing T. (Eds.),
        Encyclopedia of Social Sciences Research Methods.
        Thousand Oaks (CA): Sage.

    Examples
    --------
    **Basic usage with automatic variance calculation**

    >>> from chemotools.regression import PLSRegression
    >>> import numpy as np
    >>>
    >>> # Generate sample data
    >>> X = np.random.randn(100, 50)
    >>> y = X[:, 0] + 2*X[:, 1] + np.random.randn(100)*0.1
    >>>
    >>> # Fit model
    >>> pls = PLSRegression(n_components=5)
    >>> pls.fit(X, y)
    >>>
    >>> # Variance ratios are automatically available!
    >>> print(
    ...     f"LV1 explains {pls.explained_y_variance_ratio_[0]*100:.1f}%"
    ... )
    >>> print(f"Total Y variance: {pls.explained_y_variance_ratio_.sum()*100:.1f}%")
    >>>
    >>> # Predictions for ALL component counts 1..5 in a single call
    >>> all_predictions = pls.predict_all_components(X)
    >>>
    >>> # Use with plotting
    >>> from chemotools.plotting import ExplainedVariancePlot
    >>> plot = ExplainedVariancePlot(pls.explained_y_variance_ratio_)
    >>> plot.show()

    Notes
    -----
    **Variance Calculation:**

    - **X-space variance** is calculated using sequential
      deflation and sums to 1.0 (100%) when ``n_components`` equals the rank
      of the (centered) ``X``
    - **Y-space variance** is calculated using sequential
      deflation but may not sum to 1.0 due to asymmetric
      deflation (Y deflated with X-scores). The sum depends
      on X-Y correlation.
    - For each component, variance explained = variance reduction after deflation
    - This follows the standard PLS variance decomposition methodology (Wegelin, 2000)
    - Components beyond the numerical rank of ``X`` carry no variance and are
      assigned zero explained variance, so the ratios stay non-negative and the
      X ratios sum to 1 at full rank even when ``n_components`` exceeds that
      rank (and are identical for ``algorithm=1`` and ``algorithm=2``).

    **Differences from the previous scikit-learn (NIPALS) backend:**

    - The NIPALS-specific ``max_iter`` and ``tol`` parameters do not exist:
      Improved Kernel PLS is an exact, non-iterative solver.
    - The single ``scale`` parameter is retained with the same meaning (``X``
      and ``Y`` are always mean-centered, and ``scale`` toggles unit-variance
      scaling of both). The ikpls backend's independent ``center_X`` /
      ``center_Y`` / ``scale_X`` / ``scale_Y`` are not exposed here.
    - Score/weight/loading vectors may differ from NIPALS by a per-component
      sign; predictions, coefficients, and explained variances are unaffected.

    See Also
    --------
    sklearn.cross_decomposition.PLSRegression : scikit-learn's NIPALS-based PLS.
    chemotools.plotting.ExplainedVariancePlot : Visualization for explained variance.
    """

    _parameter_constraints: dict = {
        "n_components": [Interval(Integral, 1, None, closed="left")],
        "scale": ["boolean"],
        "algorithm": [Options(Integral, {1, 2})],
        "copy": ["boolean"],
        "dtype": "no_validation",
    }

    def __init__(
        self,
        n_components: int = 2,
        *,
        scale: bool = True,
        algorithm: int = 1,
        copy: bool = True,
        dtype: type = np.float64,
    ):
        # This wrapper deliberately mirrors the previous scikit-learn-backed
        # PLSRegression: it exposes a single ``scale`` flag (fixing
        # ``center_X = center_Y = True`` and ``scale_X = scale_Y = scale``) and
        # fixes ``ddof = 1`` (Bessel's correction, as scikit-learn uses). The
        # ikpls backend supports independent center_X/center_Y/scale_X/scale_Y
        # and a configurable ddof; a maintainer can enable them on the inner
        # model, but they are not part of the public API here -- they affect the
        # fitted model and the explained-variance computation (see the
        # limitation below).
        #
        # ``ikpls.sklearn.PLS.__init__`` only stores the preprocessing/solver
        # parameters as attributes; we set our five public ones directly (so
        # ``get_params`` / ``clone`` / ``check_estimator`` see exactly the public
        # parameters) and expose the fixed/derived backend flags as read-only
        # properties for the inherited ``fit`` to consume.
        self.n_components = n_components
        self.scale = scale
        self.algorithm = algorithm
        self.copy = copy
        self.dtype = dtype

    # The inherited ikpls ``fit`` reads ``center_X`` / ``center_Y`` / ``scale_X``
    # / ``scale_Y`` / ``ddof``; expose them as fixed (centering always on,
    # ``ddof = 1``) / derived (scaling tied to ``scale``) read-only properties.
    @property
    def center_X(self) -> bool:  # noqa: N802 - mirrors the ikpls attribute name
        return True

    @property
    def center_Y(self) -> bool:  # noqa: N802 - mirrors the ikpls attribute name
        return True

    @property
    def scale_X(self) -> bool:  # noqa: N802 - mirrors the ikpls attribute name
        return self.scale

    @property
    def scale_Y(self) -> bool:  # noqa: N802 - mirrors the ikpls attribute name
        return self.scale

    @property
    def ddof(self) -> int:
        return 1

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PLSRegression":  # type: ignore[ty:invalid-method-override]  # sample_weight intentionally not exposed; ikpls PLS.fit's optional sample_weight is deliberately dropped
        """Fit model to data and compute explained variance ratios.

        This method extends ``ikpls.sklearn.PLS.fit`` by storing the training
        scores and automatically calculating explained variance ratios after
        fitting.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors. Accepts numpy arrays, pandas DataFrames.
        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target vectors. Accepts 1D (univariate) or 2D (multivariate) targets.

        Returns
        -------
        self : PLSRegression
            Fitted estimator with populated variance attributes:
            ``explained_x_variance_ratio_`` and ``explained_y_variance_ratio_``.
        """
        self._validate_params()

        # The inner PLS centers/scales X and y IN PLACE when copy=False, so
        # snapshot the raw inputs first. Both the score computation (algorithm 2
        # re-projects via transform) and the explained-variance deflation re-apply
        # the fitted preprocessing, so they must start from the original data, not
        # data already preprocessed in place. With copy=True the inputs are left
        # untouched, so no snapshot is needed.
        if self.copy:
            X_raw, y_raw = X, y
        else:
            X_raw = np.array(X, dtype=self.dtype)
            y_raw = np.array(y, dtype=self.dtype)

        super().fit(X, y)

        # ikpls stores the training X-scores in inner_.T for algorithm 1 only
        # (algorithm 2 leaves it None), so recompute them there. Use the inner
        # model's own transform (not super().transform) so this internal recompute
        # does not re-run scikit-learn's feature-name check: with copy=False X_raw
        # is a label-stripped snapshot, which would otherwise emit a spurious "X
        # does not have valid feature names" warning for a DataFrame input.
        # np.asarray/np.copy so x_scores_ is a fresh array that never aliases the
        # inner model's stored T.
        self.x_scores_ = (
            np.asarray(
                self.inner_.transform(
                    X=np.asarray(X_raw, dtype=self.dtype),
                    n_components=self.n_components_,
                )
            )
            if self.algorithm == 2
            # inner_.T is the stored X-scores (non-None) in the algorithm-1 branch.
            else np.copy(self.inner_.T)  # type: ignore[ty:no-matching-overload]
        )

        # Calculate explained variance ratios automatically (from the raw inputs)
        (
            self.explained_x_variance_ratio_,
            self.explained_y_variance_ratio_,
        ) = self._calculate_explained_variance_deflation(X_raw, y_raw)

        return self

    def fit_transform(  # type: ignore[ty:invalid-method-override]  # narrows the ikpls fit_transform: y required (PLS is supervised), no **fit_params (sample_weight not exposed)
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Learn and apply the dimension reduction on the training data.

        Fits and returns the ``(x_scores, y_scores)`` tuple, matching
        ``sklearn.cross_decomposition.PLSRegression.fit_transform``.
        Unlike the ikpls backend's
        ``fit_transform``, this wrapper does not accept ``sample_weight`` (it is
        intentionally not exposed; see the class docstring).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors.
        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target vectors.

        Returns
        -------
        (x_scores, y_scores) : tuple of ndarray
            The training X-scores and Y-scores.
        """
        return super().fit_transform(X, y)

    def transform(self, X, y=None):
        """Project ``X`` (and optionally ``Y``) onto the latent components.

        Returns the X-scores; when ``y`` is given, also returns the Y-scores as
        an ``(x_scores, y_scores)`` tuple, matching
        ``sklearn.cross_decomposition.PLSRegression.transform``.

        Note on the Y-scores: the returned Y-scores are the projection of the
        preprocessed ``Y`` onto ``y_rotations_`` -- the same quantity
        scikit-learn's ``transform(X, y)`` returns, but *not* the classic
        NIPALS ``u`` vectors (which scikit-learn stores in its ``y_scores_``
        attribute). In regression-mode PLS, ``Y`` is deflated by the X-scores,
        so ``u`` depends on ``X`` and is not recoverable as any linear
        projection of ``Y``; the rotation projection and ``u`` coincide only
        under canonical deflation (``PLSCanonical``). This class therefore does
        not expose a ``y_scores_`` attribute.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Predictor variables to project.
        y : array-like of shape (n_samples,) or (n_samples, n_targets), optional
            Response variables to project. If given, Y-scores are also returned.

        Returns
        -------
        x_scores : ndarray of shape (n_samples, n_components)
            Returned when ``y`` is None.
        (x_scores, y_scores) : tuple of ndarray
            Returned when ``y`` is given.
        """
        return super().transform(X, y)

    def _calculate_explained_variance_deflation(
        self, X, y
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate explained variance ratios using sequential deflation.

        This implements the variance decomposition for PLS regression following
        the deflation methodology described in Wegelin (2000): each component's
        explained variance is the drop in residual variance when that component
        is deflated from ``X`` and ``Y``.

        .. warning::

            This calculation assumes the centered preprocessing the public API
            always uses (``X`` and ``Y`` are mean-centered; ``scale`` only
            toggles unit-variance scaling). Components beyond the numerical rank
            of ``X`` (which ikpls may emit when ``n_components`` is large) have a
            numerically-zero score and are skipped, so the ratios stay valid --
            non-negative, summing to one at full rank, and identical for
            ``algorithm=1`` and ``algorithm=2``. (A fully constant ``X`` or ``y``
            has zero total variance, so its ratios are ``NaN``.)

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors. Accepts numpy arrays, pandas DataFrames.
        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target vectors. Accepts 1D (univariate) or 2D (multivariate) targets.

        Returns
        -------
        tuple[ndarray, ndarray]
            - X variance ratios of shape (n_components,)
            - Y variance ratios of shape (n_components,)
        """
        # Convert to arrays and ensure y is 2D (handles pandas DataFrame/Series).
        X = np.asarray(X, dtype=self.dtype)
        y_array = np.asarray(y, dtype=self.dtype)
        y = np.atleast_2d(y_array).T if y_array.ndim == 1 else y_array

        # Preprocess into the exact space the model was fitted in -- the public
        # API always mean-centers and scales both blocks together per the single
        # ``scale`` flag -- reusing the model's own fitted moments so the
        # deflation stays consistent with the fitted scores/loadings.
        #
        # NB: an earlier version recomputed the std here as
        # ``np.maximum(std, 1.0)``. That was a bug -- it left every feature with
        # std < 1 (the common spectral case) unscaled, desyncing from the fit
        # (which divides by the real std) and yielding negative / non-normalized
        # ratios. The clamp is also unnecessary: ikpls already guards near-zero
        # std when fitting (``inner.X_std[inner.X_std <= eps] = 1``, and likewise
        # for ``Y_std``), so reusing ``inner.X_std`` / ``inner.Y_std`` is both
        # correct and safe against division by zero.
        #
        # ``X - inner.X_mean`` allocates a fresh array, so the in-place scaling
        # below (and the deflation, which works on the ``X_current`` copy) never
        # touch the caller's data.
        inner = self.inner_
        X_centered = X - inner.X_mean
        y_centered = y - inner.Y_mean
        if self.scale:
            X_centered /= inner.X_std
            y_centered /= inner.Y_std

        # Total variance in centered data
        X_total_var = np.var(X_centered, axis=0).sum()
        y_total_var = np.var(y_centered, axis=0).sum()

        # Initialize matrices for deflation
        X_current = X_centered.copy()
        y_current = y_centered.copy()

        # Components past the numerical rank of X need no special handling here.
        # When ``n_components`` exceeds that rank, the ikpls backend stops its fit
        # loop at ``max_stable_components_`` and leaves the remaining components'
        # scores and loadings as exact zeros (see ``max_stable_components_`` and
        # ``ikpls.numpy.PLS.max_stable_components``). A zero score ``t`` gives a zero
        # reconstruction ``t @ p.T``, so those components deflate nothing and
        # contribute exactly zero explained variance -- the ratios stay non-negative,
        # sum to one, and are algorithm-independent, mirroring scikit-learn's NIPALS
        # PLSRegression, which likewise stops early on rank-deficient data.
        X_var_ratios = []
        y_var_ratios = []

        # For each component, calculate variance explained then deflate
        for a in range(self.n_components):
            # Get scores and loadings for component a (using slicing to keep 2D)
            t_a = self.x_scores_[:, a : a + 1]  # (n_samples, 1)
            p_a = self.x_loadings_[:, a : a + 1]  # (n_features_X, 1)
            q_a = self.y_loadings_[:, a : a + 1]  # (n_features_y, 1)

            # Reconstruct X and y using current component
            X_hat = t_a @ p_a.T
            y_hat = t_a @ q_a.T

            # Variance of current residual before deflation
            X_var_before = np.var(X_current, axis=0).sum()
            y_var_before = np.var(y_current, axis=0).sum()

            # Deflate X and y
            X_current -= X_hat
            y_current -= y_hat

            # Variance of residual after deflation
            X_var_after = np.var(X_current, axis=0).sum()
            y_var_after = np.var(y_current, axis=0).sum()

            # Store variance explained as ratio of total variance
            X_var_ratios.append((X_var_before - X_var_after) / X_total_var)
            y_var_ratios.append((y_var_before - y_var_after) / y_total_var)

        return np.array(X_var_ratios), np.array(y_var_ratios)
