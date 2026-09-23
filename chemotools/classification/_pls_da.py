"""
The module :mod:`chemotools.classification._pls_da` implements PLS-DA
(Partial Least Squares Discriminant Analysis), a classifier built on top of
PLS regression by treating one-hot encoded class labels as the regression
target and assigning new samples to their nearest class centroid in the
resulting latent score space.
"""

# Author: Reza Bagheri
# License: MIT

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import label_binarize
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted, check_X_y

from chemotools.regression import PLSRegression


class PLSDA(ClassifierMixin, PLSRegression):
    """PLS Discriminant Analysis (PLS-DA) classifier.

    PLS-DA turns PLS regression into a classifier by regressing X against a
    one-hot encoded matrix of class labels, then assigning each sample to the
    class whose centroid (in the PLS score space) is closest.

    This estimator wraps :class:`chemotools.regression.PLSRegression` (itself
    backed by the Improved Kernel PLS algorithms from the ``ikpls`` package),
    so all parameters and fitted attributes from that class are available,
    including ``explained_x_variance_ratio_`` and ``explained_y_variance_ratio_``.

    Parameters
    ----------
    n_components : int, default=2
        Number of components to keep. Should be in
        [1, min(n_samples, n_features, n_classes)].
    scale : bool, default=True
        Whether to scale X and the one-hot encoded Y to unit standard
        deviation before fitting. Both are always mean-centered.
    algorithm : int, default=1
        Improved Kernel PLS algorithm to use, either 1 or 2; see
        :class:`chemotools.regression.PLSRegression` for details.
    copy : bool, default=True
        Whether to copy X and Y in fit before applying centering and
        potentially scaling.
    dtype : type, default=numpy.float64
        Floating point dtype used for the computations.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        The class labels seen during fit.
    centroids_ : ndarray of shape (n_classes, n_components)
        Mean PLS score vector for each class, used for nearest-centroid
        class assignment in ``predict``.

    All other fitted attributes (``x_weights_``, ``y_weights_``,
    ``x_loadings_``, ``y_loadings_``, ``x_scores_``, ``x_rotations_``,
    ``y_rotations_``, ``coef_``, ``intercept_``, ``n_features_in_``,
    ``explained_x_variance_ratio_``, ``explained_y_variance_ratio_``) are
    inherited from :class:`chemotools.regression.PLSRegression`. There is no
    ``y_scores_`` attribute (see that class's docstring for why).

    References
    ----------
    .. [1] Barker, M., & Rayens, W. (2003).
        Partial least squares for discrimination.
        Journal of Chemometrics, 17(3), 166-173.

    .. [2] Brereton, R. G., & Lloyd, G. R. (2014).
        Partial least squares discriminant analysis: taking the magic away.
        Journal of Chemometrics, 28(4), 213-225.

    Examples
    --------
    >>> from chemotools.classification import PLSDA
    >>> import numpy as np
    >>>
    >>> X = np.random.randn(100, 50)
    >>> y = np.array([0] * 50 + [1] * 50)
    >>>
    >>> plsda = PLSDA(n_components=3)
    >>> plsda.fit(X, y)
    >>> predictions = plsda.predict(X)
    >>> probabilities = plsda.predict_proba(X)

    Notes
    -----
    **Class assignment:**

    - Classification uses a nearest-centroid rule in PLS score space rather
      than a fixed decision threshold, which generalizes cleanly to any
      number of classes without assuming an ordering between them.
    - ``predict_proba`` returns a softmax over negative distances to each
      class centroid. These are bounded, sum-to-one pseudo-probabilities
      useful for ranking, but are not calibrated probabilities in the
      statistical sense.
    - ``decision_function`` derives its scores from those same centroid
      distances (negated, so closer is higher). For binary problems it
      collapses this to a single signed score per sample, matching
      scikit-learn's convention that the sign of ``decision_function``
      agrees with ``predict``.

    See Also
    --------
    chemotools.regression.PLSRegression : Underlying PLS regression model.
    """

    def __sklearn_tags__(self):
        """Correct target tags inherited from the underlying regressor.

        :class:`~chemotools.regression.PLSRegression` naturally supports
        multi-output regression targets (a ``Y`` with several response
        columns), so its tags mark ``target_tags.multi_output = True``.
        PLS-DA, however, is a single-label classifier: it consumes exactly
        one categorical label per sample and only *internally* one-hot
        encodes it before delegating to the regressor. Without correcting
        this inherited tag, ``check_estimator`` (via
        ``check_classifier_multioutput``) assumes PLS-DA accepts a genuine
        multilabel-indicator ``y`` directly and fits it on one, which
        ``fit`` correctly rejects, causing a spurious failure.
        """
        tags = super().__sklearn_tags__()
        tags.target_tags.multi_output = False
        tags.target_tags.single_output = True
        return tags

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PLSDA":
        """Fit the PLS-DA model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors.
        y : array-like of shape (n_samples,)
            Class labels.

        Returns
        -------
        self : PLSDA
            Fitted estimator with populated ``classes_`` and ``centroids_``.
        """
        X, y = check_X_y(X, y)
        check_classification_targets(y)

        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        y_dummy = label_binarize(y, classes=self.classes_)
        if n_classes == 2:
            # label_binarize collapses binary targets to a single column;
            # PLS-DA needs one column per class.
            y_dummy = np.hstack([1 - y_dummy, y_dummy])

        super().fit(X, y_dummy)

        self.centroids_ = np.array(
            [self.x_scores_[y == c].mean(axis=0) for c in self.classes_]
        )

        return self

    def transform(self, X: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        """Project X onto the fitted latent components (X-scores only).

        Unlike the underlying PLSRegression, PLS-DA does not support
        transforming an external ``y`` alongside X: the ``y`` consumed
        during ``fit`` is an internally constructed one-hot class matrix,
        not a quantity meaningful to transform for new data. Any ``y``
        passed here is ignored, and only the X-scores are returned (never
        the ``(x_scores, y_scores)`` tuple the base class can return).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to transform.
        y : ignored
            Present for API consistency; always ignored.

        Returns
        -------
        X_scores : ndarray of shape (n_samples, n_components)
            X projected into the latent space (X-scores).
        """
        return super().transform(X, y=None)

    def fit_transform(  # type: ignore[ty:invalid-method-override]  # narrows the base class: PLS-DA only ever returns X-scores, never (x_scores, y_scores)
        self, X: np.ndarray, y: np.ndarray
    ) -> np.ndarray:
        """Fit the model to X and y, then return the X-scores.

        Overridden to avoid delegating to the underlying regression
        backend's ``fit_transform``, which would incorrectly forward the
        raw class labels into ``transform`` a second time.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors.
        y : array-like of shape (n_samples,)
            Class labels.

        Returns
        -------
        X_scores : ndarray of shape (n_samples, n_components)
            X projected into the latent space (X-scores).
        """
        return self.fit(X, y).transform(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to classify.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class label for each sample, assigned to the nearest
            class centroid in PLS score space.
        """
        check_is_fitted(self)
        scores = self.transform(X)
        distances = euclidean_distances(scores, self.centroids_)
        return self.classes_[np.argmin(distances, axis=1)]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Estimate pseudo-probabilities for each class.

        These are derived from a softmax over negative distances to each
        class centroid in PLS score space, and are not calibrated
        probabilities in the statistical sense — see Notes.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to classify.

        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            Pseudo-probability of each class, rows summing to 1.
        """
        check_is_fitted(self)
        scores = self.transform(X)
        neg_distances = -euclidean_distances(scores, self.centroids_)
        exp_scores = np.exp(neg_distances - neg_distances.max(axis=1, keepdims=True))
        return exp_scores / exp_scores.sum(axis=1, keepdims=True)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Return per-class confidence scores based on distance to centroids.

        Scores are derived from the same nearest-centroid distances used by
        ``predict``, so the class (or sign, for binary problems) that
        ``decision_function`` favors always agrees with ``predict``.

        For binary classification, returns a single signed score per
        sample: positive values favor ``classes_[1]``, negative values
        favor ``classes_[0]``, matching scikit-learn's convention for
        binary decision functions.

        For multiclass problems, returns one score per class, where the
        class with the highest score is the predicted class (mirroring the
        nearest-centroid rule used by ``predict``).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to classify.

        Returns
        -------
        scores : ndarray of shape (n_samples,) for binary problems, or
            of shape (n_samples, n_classes) for multiclass problems.
            Higher values indicate greater confidence in the corresponding
            class; see above for how to interpret them.
        """
        check_is_fitted(self)
        scores = self.transform(X)
        distances = euclidean_distances(scores, self.centroids_)
        class_scores = -distances  # closer centroid -> higher score

        if len(self.classes_) == 2:
            # Signed score: positive means closer to classes_[1].
            return class_scores[:, 1] - class_scores[:, 0]
        return class_scores
