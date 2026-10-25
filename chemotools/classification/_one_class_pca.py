"""One-class classification based on Principal Component Analysis (PCA).

The module :mod:`chemotools.classification._one_class_pca` implements
OneClassPCA, which models a target class using PCA and estimates acceptance
and outlier boundaries from score and orthogonal distances.
"""

# Author: Pau Cabaneros
# License: MIT

import warnings
from numbers import Integral, Real
from typing import Literal

import numpy as np
from scipy.optimize import brentq
from scipy.stats import chi2
from sklearn.base import BaseEstimator, OutlierMixin
from sklearn.decomposition import PCA
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.validation import check_is_fitted, validate_data

from chemotools._doc_mixin import DocLinkMixin

# Approximation coefficients for fast robust estimation DoF
# Text preceding Eq. (16) in [3].
CONSTANT_A = 0.72414
CONSTANT_B = 2.68631
CONSTANT_C = 0.84332


# Helper method to ensure positive finite values
def _check_positive_finite(values, name):
    """Raise ValueError if any value is nonfinite or not strictly positive."""
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError(f"{name} must be finite and strictly positive.")


class OneClassPCA(DocLinkMixin, OutlierMixin, BaseEstimator):
    """One-class PCA (Principal Component Analysis) classifier.

    Model a target class using PCA. Score distances measure variation within
    the retained PCA subspace, while orthogonal distances measure deviation
    from it. Scaled chi-square distributions define limits for these
    distances [1]_, [2]_.

    Choose ``robust``, ``robust-fast``, or ``classic`` estimation and a
    triangular or rectangular boundary. The acceptance and outlier boundaries
    divide training samples into three categories: regular, extreme, and
    outlier [3]_.

    Parameters
    ----------
    n_components : int, default=2
        Number of principal components used to model the target class.
        Retained components must have numerically nonzero singular values.

    alpha : float, default=0.05
        Nominal probability of rejecting an individual genuine class member
        at the acceptance boundary. Must be strictly between 0 and 1.

    gamma : float, default=0.01
        Nominal probability of at least one false outlier flag across the
        training set. Must be strictly between 0 and 1 and yield
        ``outlier_alpha_ < alpha``. See Notes for the calibration assumptions.

    method : {"robust", "robust-fast", "classic"}, default="robust"
        Method for estimating the distance distributions:

        - "robust": estimate degrees of freedom by matching the observed
          IQR-to-median ratio to a chi-square distribution using Brent's
          root solver (Equation (15) in [3]_).
        - "robust-fast": estimate degrees of freedom using the explicit
          approximation in Equation (16) in [3]_.
        - "classic": estimate degrees of freedom from mean distances and
          sample standard deviations (Equation (12) in [3]_).

        All methods round degrees of freedom to the nearest integer, with
        a minimum of 1. Both robust methods estimate mean distances from
        medians, interquartile ranges, and the estimated degrees of freedom.

    boundary : {"triangle", "rectangle"}, default="triangle"
        Geometry used for both acceptance and outlier boundaries:

        - "triangle": combine score and orthogonal distances into one
          acceptance criterion (boundary II in Figure 2 and Equations
          (26)-(27) in [1]_).
        - "rectangle": apply separate limits to the two distances
          and reject a sample if either limit is exceeded (boundary I in
          Figure 2 and Eq. (25) in [1]_).

    upper_bracket : float, default=10_000
        Upper endpoint of the root-search bracket ``(1, upper_bracket)``
        for Brent's method. Used only when ``method="robust"``. The bracket
        must contain a root.

    Attributes
    ----------
    pca_ : PCA
        Fitted PCA model.

    components_ : ndarray of shape (n_components, n_features)
        Principal axes of the fitted PCA model, stored row-wise.

    explained_variance_ : ndarray of shape (n_components,)
        Variance explained by each principal component.

    explained_variance_ratio_ : ndarray of shape (n_components,)
        Fraction of total variance explained by each principal component.

    mean_ : ndarray of shape (n_features,)
        Per-feature mean of the training samples, used for PCA centering.

    scores_ : ndarray of shape (n_samples, n_components)
        Principal component scores of the training samples.

    h0_ : float
        Score-distance normalization factor. In the robust branches, this is the
        robust estimate of the mean score distance; in the classic branch, it is
        ``n_components / n_samples``.

    v0_ : float
        Orthogonal-distance normalization factor. In the robust branches, this is
        the robust estimate of the mean orthogonal distance; in the classic
        branch, it is the sample mean.

    Nh_ : int or float
        Positive, integer-valued degrees of freedom for the score-distance
        distribution.

    Nv_ : int or float
        Positive, integer-valued degrees of freedom for the orthogonal-distance
        distribution.

    training_distances_ : ndarray of shape (n_samples,)
        Combined normalized distances of the training samples, computed as
        ``Nh_ * h / h0_ + Nv_ * v / v0_`` for the triangular boundary. This
        array is populated only when ``boundary="triangle"``.

    thresholds_ : float or tuple of (float, float)
        Acceptance limits calibrated using ``alpha``. For "triangle", a
        scalar limit on the combined normalized distance. For "rectangle",
        a pair ``(h_threshold, v_threshold)`` on the original score and
        orthogonal distances, respectively. See Notes for the calibration.

    outlier_alpha_ : float
        Nominal per-sample outlier probability, calculated as
        ``1 - (1 - gamma) ** (1 / n_samples)``. Must be smaller than ``alpha``.

    outlier_thresholds_ : float or tuple of (float, float)
        Outlier limits calibrated using ``outlier_alpha_``, with the same
        structure and distance scales as ``thresholds_``.

    training_regular_ : ndarray of shape (n_samples,), dtype=bool
        Mask marking training samples inside or on the acceptance boundary.

    training_extremes_ : ndarray of shape (n_samples,), dtype=bool
        Mask marking training samples outside the acceptance boundary but
        inside or on the outlier boundary.

    training_outliers_ : ndarray of shape (n_samples,), dtype=bool
        Mask marking training samples strictly outside the outlier boundary.

    offset_ : float
        Offset relating normality and decision scores:
        ``decision_function(X) = score_samples(X) - offset_``. Set to -1.

    n_features_in_ : int
        Number of features seen during :meth:`fit`.

    feature_names_in_ : ndarray of shape (n_features_in_,)
        Names of features seen during :meth:`fit`. Defined only when the
        input has feature names that are all strings.

    See Also
    --------
    sklearn.decomposition.PCA : Principal component analysis.

    Notes
    -----
    The score distance ``h`` is the sum of squared PCA scores divided by
    the corresponding squared training singular values. The orthogonal
    distance ``v`` is the squared reconstruction error.

    For a per-sample rejection probability ``p``, the triangular boundary
    compares ``Nh_ * h / h0_ + Nv_ * v / v0_`` with the chi-square quantile
    at ``1 - p`` and ``Nh_ + Nv_`` degrees of freedom. The rectangular
    boundary compares ``h`` and ``v`` with separate scaled chi-square
    limits, each with marginal tail probability ``1 - sqrt(1 - p)``.
    This rectangular calibration assumes independent score and orthogonal
    distances.

    Acceptance limits use ``p = alpha``. Outlier limits use
    ``p = outlier_alpha_``, calibrated from ``gamma`` under an independent
    sample model. These probabilities are nominal under the fitted
    distributional assumptions, not guaranteed empirical rejection rates.

    The three training masks form a partition into regular, extreme, and
    outlier samples. Prediction is binary: regular samples receive +1,
    while both extreme and outlier samples receive -1. Scores and
    predictions use the acceptance limits, not the outlier limits.

    The PCA properties ``components_``, ``explained_variance_``,
    ``explained_variance_ratio_``, and ``mean_`` require a fitted estimator
    and expose the underlying PCA arrays without copying them.

    References
    ----------
    .. [1] Pomerantsev, A. L.
           "Acceptance areas for multivariate classification derived by
           projection methods." Journal of Chemometrics, 22(11-12), 601-609.

    .. [2] Pomerantsev, A. L., Rodionova, O., Kucheryavskiy, S.,
           "A comprehensive tutorial on Data-Driven SIMCA: Theory and
           implementation in web." Journal of Chemometrics, 38(7), 1-19.

    .. [3] Pomerantsev, A. L., Rodionova, O.,
           "Concept and role of extreme objects in PCA/SIMCA."
           Journal of Chemometrics, 28(5), 429-438.
    """

    _parameter_constraints: dict = {
        "n_components": [Interval(Integral, 1, None, closed="left")],
        "alpha": [Interval(Real, 0, 1, closed="neither")],
        "gamma": [Interval(Real, 0, 1, closed="neither")],
        "method": [StrOptions({"robust", "robust-fast", "classic"})],
        "boundary": [StrOptions({"triangle", "rectangle"})],
        "upper_bracket": [Interval(Real, 0, None, closed="neither")],
    }

    def __sklearn_tags__(self):
        """Return the scikit-learn estimator tags for this classifier."""
        tags = super().__sklearn_tags__()
        tags.estimator_type = "outlier_detector"
        tags.target_tags.required = False
        tags.target_tags.multi_output = False
        return tags

    def __init__(
        self,
        n_components: int = 2,
        alpha: float = 0.05,
        gamma: float = 0.01,
        method: Literal["robust", "robust-fast", "classic"] = "robust",
        boundary: Literal["triangle", "rectangle"] = "triangle",
        upper_bracket: float = 10_000,
    ):

        self.n_components = n_components
        self.alpha = alpha
        self.gamma = gamma
        self.method = method
        self.boundary = boundary
        self.upper_bracket = upper_bracket

    @property
    def components_(self):
        """Principal axes of the fitted PCA model.

        Returns
        -------
        components_ : ndarray of shape (n_components, n_features)
            Principal axes stored row-wise, as in ``pca_.components_``.
        """
        check_is_fitted(self, ["pca_", "thresholds_"])
        return self.pca_.components_

    @property
    def explained_variance_(self):
        """Variance explained by each principal component.

        Returns
        -------
        explained_variance_ : ndarray of shape (n_components,)
            Explained variances, as in ``pca_.explained_variance_``.
        """
        check_is_fitted(self, ["pca_", "thresholds_"])
        return self.pca_.explained_variance_

    @property
    def explained_variance_ratio_(self):
        """Fraction of total variance explained by each principal component.

        Returns
        -------
        explained_variance_ratio_ : ndarray of shape (n_components,)
            Explained variance ratios, as in ``pca_.explained_variance_ratio_``.
        """
        check_is_fitted(self, ["pca_", "thresholds_"])
        return self.pca_.explained_variance_ratio_

    @property
    def mean_(self):
        """Per-feature mean of the training samples.

        Returns
        -------
        mean_ : ndarray of shape (n_features,)
            Training mean used for PCA centering, as in ``pca_.mean_``.
        """
        check_is_fitted(self, ["pca_", "thresholds_"])
        return self.pca_.mean_

    def fit(self, X, y=None):
        """Fit the PCA model, estimate boundaries, and categorize training samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training samples from the target class. At least two samples
            are required.

        y : Ignored, default=None
            Not used, present for API consistency by convention.

        Returns
        -------
        self : OneClassPCA
            Fitted estimator with acceptance and outlier limits and training
            category masks.

        Raises
        ------
        ValueError
            If input or parameters are invalid, retained PCA components are
            numerically degenerate, or ``outlier_alpha_ >= alpha``. Also
            raised if the robust root-search bracket does not bracket a root,
            or robust-fast inputs are invalid, exceed the approximation's
            supported IQR-to-median ratio, or produce a nonfinite estimate.
        RuntimeError
            If the root solver fails to converge when ``method="robust"``.

        Warns
        -----
        UserWarning
            If ``method="robust-fast"`` estimates rounded degrees of freedom
            of at least 50, outside the approximation's reported accuracy
            range. The estimate is retained; no method switch is performed.

        Notes
        -----
        In the classic method, mean distances are ``n_components / n_samples``
        for score distance and the sample mean for orthogonal distance.
        Degrees of freedom are twice the squared ratio of each mean to its
        sample standard deviation, rounded to the nearest integer with a
        minimum of 1.

        Both robust methods use medians and interquartile ranges. The
        "robust" method solves the chi-square IQR-to-median matching equation;
        "robust-fast" uses its explicit approximation. Both round degrees of
        freedom to the nearest integer with a minimum of 1, then estimate mean
        distances using Equation (17) in reference [3] of the class docstring.

        For either boundary geometry, acceptance limits use ``alpha`` and
        outlier limits use ``1 - (1 - gamma) ** (1 / n_samples)``. The latter
        must be smaller than ``alpha`` so the outlier boundary lies outside
        the acceptance boundary. See the class Notes for threshold scales,
        probability assumptions, and category definitions.
        """
        # 1. Validate input parameters
        self._validate_params()

        # 2. Check that X is a 2D array and has only finite values
        X = validate_data(
            self,
            X,
            y="no_validation",
            ensure_min_samples=self.n_components + 2,
            ensure_min_features=self.n_components + 1,
            ensure_2d=True,
            reset=True,
            dtype=np.float64,
        )

        # 3. Get dimensions
        n_samples, _ = X.shape

        # 4. Fit the PCA model to the training data
        self.pca_ = PCA(n_components=self.n_components)
        self.scores_ = self.pca_.fit_transform(X)  # Eq. (6) in [1]

        # 4.1. Check that retained singular values are positive and finite
        s = self.pca_.singular_values_
        _check_positive_finite(s, "Retained singular values")

        rtol = np.finfo(X.dtype).eps * max(X.shape)

        if not np.isfinite(s).all() or s[-1] <= rtol * s[0]:
            raise ValueError(
                "Retained components are numerically degenerate. Reduce n_components."
            )

        # 5. Calculate distances
        # 5.1. Calculate score distance (SD, h)
        T = self.scores_
        eigval = self.pca_.explained_variance_

        # The eigenvalues are scaled because scikit-learn calculates XtX it using
        # Bessel correction.
        h = np.sum(T**2 / ((n_samples - 1) * eigval), axis=1)  # Eq. (8) in [1]

        # 5.2. Calculate orthogonal distance (OD, v)
        residuals = X - self.pca_.inverse_transform(T)  # Eq. (16) in [1]
        v = np.sum(residuals**2, axis=1)

        # 5.3 Calculate statistics for SD and OD
        if self.method == "classic":
            # Mean  distances (SD and OD)
            self.h0_ = self.n_components / n_samples  # Eq. (9) in [1]
            self.v0_ = v.mean()  # Eq. (18) in [1]

            # Distances standard deviations (SD and OD)
            h_std_ = np.std(h, ddof=1)  # Eq. (13) in [3]
            v_std_ = np.std(v, ddof=1)  # Eq. (13) in [3]

            # Moment based DoF rounded to the nearest positive int (Eq. (12) in [3])
            self.Nh_ = max(1, int(np.rint(2 * (self.h0_ / h_std_) ** 2)))
            self.Nv_ = max(1, int(np.rint(2 * (self.v0_ / v_std_) ** 2)))

        else:
            # Median calculation (Eq. (14) in [3])
            median_h = np.median(h)
            median_v0 = np.median(v)

            # Interquantile range calculation (Eq. (14) in [3])
            iqr_h = np.percentile(h, 75) - np.percentile(h, 25)
            iqr_v = np.percentile(v, 75) - np.percentile(v, 25)
            if self.method == "robust-fast":
                # Fast robust estimation of DoF for SD and OD using Eq. (16) in [3] and
                # rounding to the closest integeer as suggested in [3]
                self.Nh_ = self._estimate_robust_fast_dof(median_h, iqr_h)
                self.Nv_ = self._estimate_robust_fast_dof(median_v0, iqr_v)

            if self.method == "robust":
                # Robust estimation of DoF for SD and OD solving Eq. (15) in [3] using
                # Berntq's method in the interval (1, self.upper_bracket)
                self.Nh_ = self._estimate_robust_dof(
                    median_h, iqr_h, bracket=(1e-3, self.upper_bracket)
                )
                self.Nv_ = self._estimate_robust_dof(
                    median_v0, iqr_v, bracket=(1e-3, self.upper_bracket)
                )

            # Estimate the sample means using Eq. (17) in [3]
            self.h0_ = self._estimate_robust_distance_mean(median_h, iqr_h, self.Nh_)
            self.v0_ = self._estimate_robust_distance_mean(median_v0, iqr_v, self.Nv_)

        # 6. Calculate acceptance and outlier boundaries.
        self.outlier_alpha_ = -np.expm1(np.log1p(-self.gamma) / n_samples)

        # The outlier boundary must lie outside the acceptance boundary
        # to define regular, extreme, and outlier regions.
        if self.outlier_alpha_ >= self.alpha:
            raise ValueError(
                "gamma must give a per-sample outlier probability smaller than alpha."
            )

        self.thresholds_ = self._calculate_thresholds(self.alpha)
        self.outlier_thresholds_ = self._calculate_thresholds(self.outlier_alpha_)

        # 7. Identify training samples outside each boundary.
        if self.boundary == "triangle":
            self.training_distances_ = (
                self.Nh_ * h / self.h0_ + self.Nv_ * v / self.v0_
            )  # Eq. (18) in [3]

            outside_acceptance = (
                self.training_distances_ > self.thresholds_
            )  # Eq. (19) in [3]
            self.training_outliers_ = (
                self.training_distances_ > self.outlier_thresholds_
            )  # Eq. (20) in [3]

        else:  # rectangle
            h_threshold, v_threshold = self.thresholds_
            h_outlier_threshold, v_outlier_threshold = self.outlier_thresholds_

            # Eq. (25) in [1]
            outside_acceptance = (h > h_threshold) | (v > v_threshold)
            self.training_outliers_ = (h > h_outlier_threshold) | (
                v > v_outlier_threshold
            )

        # Eq. (20) in [3]
        self.training_regular_ = ~outside_acceptance
        self.training_extremes_ = outside_acceptance & ~self.training_outliers_

        # 8. Set decision offset.
        # API convention for implementing the selected acceptance rule
        self.offset_ = -1

        return self

    def predict(self, X, y=None) -> np.ndarray[int]:
        """Classify samples as accepted or rejected by the fitted model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to classify using the fitted model.

        y : Ignored, default=None
            Not used, present for API consistency by convention.

        Returns
        -------
        labels : ndarray of shape (n_samples,), dtype=int
            +1 for samples inside or on the acceptance boundary and -1 for
            samples outside it, including both extremes and outliers.

        Raises
        ------
        NotFittedError
            If the estimator has not been fitted.

        Notes
        -----
        Labels use the acceptance limits for the selected boundary geometry,
        not the outlier limits. The three-way categorization is recorded
        separately for training samples during :meth:`fit`.
        """
        check_is_fitted(self, ["pca_", "thresholds_"])
        X = validate_data(
            self, X, y="no_validation", ensure_2d=True, reset=False, dtype=np.float64
        )
        score = self.decision_function(X)

        return np.where(score < 0, -1, 1)

    def score_samples(self, X) -> np.ndarray[float]:
        """Compute a normality score for each sample.

        Scores are negative normalized distance measures. Higher values
        indicate more normal observations. The acceptance boundary corresponds
        to a score of -1.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to evaluate using the fitted model.

        Returns
        -------
        scores : ndarray of shape (n_samples,)
            Negative normalized distances. Values greater than or equal to
            -1 indicate acceptance; values below -1 indicate rejection.

        Raises
        ------
        NotFittedError
            If the estimator has not been fitted.

        Notes
        -----
        For the triangular boundary, the combined distance is divided by its
        threshold. For the rectangular boundary, the maximum of the two
        distance-to-threshold ratios is used. The resulting measure is negated.

        Scores use ``thresholds_``, not ``outlier_thresholds_``. They are not
        probabilities of class membership, and their magnitudes need not be
        comparable across boundary geometries.
        """
        check_is_fitted(self, ["pca_", "thresholds_"])

        h, v = self._compute_distances(X)

        if self.boundary == "triangle":
            # Combines scaled distance with Nh and Nv DoF (Eq. (18) in [3])
            combined_distance = self.Nh_ * h / self.h0_ + self.Nv_ * v / self.v0_
            distance = (
                -combined_distance / self.thresholds_
            )  # Convention for scoring API

        if self.boundary == "rectangle":
            # Scalar representation of the rectangular rule
            # Region supported by Eq. (25) in [1]
            h_threshold, v_threshold = self.thresholds_
            distance = -np.maximum(
                h / h_threshold,
                v / v_threshold,
            )

        return distance

    def decision_function(self, X) -> np.ndarray:
        """Compute signed scores relative to the acceptance boundary.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to evaluate using the fitted model.

        Returns
        -------
        decision : ndarray of shape (n_samples,)
            Signed scores. Positive values indicate samples inside the
            acceptance region, zero indicates the boundary, and negative
            values indicate samples outside the acceptance region.

        Raises
        ------
        NotFittedError
            If the estimator has not been fitted.

        Notes
        -----
        Calculated as ``score_samples(X) - offset_``, where ``offset_ = -1``
        for either boundary geometry. Both extreme and outlier samples have
        negative scores; the outlier limits do not affect this calculation.
        """
        check_is_fitted(self, ["pca_", "thresholds_"])
        return self.score_samples(X) - self.offset_

    def _compute_distances(self, X) -> tuple:
        """Compute score and orthogonal distances from the fitted PCA model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to project onto the fitted PCA model.

        Returns
        -------
        h : ndarray of shape (n_samples,)
            Score distances: sums of squared component scores divided by
            the corresponding squared training singular values.
        v : ndarray of shape (n_samples,)
            Orthogonal distances: sums of squared reconstruction residuals.
        """
        # Calculate the scores onto the model
        scores = self.pca_.transform(X)

        # Calculate the score distances, Eq. (8) in [1]
        h = np.sum(scores**2 / self.pca_.singular_values_**2, axis=1)

        # Calculate the orthogonal distances, Eq. (16) in [1]
        residuals = X - self.pca_.inverse_transform(scores)
        v = np.sum(residuals**2, axis=1)

        return h, v

    def _estimate_robust_dof(
        sef, median_u: float, iqr_u: float, bracket=(1e-12, 10_000)
    ) -> float:
        """Estimate degrees of freedom by chi-square IQR-to-median matching.

        Solve Equation (15) in reference [3] of the class docstring using
        Brent's method, then round the root to the nearest integer with a
        minimum of 1.

        Parameters
        ----------
        median_u : float
            Median of the observed distances.
        iqr_u : float
            Interquartile range of the observed distances.
        bracket : tuple of (float, float), default=(1, 10_000)
            Root-search endpoints. The interval must bracket a root.

        Returns
        -------
        dof : int or float
            Positive, integer-valued estimate of the degrees of freedom.

        Raises
        ------
        ValueError
            If the objective is invalid at the bracket endpoints or the
            bracket does not bracket a root.
        RuntimeError
            If Brent's method fails to converge.
        """

        def objective(dof):
            q25, q50, q75 = chi2.ppf([0.25, 0.5, 0.75], df=dof)
            return (q75 - q25) / q50 - iqr_u / median_u

        return max(1, np.round(brentq(objective, *bracket)))

    def _estimate_robust_fast_dof(self, median_u: float, iqr_u: float) -> float:
        """Approximate degrees of freedom from the observed IQR-to-median ratio.

        Use Equation (16) in reference [3] of the class docstring, then round
        to the nearest integer with a minimum of 1.

        Parameters
        ----------
        median_u : float
            Finite, strictly positive median of the observed distances.
        iqr_u : float
            Finite, strictly positive interquartile range of the observed
            distances.

        Returns
        -------
        dof : int
            Positive estimate of the degrees of freedom.

        Raises
        ------
        ValueError
            If inputs are nonfinite or not strictly positive,
            ``iqr_u / median_u`` exceeds the supported maximum of 2.68631,
            or the estimated degrees of freedom are nonfinite.

        Warns
        -----
        UserWarning
            If the rounded estimate is at least 50, outside the reported
            accuracy range. The estimate is retained.
        """
        # Check for valid median_u and iqr_u values
        if not (
            np.isfinite(median_u) and median_u > 0 and np.isfinite(iqr_u) and iqr_u > 0
        ):
            raise ValueError(
                "Median_U and IQR_U must be finite and strictly positive, but found "
                f"median_u={median_u}, iqr_u={iqr_u}"
            )

        # Check the log-term for invalid calculation
        log_term = np.log(CONSTANT_B) + np.log(median_u) - np.log(iqr_u)

        if log_term < 0:
            warnings.warn(
                "Cannot estimate robust degrees of freedom using Equation (16): "
                f"IQR / median = {iqr_u / median_u:.6g} exceeds "
                f"the approximation's supported maximum of {CONSTANT_B}. "
                "Setting DoF to 1.",
                UserWarning,
                stacklevel=2,
            )
            return 1

        # Calculate degrees of freedom
        dof = np.exp((1 / CONSTANT_A * log_term) ** (1 / CONSTANT_C))

        # Check for valdid degrees of freedom
        if not np.isfinite(dof):
            raise ValueError(
                f"The robust-fast DoF estimate is not finite, but found dof={dof}"
            )

        # Round DoF to the nearest positive integer as defined in [3]
        rounded_dof = max(1, int(np.rint(dof)))

        # Warn user if estimated DoF is above reported approximation accuracy.
        if rounded_dof >= 50:
            warnings.warn(
                f"The robust-fast DoF estimate rounds to {rounded_dof:.0f}. "
                "Eq. (16) in [3] is being used outside its reported N < 50 range. "
                "The estimate is retained, but approximation accuracy "
                "outside this range is not established.",
                UserWarning,
                stacklevel=2,
            )

        return rounded_dof

    def _estimate_robust_distance_mean(
        self, median_u: float, iqr_u: float, N: float
    ) -> float:
        """Estimate the mean distance from robust summary statistics.

        Use Equation (17) in reference [3] of the class docstring.

        Parameters
        ----------
        median_u : float
            Median of the observed distances.
        iqr_u : float
            Interquartile range of the observed distances.
        N : float
            Estimated degrees of freedom for the distance distribution.

        Returns
        -------
        mean_u : float
            Robust estimate of the distance mean.
        """
        return (
            0.5
            * N
            * (
                median_u / chi2.ppf(0.5, df=N)
                + iqr_u / (chi2.ppf(0.75, df=N) - chi2.ppf(0.25, df=N))
            )
        )

    def _calculate_thresholds(self, tail_probability: float):
        """Calculate limits for the selected boundary geometry.

        Parameters
        ----------
        tail_probability : float
            Nominal per-sample rejection probability, strictly between 0
            and 1. Use ``alpha`` for acceptance limits or ``outlier_alpha_``
            for outlier limits.

        Returns
        -------
        thresholds : float or tuple of (float, float)
            For "triangle", a chi-square quantile with ``Nh_ + Nv_`` degrees
            of freedom on the combined normalized distance scale. For
            "rectangle", a pair of scaled chi-square limits on the original
            score and orthogonal distances, in that order.

        Notes
        -----
        Rectangular limits use equal marginal tail probabilities
        ``1 - sqrt(1 - tail_probability)``. This yields the requested joint
        rejection probability under independent score and orthogonal
        distance models.
        """
        if self.boundary == "triangle":
            # Eq. (4)-(5) in [2]
            return chi2.isf(tail_probability, df=self.Nh_ + self.Nv_)

        # Separate limits with equal marginal rejection probabilities,
        # assuming independent score and orthogonal distances.
        # Eq. (25) in [1]
        marginal_tail = -np.expm1(np.log1p(-tail_probability) / 2)

        # Separated chi-square limits for rectangular regions (Eq. (25) in [1])
        h_threshold = self.h0_ / self.Nh_ * chi2.isf(marginal_tail, df=self.Nh_)
        v_threshold = self.v0_ / self.Nv_ * chi2.isf(marginal_tail, df=self.Nv_)

        return h_threshold, v_threshold
