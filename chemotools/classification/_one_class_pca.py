"""One-class classification based on Principal Component Analysis (PCA).

The module :mod:`chemotools.classification._one_class_pca` implements
OneClassPCA, which models a target class using PCA and estimates an acceptance
limit from the training samples' score and orthogonal distances.
"""

# Author: Pau Cabaneros
# License: MIT

from numbers import Integral, Real
from typing import Literal

import numpy as np
from scipy.stats import chi2
from sklearn.base import BaseEstimator, OutlierMixin
from sklearn.decomposition import PCA
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.validation import check_is_fitted, validate_data

from chemotools._doc_mixin import DocLinkMixin


class OneClassPCA(DocLinkMixin, OutlierMixin, BaseEstimator):
    """One-class PCA (Principal Component Analysis) classifier.

    Fit a target-class PCA model and estimate a triangular acceptance region
    from the combined normalized score and orthogonal distances [1]_.
    Distributional parameters are estimated from the training data [2]_.

    Parameters
    ----------
    n_components : int, default=2
        Number of principal components used to model the target class.

    threshold : float, default=0.95
        Confidence level used to determine the acceptance limits.

    method : {"robust", "classic"}, default="robust"
        How the distributional parameters are estimated:

        - "robust": use medians and median-normalized interquartile ranges.
        - "classic": use mean distances and sample standard deviations.

    boundary : {"triangle", "rectangle"}, default="triangle"
        Acceptance boundary selection:

        - "triangle": combine score and orthogonal distances into one
          acceptance criterion (boundary II in Figure 2 and Equations
          (26)-(27) in [1]_).
        - "rectangle": intended to apply separate limits to the two distances
          and reject a sample if either limit is exceeded (boundary I in
          Figure 2 and Equations (24)-(25) in [1]_).

        Currently ignored: fitting always computes the triangular boundary.

    Attributes
    ----------
    pca_ : PCA
        Fitted PCA model.

    scores_ : ndarray of shape (n_samples, n_components)
        Principal component scores of the training samples.

    h0_ : float
        Score distance normalization factor. The median score distance for
        robust estimation, or ``n_components / n_samples`` in the classic
        estimation branch.

    v0_ : float
        Orthogonal distance normalization factor. The median orthogonal
        distance for robust estimation, or the sample mean in the classic
        estimation branch.

    Nh_ : float
        Estimated degrees of freedom for the score distance distribution.

    Nv_ : float
        Estimated degrees of freedom for the orthogonal distance distribution.

    training_distances_ : ndarray of shape (n_samples,)
        Combined normalized distances of the training samples, computed as
        ``Nh_ * h / h0_ + Nv_ * v / v0_``, where ``h`` and ``v`` are the
        score and orthogonal distances, respectively.

    threshold_ : float
        Chi-square quantile at ``threshold`` with ``Nh_ + Nv_`` degrees of
        freedom, used as the combined-distance acceptance limit.

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
    The ``boundary`` parameter does not currently affect the calculation;
    only the triangular acceptance boundary is implemented. Prediction and
    decision-function methods are not implemented.

    References
    ----------
    .. [1] Pomerantsev, A. L.
           "Acceptance areas for multivariate classification derived by
           projection methods." Journal of Chemometrics, 22(11-12), 601-609.

    .. [2] Pomerantsev, A. L., Rodionova, O., Kucheryavskiy, S.,
           "A comprehensive tutorial on Data-Driven SIMCA: Theory and
           implementation in web." Journal of Chemometrics, 38(7), 1-19.
    """

    _parameter_constraints: dict = {
        "n_components": [Interval(Integral, 1, None, closed="left")],
        "threshold": [Interval(Real, 0, None, closed="neither")],
        "method": [StrOptions["robust", "classic"]],
        "boundary": [StrOptions["triangle", "rectangle"]],
    }

    def __sklearn_tags__(self):
        """Return the scikit-learn estimator tags for this classifier."""
        tags = super().__sklearn_tags__()
        tags.multioutput = False
        tags.requires_y = False
        return tags

    def __init__(
        self,
        n_components: int = 2,
        threshold: float = 0.95,
        method: Literal["robust", "classic"] = "robust",
        boundary: Literal["triangle", "rectangle"] = "triangle",
    ):

        self.n_components = n_components
        self.threshold = threshold
        self.method = method
        self.boundary = boundary

    @property
    def components_(self):
        """Principal components of the fitted PCA model.

        Returns
        -------
        components_ : ndarray of shape (n_components, n_features)
            Principal components of the fitted PCA model.
        """
        check_is_fitted(self, ["pca_", "threshold_"])
        return self.pca_.components_

    @property
    def explained_variance_(self):
        """Explained variance of each component in the fitted PCA model.

        Returns
        -------
        explained_variance_ : ndarray of shape (n_components)
            Explained variance of the fitted PCA model.
        """
        check_is_fitted(self, ["pca_", "threshold_"])
        return self.pca_.explained_variance_

    @property
    def explained_variance_ratio_(self):
        """Explained variance ratio of each component in the fitted PCA model.

        Returns
        -------
        explained_variance_ratio_ : ndarray of shape (n_components)
            Explained variance ratio of the fitted PCA model.
        """
        check_is_fitted(self, ["pca_", "threshold_"])
        return self.pca_.explained_variance_ratio_

    def fit(self, X, y=None):
        """Fit the PCA model and estimate the acceptance boundary.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors.

        y : Ignored, default=None
            Not used, present for API consistency by convention.

        Returns
        -------
        self : OneClassPCA
            Fitted estimator.

        Notes
        -----
        The score and orthogonal distances are modeled using scaled chi-square
        distributions. For ``method="robust"``, the normalization factors
        are the median distances, and the effective degrees of freedom are
        estimated from median-normalized interquartile ranges using the
        approximation in Equation (15) of Pomerantsev [1]_.

        The classic estimation branch uses ``n_components / n_samples``
        for the score distance normalization factor and the sample mean
        for the orthogonal distance normalization factor. The effective
        degrees of freedom are estimated as twice the squared ratio of
        each normalization factor to its sample standard deviation
        (Section 3.2 in [2]_).

        The triangular acceptance region is constructed from the combined
        normalized distances, using a chi-square quantile at the specified
        confidence level (Equations (26)-(27) in [1]_), regardless of
        ``boundary``.

        References
        ----------
        .. [1] Pomerantsev, A. L.
               "Acceptance areas for multivariate classification derived by
               projection methods." Journal of Chemometrics, 22(11-12), 601-609.

        .. [2] Pomerantsev, A. L., Rodionova, O., Kucheryavskiy, S.,
               "A comprehensive tutorial on Data-Driven SIMCA: Theory and
               implementation in web." Journal of Chemometrics, 38(7), 1-19.
        """
        # 1. Validate input parameters
        self._validate_params()

        # 2. Check that X is a 2D array and has only finite values
        X = validate_data(
            self, X, y="no_validation", ensure_2d=True, reset=True, dtype=np.float64
        )

        # 3. Get dimensions
        n_samples, _ = X.shape

        # 4. Fit the PCA model to the training data
        self.pca_ = PCA(n_components=self.n_components)
        self.scores_ = self.pca_.fit_transform(X)

        # 5. Calculate distances
        # 5.1. Calculate score distance (SD, h)
        T = self.scores_
        eigval = self.pca_.explained_variance_

        # The eigenvalues are scaled because scikit-learn calculates XtX it using
        # Bessel correction.
        h = np.sum(T**2 / ((n_samples - 1) * eigval), axis=1)  # Equation (8) in [1]

        # 5.2. Calculate orthogonal distance (OD, v)
        residuals = X - self.pca_.inverse_transform(T)
        v = np.sum(residuals**2, axis=1)

        # 5.3 Calculate Degrees of Freedom for SD and OD
        if self.method == "classic":
            # Distances means
            self.h0_ = self.n_components / n_samples  # SD, Equation (9) in [1]
            self.v0_ = v.mean()  # OD, Equation (9) in [1]

            # Distances standard deviations
            h_std_ = np.std(h, ddof=1)  # STD of SD (section 3.2 in [2])
            v_std_ = np.std(v, ddof=1)  # STD of OD (section 3.2 in [2])

            # Calculate Degrees of Freedom for SD and OD (section 3.2 in [2])
            self.Nh_ = 2 * (self.h0_ / h_std_) ** 2  # Section 3.2 in [2]
            self.Nv_ = 2 * (self.v0_ / v_std_) ** 2  # Section 3.2 in [2]

        if self.method == "robust":
            # Use meadian for robust estimation as suggested in Section 3.3 in [2]
            self.h0_ = np.median(h)
            self.v0_ = np.median(v)

            # Use the interquartile range for robust estimation Equation (14) in [1]
            relative_iqr_h = (np.percentile(h, 75) - np.percentile(h, 25)) / self.h0_
            relative_iqr_v = (np.percentile(v, 75) - np.percentile(v, 25)) / self.v0_

            # Estimate the Degrees of Freedom for SD and OD using the relative IQR
            # (Equation (15) in [1])
            self.Nh_ = np.exp(4.36 * np.log(1.24 / relative_iqr_h)) ** 0.72
            self.Nv_ = np.exp(4.36 * np.log(1.24 / relative_iqr_v)) ** 0.72

        # 6. Calculate boundaries
            # 6.1. Triangle boundary (Boundary II, Figure 2 in [1])
            if self.boundary == "triangle":
                # 6.1.1. Combined distance Equations (26-27) in [1]
                self.training_distances_ = (
                    self.Nh_ * h / self.h0_ + self.Nv_ * v / self.v0_
                ) 

                # 6.1.2. Decision boundary
                self.thresholds_ = (chi2.ppf(self.threshold, df=self.Nh_ + self.Nv_))

            # 6.2. Rectabgle boundary (Boundary I, Figure 2 in [1])
            if self.boundary == "rectangle":
                # TODO: writing this from memory, verify scalings with [1]
                # 6.2.1. Calculate threshold for SD
                h_threshold = chi2.ppf(self.threshold, df=self.Nh_)

                # 6.2.2. Calculate threshold for OD
                v_threshold = chi2.ppf(self.threshold, df=self.Nv_)

                self.thresholds_ = (h_threshold, v_threshold)

        return self
