"""
The module :mod:chemotools.classification._one_class_pca implements OneClassPCA, a
one-class classifier based on Principal Component Analysis (PCA). It models the target
class and classifies new samples based on their score and orthogonal distances to the
PCA model, using acceptance limits defined by either a parametric or a data-driven
approach.
"""

# Author: Pau Cabaneros
# License: MIT

from numbers import Integral, Real
from typing import Literal

import numpy as np
from scipy.stats import chi2
from sklearn.base import BaseEstimator, OutlierMixin
from sklearn.decomposition import PCA
from sklearn.utils._param_validation import Interval
from sklearn.utils.validation import validate_data

from chemotools._doc_mixin import DocLinkMixin


class OneClassPCA(DocLinkMixin, OutlierMixin, BaseEstimator):
    """
    One-Class PCA (Principal Component Analysis) classifier.

    Models a target class using PCA and identifies outliers based on
    score and orthogonal distances to the PCA model. Acceptance limits
    are determined using either a parametric or a data-driven approach.

    Parameters
    ----------
    n_components : int, default=2
        Number of principal components used to model the target class.

    threshold : float, default=0.95
        Confidence level used to determine the acceptance limits.

    boundary : {"parametric", "data-driven"}, default="data-driven"
        Method used to determine the acceptance limits. The parametric
        approach uses statistical distribution assumptions, while the
        data-driven approach estimates the limits from the training data.

    References
    ----------
    .. [1] Pomerantsev, A. L.
        Acceptance areas for multivariate classification derived by projection methods
        Journal of Chemometrics, 22(11-12), 601-609.
    """

    _parameter_constraints: dict = {
        "n_components": [Interval(Integral, 1, None, closed="left")],
        "threshold": [Interval(Real, 0, None, closed="neither")],
        "boundary": [Literal["parametric", "data-driven"]],
    }

    def __sklearn_tags__(self):
        """
        Return the scikit-learn estimator tags for this One-Class PCA classifier.
        """
        tags = super().__sklearn_tags__()
        tags.multioutput = False
        tags.requires_y = False
        return tags

    def __init__(
        self,
        n_components: int = 2,
        threshold: float = 0.95,
        boundary: Literal["parametric", "data-driven"] = "data-driven",
    ):

        self.n_components = n_components
        self.threshold = threshold
        self.boundary = boundary

    def fit(self, X, y=None):
        """
        Fit the One-Class PCA model to the training data and calculates the decision
        boundaries based on the data-driven method-of-moments approach suggested by
        Pomerantsev [1].

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
            Fitted estimator.
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
        h = np.sum(T**2 / ((n_samples - 1) * eigval), axis=1)  # Equation 8 in [1]
        self.h0_ = self.n_components / n_samples  # Equation 9 in [1]

        # 5.2. Calculate orthogonal distance (OD, v)
        residuals = X - self.pca_.inverse_transform(T)
        v = np.sum(residuals**2, axis=1)
        self.v0 = v.mean()

        # 6. Calculate decision boundaries
        # 6.1. Calculate relative IQR for SD and OD
        relative_iqr_h = (
            np.percentile(h, 75) - np.percentile(h, 25)
        ) / self.h0_  # Equation 14 in [1]
        relative_iqr_v = (
            np.percentile(v, 75) - np.percentile(v, 25) / self.v0_
        )  # From Equation 14 in [1]

        # 6.2. Estimate the DoF for SD and OD
        self.Nh_ = (
            np.exp(4.36 * np.ln(1.24 / relative_iqr_h)) ** 0.72
        )  # Equation 15 in [1]

        # 6.3. Calculate the DoF for OD distribution
        self.Nv_ = (
            np.exp(4.36 * np.ln(1.24 / relative_iqr_v)) ** 0.72
        )  # As suggested from Equations 14-15 in [1]

        # 6.4. Combined distance (Boundary II, Figure 2 in [1])
        self.training_distances_ = (
            self.Nh_ * h / self.h0_ + self.Nv_ * v / self.v0_
        )  # Equations 26-27 in [1]

        # 6.5 Decision boundary
        self.threshold_ = chi2.ppf(self.threshold, self.Nh + self.Nv)

        return self
