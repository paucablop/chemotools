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

        # 3. Fit the PCA model to the training data
        self.pca_ = PCA(n_components=self.n_components)
        self.scores_ = self.pca_.fit_transform(X)

        # 4. Calculate distances
        # 4.1. Calculate score distance (SD, h)
        T = self.scores_
        eigval = self.pca_.explained_variance_
        # The eigenvalues are scaled because scikit-learn calculates XtX it using
        # Bessel correction.
        h = np.sum(T**2 / (T.shape[0] - 1) * eigval, axis=1)  # Equation 8 in [1]
        self.h0_ = np.mean(h, axis=0)  # Equation 9 in [1]

        # 4.2. Calculate orthogonal distance (OD, v)
        v = np.einsum("ij,ij->i", X, X) - np.einsum(
            "ij,ij->i", T, T
        )  # Equation 17 in [1]
        self.v0_ = np.mean(v, axis=0)  # Equation 18 in [1]

        # 5. Calculate decision boundaries
        # 5.1. Calculate DoF for SD distribution
        iqr_h = np.percentile(h, 75) - np.percentile(h, 25)
        self.Nh_ = np.exp(4.36 * np.ln(1.24 / iqr_h)) ** 0.72  # Equation 15 in [1]

        # 5.3. Calculate the DoF for OD distribution
        iqr_v = np.percentile(v, 75) - np.percentile(v, 25)
        self.Nv_ = (
            np.exp(4.36 * np.ln(1.24 / iqr_v)) ** 0.72
        )  # As suggested from Equations 14-15 in [1]

        # 5.4 Decision boundary
        self.threshold_ = self.Nh_ * h/self.h0_ + self.Nv_ * v/self.h0_

        return self
