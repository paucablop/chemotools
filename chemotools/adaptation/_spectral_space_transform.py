"""
The :mod: `chemotools.adaptation._spectral_space_transform`
module implements the Spectral Space Transform (SST) transformer
"""

# Authors: Ruggero Guerrini
# License: MIT

import warnings
from numbers import Integral

import numpy as np
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin
from sklearn.utils._param_validation import Interval
from sklearn.utils.validation import (
    check_is_fitted,
    validate_data,
)

from chemotools._doc_mixin import DocLinkMixin


class SpectralSpaceTransform(
    DocLinkMixin, OneToOneFeatureMixin, TransformerMixin, BaseEstimator
):
    """
    Spectral Space Transform (SST) is a transformer used for domain adaptation
    (calibration) applications.
    SST is a linear transformation method used to transfer spectral data from
    a target domain to a source (reference) domain, allowing calibration
    models to remain valid across different instruments or acquisition
    conditions, following the implementation by [1]_.
    The method constructs a shared latent space using singular value
    decomposition (SVD) of concatenated source and target data, and derives
    projection matrices to align the target data to the source space.

    Parameters
    ----------
    n_components : int, default=2
        Number of latent components retained from the singular value
        decomposition (SVD). Controls the dimensionality of the shared subspace.


    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during

    p1_ : ndarray of shape (n_components, n_features), or None
        Projection matrix associated with the source domain. It maps the
        shared latent space back to the source spectral space.

    p2_ : ndarray of shape (n_components, n_features), or None
        Projection matrix associated with the target domain. It maps target
        data into the shared latent space.

    x_source_provided_ : bool
        Boolean flag indicating if X_source was provided during fitting.

    References
    ---------
    [1] Du, W., Chen, Z.-P., Zhong, L.-J.,
        Wang, S.-X., Yu, R.-Q., Nordon, A.,
        Littlejohn, D., & Holden, M. (2011).
        Maintaining the predictive abilities
        of multivariate calibration models by
        spectral space transformation.
        Analytica Chimica Acta, 690(1), 64–70.
        https://doi.org/10.1016/j.aca.2011.02.014

    See Also
    --------
    PiecewiseDirectStandardization : Local standardization using moving windows.
    DirectStandardization : Global linear transformation without local windows.

    Examples
    --------
    **Basic usage**
    >>> import numpy as np
    >>> from chemotools.adaptation import SpectralSpaceTransform
    >>>
    >>> rng = np.random.default_rng(17)
    >>> X_source = rng.normal(size=(100, 20))
    >>> X_target = X_source * 2 - rng.normal(size=(100, 20)) * 0.02
    >>>
    >>> sst = SpectralSpaceTransform(n_components=2).fit(X_target, X_source=X_source)
    >>> X_transf = sst.transform(X_target)

    """

    # Fitted attributes (set during fit, typed for type checkers)
    n_features_in_: int
    p1_: np.ndarray | None
    p2_: np.ndarray | None
    x_source_provided_: bool

    _parameter_constraints = {
        "n_components": [Interval(Integral, 1, None, closed="left")],
    }

    def __init__(self, n_components: int = 2):
        self.n_components = n_components

    def fit(
        self, X: np.ndarray, y=None, *, X_source: np.ndarray | None = None
    ) -> "SpectralSpaceTransform":
        """
        Fit the SpectralSpaceTransform model.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Data from the target instrument.

        y : None
            Ignored to align with API.

        X_source : np.ndarray of shape (n_samples, n_features), optional
            Data from the source instrument. If None, the transformer defaults to
            an identity transformation.

        Returns
        -------
        self : SpectralSpaceTransform
        """
        # Validate the input parameters
        # Check that X is a 2D array and has only finite values
        X = validate_data(self, X, ensure_2d=True, reset=True, dtype=np.float64)

        # If X_source is None, default to identity transformation
        if X_source is None:
            warnings.warn(
                "X_source is None, the transformer will act as an identity "
                "transformation."
            )
            self.x_source_provided_ = False
            self.n_features_in_ = X.shape[1]
            self.p1_ = None
            self.p2_ = None
            return self

        # Check that X_source is a 2D array and has only finite values
        X_source = validate_data(
            self, X_source, ensure_2d=True, reset=False, dtype=np.float64
        )

        # Check consistency between X and X_source
        if X_source.shape != X.shape:
            raise ValueError(
                f"X and X_source must have the same shape, "
                f"got X={X.shape} and X_source={X_source.shape}."
            )

        x = np.hstack([X_source, X])

        # Validate that n_components does not exceed the rank of X ( [X_source, X] )
        # after centering.
        # Centering reduces the effective rank in the sample direction by 1, so the
        # maximum number of meaningful components is min(n_samples - 1, 2*n_features).
        max_components = min(x.shape[0] - 1, x.shape[1])
        if self.n_components > max_components:
            raise ValueError(
                f"n_components={self.n_components} is too large. "
                f"After mean-centering, the effective rank of X is at most "
                f"min(n_samples - 1, 2*n_features) = {max_components}. "
                f"Set n_components to a value <= {max_components}."
            )
        # Compute the SVD of the joint matrix x = [X_source | X].
        u, s, vh = np.linalg.svd(x, full_matrices=False)
        # Transpose vh to get V of shape (2*n_features, n_components)
        v = vh.T
        # n_col_ref corresponds to the number of features (for both Source and Target)
        n_col_ref = X_source.shape[1]
        # p1_: projection matrix for the source domain.
        self.p1_ = v[0:n_col_ref, 0 : self.n_components].T
        # p2_: projection matrix for the target domain.
        self.p2_ = v[n_col_ref:, 0 : self.n_components].T

        self.x_source_provided_ = True
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Use the trained model to transform the target data

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to transform

        Returns
        -------
        X_transformed : np.ndarray of shape (n_samples, n_features)
            Data transformed
        """
        # Verify that the model was trained
        check_is_fitted(self, ["p1_", "p2_"])

        # Check the data
        X = validate_data(
            self,
            X,
            ensure_2d=True,
            reset=False,
            dtype=np.float64,
        )

        # If fitted as identity, return X unchanged
        if not self.x_source_provided_:
            return X

        # Type assertions for type checker - these are guaranteed non-None when
        # x_source_provided_ is True
        assert self.p1_ is not None
        assert self.p2_ is not None
        # Compute the pseudo-inverse of p2_ to project from target latent space
        # back to the feature space.
        p2_inv = np.linalg.pinv(self.p2_)
        # Apply the transformation:
        # X @ p2_inv @ self.p1_ : map X to the latent space, then reconstruct
        # in source domain
        # X - (X @ p2_inv @ self.p2_) : add back the residual not captured by
        # the latent space
        return (X @ p2_inv @ self.p1_) + X - (X @ p2_inv @ self.p2_)
