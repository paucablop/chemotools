"""
The :mod: `chemotools.adaptation._subspace_alignment`
module implements the Subspace aslignment (SA) transformer
"""

# Authors: Ruggero Guerrini
# License: MIT

import warnings
from numbers import Integral

import numpy as np
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.utils._param_validation import Interval
from sklearn.utils.validation import (
    check_is_fitted,
    validate_data,
)

from chemotools._doc_mixin import DocLinkMixin


class Subspaceaslignment(
    DocLinkMixin, OneToOneFeatureMixin, TransformerMixin, BaseEstimator
):
    """
    Subspace aslignment (SA) is a transformer used for domain adaptation (calibration
    ) applications. [..], following the
    implementation by [1]_.

    Attributes
    ----------
    T_ : np.ndarray of shape (n_features, n_features)
        Linear transformation matrix mapping target instrument space to source
        instrument space.

    x_source_provided_ : bool
        Boolean value to flag if X_source was provided during fitting

    n_features_in_: int
    components_X: np.ndarray | None
    components_X_source: np.ndarray | None
    x_source_provided_: bool
    X_mean_: np.ndarray | None
    X_std_: np.ndarray | None
    X_source_mean_: np.ndarray | None
    X_source_std_: np.ndarray | None
    Raises
    ------
    ValueError
        If X and X_source do not have the same shape.

    See Also
    --------
    PiecewiseDirectStandardization : Localized version using windowed PLS regression.

    References
    ----------
    .. [1] B. Fernando, A. Habrard, M. Sebban, and T. Tuytelaars,
        “Unsupervised Visual Domain Adaptation Using Subspace Alignment,”
        in Proceedings of the IEEE International Conference on Computer
        Vision (ICCV), 2013, pp. 2960–2967

    Examples
    --------
    **Basic usage**
    >>> import numpy as np
    >>> from chemotools.adaptation import Subspaceaslignment
    >>>
    >>> rng = np.random.default_rng(17)
    >>> X_source = rng.normal(size=(100, 20))
    >>> X_target = X_source * 2 - rng.normal(size=(100, 20)) * 0.02
    >>>
    >>> ds = Subspaceaslignment().fit(X_target, X_source=X_source)
    >>> X_transf = ds.transform(X_target)

    """

    _parameter_constraints: dict = {
        "n_components": [Interval(Integral, 1, None, closed="left")],
        "scale": ["boolean"],
    }

    # Fitted attributes
    n_features_in_: int
    components_X: np.ndarray | None
    components_X_source: np.ndarray | None
    x_source_provided_: bool
    X_mean_: np.ndarray | None
    X_std_: np.ndarray | None
    X_source_mean_: np.ndarray | None
    X_source_std_: np.ndarray | None

    def __init__(
        self,
        n_components: int = 4,
        scale: bool = False,
    ):
        self.n_components = n_components
        self.scale = scale

    def fit(
        self, X: np.ndarray, y=None, *, X_source: np.ndarray | None = None
    ) -> "Subspaceaslignment":
        """
        Fit the Direct Standardization model.

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
        self : DirectStandardization
        """
        # Validate the input parameters
        self._validate_params()
        # Check that X is a 2D array and has only finite values
        X = validate_data(self, X, ensure_2d=True, reset=True, dtype=np.float64)

        # If X_source is None, default to identity transformation
        if X_source is None:
            warnings.warn(
                "X_source is None, the transformer will act as an identity "
                "transformation."
            )
            self.components_X_ = None
            self.components_X_source_ = None
            self.x_source_provided_ = False
            self.X_mean_ = None
            self.X_std_ = None
            self.X_source_mean_ = None
            self.X_source_std_ = None
            return self

        # Check that X_source is a 2D array and has only finite values
        X_source = validate_data(
            self, X_source, ensure_2d=True, reset=False, dtype=np.float64
        )

        # Check consistency between X and X_source
        # if X_source.shape != X.shape:
        #     raise ValueError(
        #         f"X and X_source must have the same shape, "
        #         f"got X={X.shape} and X_source={X_source.shape}."
        #     )
        self.X_mean_ = X.mean(axis=0)
        self.X_std_ = X.std(axis=0)
        self.X_source_mean_ = X_source.mean(axis=0)
        self.X_source_std_ = X_source.std(axis=0)
        if self.scale:
            X_scaled = (X - self.X_mean_) / self.X_std_
            X_source_scaled = (X_source - self.X_source_mean_) / self.X_source_std_
        else:
            X_scaled = X.copy()
            X_source_scaled = X_source.copy()

        self.components_X_ = (
            PCA(n_components=self.n_components).fit(X_scaled).components_
        )
        self.components_X_source_ = (
            PCA(n_components=self.n_components).fit(X_source_scaled).components_
        )
        self.A_ = (
            self.components_X_.T @ self.components_X_ @ self.components_X_source_.T
        )

        self.x_source_provided_ = True

        return self

    def transform(self, X) -> np.ndarray:
        """
        Transform the data from the target space to the source space using the map
        ``self.T_``.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            The input data to transform

        Returns
        -------
        X_transf : np.ndarray of shape (n_samples, n_features)
            The data transformed
        """
        # Check that the estimator is fitted

        check_is_fitted(self)

        # Validate the input data
        X = validate_data(
            self,
            X,
            ensure_2d=True,
            reset=False,
            dtype=np.float64,
        )
        if not self.x_source_provided_:
            return X
        if self.scale:
            X_centered = (X - self.X_mean_) / self.X_std_
        else:
            X_centered = X - self.X_mean_

        return X_centered @ self.A_ @ self.components_X_source_ + self.X_source_mean_
