"""
The :mod: `chemotools.adaptation._direct_standardization`
module implements the Direct Standardization (DS) transformer
"""

# Authors: Ruggero Guerrini
# License: MIT

import warnings

import numpy as np
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin
from sklearn.utils.validation import (
    check_is_fitted,
    validate_data,
)

from chemotools._doc_mixin import DocLinkMixin


class DirectStandardization(
    DocLinkMixin, OneToOneFeatureMixin, TransformerMixin, BaseEstimator
):
    """
    Direct Standardization (DS) is a transformer used for domain adaptation (calibration
    ) applications. The transformer uses least squares to find a linear map from
    the target instrument space to the source instrument space, following the
    implementation by [1]_.

    Attributes
    ----------
    T_ : np.ndarray of shape (n_features, n_features)
        Linear transformation matrix mapping target instrument space to source
        instrument space.

    x_source_provided_ : bool
        Boolean value to flag if X_source was provided during fitting

    Raises
    ------
    ValueError
        If X and X_source do not have the same shape.

    See Also
    --------
    PiecewiseDirectStandardization : Localized version using windowed PLS regression.

    References
    ----------
    .. [1] Wang, Yongdong., Veltkamp, D. J., & Kowalski, B. R. (1991),
        Multivariate instrument standardization,
        Analytical Chemistry, 63(23), Pages 2750–2756,
        https://doi.org/10.1021/ac00023a016.

    Examples
    --------
    **Basic usage**
    >>> import numpy as np
    >>> from chemotools.adaptation import DirectStandardization
    >>>
    >>> rng = np.random.default_rng(17)
    >>> X_source = rng.normal(size=(100, 20))
    >>> X_target = X_source * 2 - rng.normal(size=(100, 20)) * 0.02
    >>>
    >>> ds = DirectStandardization().fit(X_target, X_source=X_source)
    >>> X_transf = ds.transform(X_target)

    """

    # Fitted attributes (set during fit, typed for type checkers)
    n_features_in_: int
    T_: np.ndarray
    x_source_provided_: bool

    _parameter_constraints: dict = {}

    def fit(
        self, X: np.ndarray, y=None, *, X_source: np.ndarray | None = None
    ) -> "DirectStandardization":
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
            self.T_ = np.eye(X.shape[1])
            self.x_source_provided_ = False

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

        self.T_, _, _, _ = np.linalg.lstsq(X, X_source, rcond=None)
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

        check_is_fitted(self, ["T_"])

        # Validate the input data
        X = validate_data(
            self,
            X,
            ensure_2d=True,
            reset=False,
            dtype=np.float64,
        )

        # Apply the transformation
        return X @ self.T_
