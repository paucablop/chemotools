"""
The :mod:`chemotools.baseline._rubberband_correction` module implements
a rubberband baseline correction transformer.
"""

# Author: Lasse Skjoldborg Krog
# License: MIT

import numpy as np
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin
from sklearn.utils.validation import check_is_fitted, validate_data

from chemotools._doc_mixin import DocLinkMixin


class RubberbandCorrection(
    DocLinkMixin, TransformerMixin, OneToOneFeatureMixin, BaseEstimator
):
    """
    A transformer that removes a baseline using the rubberband method.

    The rubberband baseline is the lower convex hull of the spectrum — the
    set of straight-line segments a rubber band would form if stretched
    along the underside of the spectrum. The baseline is subtracted from
    each spectrum, leaving the peaks resting on a flat zero background.

    The lower convex hull is computed with Andrew's monotone chain
    algorithm [1]_ [2]_ in feature-index space, so the feature axis
    (e.g. wavenumbers) is assumed to be sorted; even spacing is not required.
    The method has no parameters.

    Attributes
    ----------
    n_features_in_ : int
        The number of features in the input data.

    References
    ----------
    .. [1] A. M. Andrew, "Another efficient algorithm for convex hulls in two
       dimensions", Information Processing Letters, 9(5), 216-219, 1979.
    .. [2] Monotone chain convex hull, reference implementation:
       https://en.wikibooks.org/wiki/Algorithm_Implementation/Geometry/Convex_hull/Monotone_chain

    Examples
    --------
    >>> from chemotools.baseline import RubberbandCorrection
    >>> from chemotools.datasets import load_fermentation_train
    >>> # Load sample data
    >>> X, _ = load_fermentation_train()
    >>> # Instantiate the transformer
    >>> transformer = RubberbandCorrection()
    RubberbandCorrection()
    >>> transformer.fit(X)
    >>> # Generate baseline-corrected data
    >>> X_corrected = transformer.transform(X)
    """

    _parameter_constraints: dict = {}

    def fit(self, X: np.ndarray, y=None) -> "RubberbandCorrection":
        """
        Fit the transformer to the input data.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            The input data to fit the transformer to.

        y : None
            Ignored to align with API.

        Returns
        -------
        self : RubberbandCorrection
            The fitted transformer.
        """
        # Validate the input parameters
        self._validate_params()

        # Check that X is a 2D array and has only finite values
        X = validate_data(
            self, X, y="no_validation", ensure_2d=True, reset=True, dtype=np.float64
        )

        return self

    def transform(self, X: np.ndarray, y=None) -> np.ndarray:
        """
        Transform the input data by subtracting the rubberband baseline.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            The input data to transform.

        y : None
            Ignored to align with API.

        Returns
        -------
        X_transformed : np.ndarray of shape (n_samples, n_features)
            The baseline-corrected data.
        """
        # Check that the estimator is fitted
        check_is_fitted(self, "n_features_in_")

        # Check that X is a 2D array and has only finite values
        X_ = validate_data(
            self,
            X,
            y="no_validation",
            ensure_2d=True,
            copy=True,
            reset=False,
            dtype=np.float64,
        )

        # Subtract the rubberband baseline from each spectrum
        for i, x in enumerate(X_):
            X_[i] = x - self._rubberband_baseline(x)

        return X_.reshape(-1, 1) if X_.ndim == 1 else X_

    @staticmethod
    def _rubberband_baseline(x: np.ndarray) -> np.ndarray:
        """Return the rubberband (lower convex hull) baseline of one spectrum.

        The lower convex hull is found with Andrew's monotone chain
        algorithm and linearly interpolated back onto every feature index.

        See the ``References`` section of the class docstring for the
        algorithm source.
        """
        n = x.size

        # Andrew's monotone chain — lower hull only. Reference implementation:
        # https://en.wikibooks.org/wiki/Algorithm_Implementation/Geometry/Convex_hull/Monotone_chain
        lower: list[tuple[int, float]] = []
        for i in range(n):
            point = (i, float(x[i]))
            while len(lower) >= 2:
                o, a = lower[-2], lower[-1]
                cross = (a[0] - o[0]) * (point[1] - o[1]) - (a[1] - o[1]) * (
                    point[0] - o[0]
                )
                if cross <= 0:
                    lower.pop()
                else:
                    break
            lower.append(point)

        hull_idx = np.array([p[0] for p in lower])
        hull_val = np.array([p[1] for p in lower])
        return np.interp(np.arange(n), hull_idx, hull_val)
