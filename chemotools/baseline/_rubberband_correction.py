"""
The :mod:`chemotools.baseline._rubberband_correction` module implements
a rubberband baseline correction transformer.
"""

# Author: Lasse Skjoldborg Krog
# License: MIT

from numbers import Integral

import numpy as np
from scipy.spatial import ConvexHull, QhullError
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin
from sklearn.utils._param_validation import Interval
from sklearn.utils.validation import check_is_fitted, validate_data

from chemotools._doc_mixin import DocLinkMixin
from chemotools._parallel import parallel_apply_by_rows


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

    _parameter_constraints: dict = {
        "n_jobs": [
            Interval(Integral, None, -1, closed="right"),
            Interval(Integral, 1, None, closed="left"),
        ],
    }

    def __init__(self, n_jobs: int = 1):
        self.n_jobs = n_jobs

    def __setstate__(self, state: dict) -> None:
        """Restore state while keeping backward compatibility with old pickles."""
        super().__setstate__(state)
        if "n_jobs" not in self.__dict__:
            self.n_jobs = 1

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

        X_transformed = parallel_apply_by_rows(
            X_, n_jobs=self.n_jobs, block_fn=self._transform_block
        )
        return X_transformed

    @staticmethod
    def _rubberband_baseline_single(x: np.ndarray) -> np.ndarray:
        """Return the rubberband (lower convex hull) baseline of one spectrum.

        The lower convex hull is extracted from ``scipy.spatial.ConvexHull``
        by selecting edges whose outward normal points downward (``b < 0`` in
        the ``ax + by + c = 0`` facet equation).  Linear interpolation fills
        in every feature index.

        Falls back to a straight line between the two endpoints when the input
        is degenerate (collinear points → ``QhullError``).
        """
        n = x.size
        pts = np.empty((n, 2), dtype=np.float64)
        pts[:, 0] = np.arange(n, dtype=np.float64)
        pts[:, 1] = x

        try:
            hull = ConvexHull(pts)
        except QhullError:
            # Flat or monotone spectra: lower hull is the straight
            # line form the first to the last point.
            return np.interp(
                np.arange(n, dtype=np.float64),
                [0.0, float(n - 1)],
                [x[0], x[-1]],
            )

        # Calculation of the lower hull
        # hull.equations has shape (nfacets, 3): each row [a, b, c] is the
        # outward-facing half-plane ax + by + c = 0.  Lower hull edges have
        # their outward normal pointing downward, so b < 0.
        lower_verts = np.unique(hull.simplices[hull.equations[:, 1] < 0])
        return np.interp(
            np.arange(n, dtype=np.float64),
            lower_verts.astype(np.float64),
            x[lower_verts],
        )

    @staticmethod
    def _rubberband_baseline_block(X_block: np.ndarray) -> np.ndarray:
        """Apply rubberband baseline correction to a block of spectra."""
        return np.array(
            [
                X_block[i]
                - RubberbandCorrection._rubberband_baseline_single(X_block[i])
                for i in range(X_block.shape[0])
            ]
        )

    def _transform_block(self, X_block: np.ndarray) -> np.ndarray:
        return self._rubberband_baseline_block(X_block)
