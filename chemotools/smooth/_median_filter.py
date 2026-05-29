"""
The :mod:`chemotools.smooth._median_filter` module implements
the Median Filter (MD) transformation.
"""

# Authors: Pau Cabaneros
# License: MIT

from numbers import Integral
from typing import Literal

import numpy as np
from joblib import Parallel, delayed, effective_n_jobs
from scipy.ndimage import median_filter
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.validation import check_is_fitted, validate_data

from chemotools._deprecation import (
    DEPRECATED_PARAMETER,
    deprecated_parameter_constraint,
    resolve_renamed_parameter,
)
from chemotools._doc_mixin import DocLinkMixin


class MedianFilter(DocLinkMixin, TransformerMixin, OneToOneFeatureMixin, BaseEstimator):
    """
    A smoothing transformer that calculates the median filter of the input data.

    Parameters
    ----------
    window_length : int, optional, default=3
        The size of the window to use for the median filter. Must be odd. Default is 3.

    mode : str, optional, default="nearest"
        The mode to use for the median filter. Can be "nearest", "constant", "reflect",
        "wrap", "mirror", "grid-constant", "grid-mirror" or "grid-wrap".
        Default is "nearest".

    n_jobs : int, optional, default=1
        Number of parallel jobs used to process samples independently.
        Uses serial execution when set to 1.

    window_size : int, optional
        Deprecated alias for ``window_length``.

    Attributes
    ----------
    n_features_in_ : int
        The number of features in the training data.

    Examples
    --------
    >>> from chemotools.datasets import load_fermentation_train
    >>> from chemotools.smooth import MedianFilter
    >>> # Load sample data
    >>> X, _ = load_fermentation_train()
    >>> # Initialize MedianFilter
    >>> md = MedianFilter()
    MedianFilter()
    >>> # Fit and transform the data
    >>> X_smoothed = md.fit_transform(X)
    """

    _parameter_constraints: dict = {
        "window_length": [Interval(Integral, 3, None, closed="left")],
        "mode": [
            StrOptions(
                {
                    "nearest",
                    "constant",
                    "reflect",
                    "wrap",
                    "mirror",
                    "grid-constant",
                    "grid-mirror",
                    "grid-wrap",
                }
            )
        ],
        "window_size": [
            Interval(Integral, 3, None, closed="left"),
            deprecated_parameter_constraint(),
        ],
        "n_jobs": [
            Interval(Integral, None, -1, closed="right"),
            Interval(Integral, 1, None, closed="left"),
        ],
    }

    def __init__(
        self,
        window_length: int = 3,
        mode: Literal[
            "reflect",
            "constant",
            "nearest",
            "mirror",
            "wrap",
            "grid-constant",
            "grid-mirror",
            "grid-wrap",
        ] = "nearest",
        n_jobs: int = 1,
        window_size=DEPRECATED_PARAMETER,
    ) -> None:
        self.window_length = window_length
        self.window_size = window_size
        self.mode = mode
        self.n_jobs = n_jobs

    def __setstate__(self, state: dict) -> None:
        """Restore state while keeping backward compatibility with old pickles."""
        super().__setstate__(state)
        if "n_jobs" not in self.__dict__:
            self.n_jobs = 1

    def fit(self, X: np.ndarray, y=None) -> "MedianFilter":
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
        self : MedianFilter
            The fitted transformer.
        """
        # Validate the input parameters
        self._validate_params()

        # Check that X is a 2D array and has only finite values
        X = validate_data(
            self, X, y="no_validation", ensure_2d=True, reset=True, dtype=np.float64
        )

        self.window_length_ = resolve_renamed_parameter(
            new_name="window_length",
            new_value=self.window_length,
            new_default=3,
            old_name="window_size",
            old_value=self.window_size,
        )

        if self.window_length_ % 2 == 0:
            raise ValueError("window_length must be odd")

        if self.n_jobs == 0:
            raise ValueError("n_jobs must be different from 0")

        return self

    def transform(self, X: np.ndarray, y=None) -> np.ndarray:
        """
        Transform the input data by calculating the median filter.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            The input data to transform.

        y : None
            Ignored to align with API.

        Returns
        -------
        X_transformed : np.ndarray of shape (n_samples, n_features)
            The transformed data.
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

        # Median filter the data
        if self.n_jobs == 1:
            return median_filter(
                X_, size=self.window_length_, mode=self.mode, axes=(1,)
            )

        # Parallelize across samples (rows)
        n_effective_jobs = effective_n_jobs(self.n_jobs)
        if n_effective_jobs <= 1 or X_.shape[0] < 2:
            return median_filter(
                X_, size=self.window_length_, mode=self.mode, axes=(1,)
            )

        n_samples = X_.shape[0]
        chunk_size = max(1, (n_samples + n_effective_jobs - 1) // n_effective_jobs)
        chunks = [
            X_[start : start + chunk_size] for start in range(0, n_samples, chunk_size)
        ]

        filtered_chunks = Parallel(n_jobs=self.n_jobs, prefer="threads")(
            delayed(median_filter)(
                chunk,
                size=self.window_length_,
                mode=self.mode,
                axes=(1,),
            )
            for chunk in chunks
        )

        X_out = np.empty_like(X_)
        start = 0
        for filtered_chunk in filtered_chunks:
            stop = start + filtered_chunk.shape[0]
            X_out[start:stop] = filtered_chunk
            start = stop

        return X_out
