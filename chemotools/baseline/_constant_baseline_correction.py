"""
The :mod:`chemotools.baseline._constant_baseline_correction` module implements
a constant baseline correction transformer.
"""

# Author: Pau Cabaneros
# License: MIT

from numbers import Integral
from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin
from sklearn.utils._param_validation import Interval
from sklearn.utils.validation import check_is_fitted, validate_data

from chemotools._deprecation import (
    DEPRECATED_PARAMETER,
    deprecated_parameter_constraint,
    resolve_renamed_parameter,
)
from chemotools._doc_mixin import DocLinkMixin


class ConstantBaselineCorrection(
    DocLinkMixin, TransformerMixin, OneToOneFeatureMixin, BaseEstimator
):
    """
    A transformer that corrects a baseline by subtracting a constant value.
    The constant value is taken by the mean of the features between the start
    and end indices. This is a common preprocessing technique for UV-Vis spectra.

    Parameters
    ----------
    start : int, optional, default=0
        The index of the first feature to use for the baseline correction.

    end : int, optional, default=1
        The index of the last feature to use for the baseline correction.

    x_axis : np.ndarray, optional, default=None
        The x-axis values corresponding to each feature in the input data.

    wavenumbers : np.ndarray, optional
        Deprecated alias for ``x_axis``.

    Attributes
    ----------
    start_index_ : int
        The index of the start of the range. It is 0 if the
        wavenumbers are not provided.

    end_index_ : int
        The index of the end of the range. It is 1 if the wavenumbers are not provided.

    Examples
    --------
    >>> from chemotools.baseline import ConstantBaselineCorrection
    >>> from chemotools.datasets import load_fermentation_train
    >>> # Load sample data
    >>> X, _ = load_fermentation_train()
    >>> # Instantiate the transformer
    >>> transformer = ConstantBaselineCorrection(start=0, end=1)
    >>> transformer.fit(X)
    >>> # Generate baseline-corrected data
    >>> X_corrected = transformer.transform(X)
    """

    _parameter_constraints: dict = {
        "start": [Interval(Integral, 0, None, closed="left")],
        "end": [Interval(Integral, 0, None, closed="left")],
        "x_axis": ["array-like", None],
        "wavenumbers": ["array-like", None, deprecated_parameter_constraint()],
    }

    def __init__(
        self,
        start: int = 0,
        end: int = 1,
        x_axis: Optional[np.ndarray] = None,
        wavenumbers=DEPRECATED_PARAMETER,
    ) -> None:
        self.start = start
        self.end = end
        self.x_axis = x_axis
        self.wavenumbers = wavenumbers

    def fit(self, X: np.ndarray, y=None) -> "ConstantBaselineCorrection":
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
        self : ConstantBaselineCorrection
            The fitted transformer.
        """
        # Validate the input parameters
        self._validate_params()

        # Check that X is a 2D array and has only finite values
        X = validate_data(
            self, X, y="no_validation", ensure_2d=True, reset=True, dtype=np.float64
        )

        axis_values = self._resolve_x_axis()

        # Set the start and end indices
        if axis_values is None:
            self.start_index_ = self.start
            self.end_index_ = self.end
        else:
            self.start_index_ = self._find_index(self.start, axis_values)
            self.end_index_ = self._find_index(self.end, axis_values)

        return self

    def transform(self, X: np.ndarray, y=None) -> np.ndarray:
        """
        Transform the input data by subtracting the constant baseline value.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            The input data to transform.

        y : None
            Ignored to align with API.

        Returns
        -------
        X_transformed : np.ndarray of shape (n_samples, n_features)
            The transformed input data.
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

        # Base line correct the spectra
        for i, x in enumerate(X_):
            mean_baseline = np.mean(x[self.start_index_ : self.end_index_ + 1])
            X_[i, :] = x - mean_baseline
        return X_.reshape(-1, 1) if X_.ndim == 1 else X_

    def _find_index(self, target: float, axis_values) -> int:
        wavenumbers = np.array(axis_values)
        return np.argmin(np.abs(wavenumbers - target)).astype(int)

    def _resolve_x_axis(self):
        return resolve_renamed_parameter(
            new_name="x_axis",
            new_value=self.x_axis,
            new_default=None,
            old_name="wavenumbers",
            old_value=self.wavenumbers,
        )
