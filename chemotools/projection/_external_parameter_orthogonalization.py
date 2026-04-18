"""
The :mod:`chemotools.projection._external_parameter_orthogonalization` module
implements the External Parameter Orthogonalization (EPO) technique for preprocessing
spectral data by removing variations orthogonal to the external parameters.
"""

# Author: Pau Cabaneros
# License: MIT

from numbers import Integral
from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils._param_validation import Interval
from sklearn.utils.validation import check_array, check_is_fitted, validate_data


class ExternalParameterOrthogonalization(TransformerMixin, BaseEstimator):
    """
    Remove variation linked to known external nuisance parameters using
    External Parameter Orthogonalization (EPO) [1]_.

    EPO is designed for situations where spectral measurements are affected by
    controlled external factors (e.g., temperature, humidity, instrument
    differences) that are not related to the target property. The method
    estimates a nuisance subspace from an auxiliary dataset or from structured
    replicates in which the external parameter varies while the underlying
    sample composition is held constant.

    A matrix capturing this external variation is constructed (e.g., from
    differences or deviations within replicate groups), and its dominant
    components are obtained via SVD/PCA. These components define a subspace
    associated with the external parameter. A projection operator is then
    applied to X to remove variation in this subspace.

    When ``sample_ids`` are provided, the external-effect matrix is formed from
    within-sample deviations (i.e., each spectrum minus its sample mean),
    isolating variation due to the external parameter from chemical variation.

    The transformer preserves the original number of features and is intended
    as a signal correction step rather than dimensionality reduction.

    Parameters
    ----------
    n_components : int, default=2
        Number of orthogonal components to remove. Must be a positive integer.

    copy : bool, default=True
        Placeholder argument kept for API compatibility with scikit-learn style
        estimators. Input validation currently relies on the default behavior of
        the underlying validation utilities.

    Attributes
    ----------
    mean_X_ : ndarray of shape (n_features,)
        Mean spectrum computed from the calibration data passed to `fit()`.

    P_epo_ : ndarray of shape (n_features, n_features)
        Orthogonal projection matrix used to suppress nuisance variation.
        Applying `X_centered @ P_epo_` removes the subspace spanned by the first
        `n_components` singular vectors of the external variation matrix.

    n_features_in_ : int
        Number of features seen during `fit()`.


    References
    ----------
    .. [1] Jean-Michel Roger, Fabien Chauchard, Veronique Bellon-Maurel (2003),
        EPO–PLS external parameter orthogonalisation of PLS application to
        temperature-independent measurement of sugar content of intact fruits,
        Chemometrics and Intelligent Laboratory Systems,
        Volume 66, Issue 2, Pages 191-204,
        https://doi.org/10.1016/S0169-7439(03)00051-0

    Examples
    --------
    >>> import numpy as np
    >>> from chemotools.projection import (
    ...     ExternalParameterOrthogonalization,
    ... )
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(6, 4))
    >>> X_external = X + 0.2 * rng.normal(size=(6, 4))
    >>> epo = ExternalParameterOrthogonalization(n_components=1)
    >>> X_epo = epo.fit_transform(X, X_external=X_external)
    >>> X_epo.shape
    (6, 4)

    Repeated measurements of the same physical sample can be grouped through
    `sample_ids` so that the nuisance subspace is estimated from within-sample
    differences only.

    >>> sample_ids = np.array([0, 0, 1, 1, 2, 2])
    >>> epo = ExternalParameterOrthogonalization(n_components=1)
    >>> epo.fit(X, X_external=X_external, sample_ids=sample_ids)
    ExternalParameterOrthogonalization(n_components=1)

    Notes
    -----
    EPO is commonly used when spectral measurements are affected by known
    nuisance sources such as temperature, instrument transfer, humidity, or
    acquisition conditions. The nuisance structure is estimated from
    `X_external`, then projected out from `X`.

    If `sample_ids` are provided, the difference matrix is built from deviations
    around the mean spectrum of each repeated sample. This isolates variation due
    to the external condition while suppressing the underlying chemical signal.

    The transformer preserves the original number of features. It performs signal
    correction, not dimensionality reduction.


    See Also
    --------
    chemotools.projection.OrthogonalSignalCorrection : Remove variation
        orthogonal to a supervised target.
    sklearn.pipeline.make_pipeline : Compose EPO with downstream estimators.

    """

    _parameter_constraints: dict = {
        "n_components": [Interval(Integral, 1, None, closed="left")],
        "copy": ["boolean"],
    }

    def __init__(
        self,
        n_components: int = 2,
        copy: bool = True,
    ):
        """Initialize the External Parameter Orthogonalization (EPO) transformer.

        Parameters
        ----------
        n_components : int, default=2
            Number of orthogonal components to remove. Must be a positive integer.
        copy : bool, default=True
            Whether to copy X and Y in fit before applying centering.
        """
        self.n_components = n_components
        self.copy = copy

    def fit(
        self,
        X: np.ndarray,
        y=None,
        X_external: Optional[np.ndarray] = None,
        sample_ids: Optional[np.ndarray] = None,
    ):
        """Fit the EPO projection from calibration and nuisance spectra.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Calibration spectra that will later be corrected by the learned EPO
            projection.

        y : None, default=None
            Ignored. Present for scikit-learn API compatibility.

        X_external : array-like of shape (n_samples, n_features)
            Spectra describing the nuisance variation to remove. These may be the
            same samples measured under perturbed external conditions, transfer
            standards, or any dataset representative of the unwanted subspace.

        sample_ids : array-like of shape (n_samples,), default=None
            Optional identifiers linking repeated measurements of the same sample.
            When provided, the nuisance difference matrix is computed within each
            sample group, which helps isolate external variation from chemical
            differences between samples.

        Returns
        -------
        self : ExternalParameterOrthogonalization
            Fitted estimator storing the projection matrix in `P_epo_`.

        Notes
        -----
        The nuisance variation matrix $D$ is constructed as either centered
        `X_external` or within-group deviations when `sample_ids` are available.
        A singular value decomposition of $D$ yields the dominant nuisance
        directions, and the projection matrix is then defined as
        $P = I - VV^T$.
        """
        self._validate_params()
        # 1. Check that X is a 2D array and has only finite values
        X = validate_data(self, X, dtype=np.float64)
        self.mean_X_ = np.mean(X, axis=0)

        if X_external is None:
            # No nuisance subspace: identity projection (no correction)
            self.P_epo_ = np.eye(X.shape[1])
            self.n_features_in_ = X.shape[1]
            return self

        X_ext = check_array(X_external, dtype=np.float64)

        # 2. Construct the Variation Matrix D
        if sample_ids is not None:
            # Case A: labels are available, isolate within-sample variation
            D = self._build_difference_matrix(X_ext, sample_ids)
        else:
            # Case B: No labels. treat the whole X_ext cloud as the noise
            # distribution.
            # Centering identifies the 'directions of maximum interference'.
            D = X_ext - np.mean(X_ext, axis=0)

        # 3. SVD on the Variation Matrix
        # We perform SVD on D to find the loadings (Vt) of the nuisance.
        _, _, Vt = np.linalg.svd(D, full_matrices=False)

        # 4. Define the projection operator
        # V are the first k components that span the 'bad' subspace
        V = Vt[: self.n_components, :].T

        # P = I - V * V^T
        self.P_epo_ = np.eye(X.shape[1]) - (V @ V.T)

        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: np.ndarray):
        """Project spectra onto the subspace orthogonal to nuisance variation.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Spectra to correct.

        Returns
        -------
        ndarray of shape (n_samples, n_features)
            Corrected spectra after centering with `mean_X_` and projection with
            `P_epo_`.
        """
        check_is_fitted(self, "P_epo_")
        X = validate_data(self, X, reset=False)

        # 1. Center the new data using the TRAINING mean
        X_centered = X - self.mean_X_

        # 2. Project into the orthogonal space
        # X_corr = (X - mu) @ (I - VV^T)
        return X_centered @ self.P_epo_ + self.mean_X_

    def _build_difference_matrix(
        self, X_ext: np.ndarray, sample_ids: np.ndarray
    ) -> np.ndarray:
        """Build the nuisance variation matrix."""
        X_ext = check_array(X_ext, dtype=np.float64)
        sample_ids = np.asarray(sample_ids)

        # 1. Map each unique sample ID to a contiguous integer index (0, 1, 2...)
        # inverse_indices has the same shape as sample_ids
        unique_ids, inverse_indices = np.unique(sample_ids, return_inverse=True)

        # 2. Count the number of replicates per sample ID
        counts = np.bincount(inverse_indices)

        # 3. Sum the spectra for each unique sample ID
        group_sums = np.zeros((len(unique_ids), X_ext.shape[1]), dtype=X_ext.dtype)
        np.add.at(group_sums, inverse_indices, X_ext)

        # 4. Calculate the mean spectrum for each group
        # Adding a small epsilon prevents division by zero if a group is empty
        # though np.unique guarantees no empty groups here.
        group_means = group_sums / counts[:, np.newaxis]

        # 5. Broadcast the group means back to the original X_ext shape and subtract
        # This is the mathematical equivalent of: D_i = X_i - mean(X_group)
        D = X_ext - group_means[inverse_indices]

        # 6. Filter out singleton groups (where count < 2)
        # We broadcast the counts array back to the original shape to create a mask
        valid_mask = counts[inverse_indices] >= 2

        return D[valid_mask]
