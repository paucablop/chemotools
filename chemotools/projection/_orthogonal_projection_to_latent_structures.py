"""
The :mod:`chemotools.projection._orthogonal_projection_to_latent_structures` module
implements the Orthogonal Projection to Latent Structures (OPLS) technique for
preprocessing spectral data by removing variations orthogonal to the target variable.
"""

# Author: Pau Cabaneros
# License: MIT

import numpy as np
from scipy.linalg import svd
from sklearn.base import BaseEstimator, TransformerMixin


class OrthogonalPLS(TransformerMixin, BaseEstimator):
    def __init__(self, n_components: int = 1, copy=False):
        self.n_components = n_components
        self.copy = copy

    def fit(self, X: np.ndarray, y: np.ndarray) -> "OrthogonalPLS":
        # Get the dimensions
        n = X.shape[0]
        p = X.shape[1]

        # TODO: Mean center and optionally scale the data
        Xk = X.copy()
        yk = y.copy()
        yk = yk.reshape(-1, 1)

        # Allocate scores and weights
        self.x_weights_ = np.zeros((p, self.n_components))  # w in [1]
        self.x_weights_orth_ = np.zeros((p, self.n_components))  # w_ortho in [1]
        self.x_loadings_ = np.zeros((p, self.n_components))  # p in [1]
        self.x_loadings_orth_ = np.zeros((p, self.n_components))  # p_ortho in [1]
        self.x_scores_ = np.zeros((n, self.n_components))  # t in [1]
        self.x_scores_orth_ = np.zeros((n, self.n_components))  # t_ortho in [1]

        # For each component
        for k in range(self.n_components):
            # Step 1: Weights are calculated through SVD to support multi y
            # Step 1.1. Calculate covariance matrix (C)
            C = np.dot(Xk.T, yk)

            # Step 1.2: Calculate the SVD of C
            U, _, _ = svd(C, full_matrices=True)

            # Step 1.3: We just use the first weight
            x_weights = U[:, 0]

            # Step 2. Normalize the weights (Step 2 in [1])
            x_weights /= np.linalg.norm(x_weights)

            # Step 3: Calculate the x_scores (Step 3 in [1])
            x_scores = np.dot(Xk, x_weights) / np.dot(x_weights.T, x_weights)

            # Step 4: Calculate the x_loadings (Step 6 in [1])
            x_loadings = np.dot(x_scores.T, Xk) / np.dot(x_scores.T, x_scores)
            x_loadings = x_loadings.T

            # Step 5: Calculate orthogonal x weights (Step 7 in [1])
            x_weights_orth = (
                x_loadings
                - (np.dot(x_weights.T, x_loadings) / np.dot(x_weights.T, x_weights))
                * x_weights
            )

            # Step 6: Normalize the orthogonal weights (Step 8 in [1])
            x_weights_orth /= np.linalg.norm(x_weights_orth)

            # Step 7: Calculate orthogonal x scores (Step 9 in [1])
            x_scores_orth = np.dot(Xk, x_weights_orth) / np.dot(
                x_weights_orth.T, x_weights_orth
            )

            # Step 7: Calculate orthogonal x loadings (Step 10 in [1])
            x_loadings_orth = np.dot(x_scores_orth.T, Xk) / np.dot(
                x_scores_orth.T, x_scores_orth
            )

            # Step 8: Deflation of X matrix (Step 11 in [1])
            Xk -= np.outer(x_scores_orth, x_loadings_orth)

            # Step 9: Collect the variables
            self.x_weights_[:, k] = x_weights
            self.x_weights_orth_[:, k] = x_weights_orth
            self.x_loadings_[:, k] = x_loadings
            self.x_loadings_orth_[:, k] = x_loadings_orth
            self.x_scores_[:, k] = x_scores
            self.x_scores_orth_[:, k] = x_scores_orth

            return self

    def transform(self, X: np.ndarray, y: np.ndarray, copy=True) -> np.ndarray:
        return X - self.dot(self.scores_orth_, self.x_loadings_orth_)
