from typing import Literal, Optional

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, OneToOneFeatureMixin
from sklearn.utils.validation import check_is_fitted, validate_data


class AddNoise(TransformerMixin, OneToOneFeatureMixin, BaseEstimator):
    """Add noise to input data from various probability distributions.

    This transformer adds random noise from specified probability distributions
    to the input data. Supported distributions include Gaussian, Poisson, and
    exponential.

    Parameters
    ----------
    distribution : {'gaussian', 'poisson', 'exponential'}, default='gaussian'
        The probability distribution to sample noise from.
    scale : float, default=0.0
        Scale parameter for the noise distribution:
        - For gaussian: standard deviation
        - For poisson: multiplication factor for sampled values
        - For exponential: scale parameter (1/λ)
    random_state : int, optional
        Random seed for reproducibility.

    Attributes
    ----------
    n_features_in_ : int
        Number of features in the training data.
    """

    def __init__(
        self,
        distribution: Literal["gaussian", "poisson", "exponential"] = "gaussian",
        scale: float = 0.0,
        random_state: Optional[int] = None,
    ):
        self.distribution = distribution
        self.scale = scale
        self.random_state = random_state

    def fit(self, X: np.ndarray, y=None) -> "AddNoise":
        """
        Fit the transformer to the input data.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            The input data to fit the transformer to.

        y : None
            Ignored.

        Returns
        -------
        self : NormalNoise
            The fitted transformer.
        """
        # Check that X is a 2D array and has only finite values
        X = validate_data(
            self, X, y="no_validation", ensure_2d=True, reset=True, dtype=np.float64
        )

        # Set the number of features
        self.n_features_in_ = X.shape[1]

        # Set the fitted attribute to True
        self._is_fitted = True

        # Instantiate the random number generator
        self._rng = np.random.default_rng(self.random_state)

        return self

    def transform(self, X: np.ndarray, y=None) -> np.ndarray:
        """
        Transform the input data by adding random normal noise.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            The input data to transform.

        y : None
            Ignored.

        Returns
        -------
        X_ : np.ndarray of shape (n_samples, n_features)
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

        # Select the noise function based on the selected distribution
        noise_func = {
            "gaussian": self._add_gaussian_noise,
            "poisson": self._add_poisson_noise,
            "exponential": self._add_exponential_noise,
        }.get(self.distribution)

        if noise_func is None:
            raise ValueError(
                f"Invalid noise distribution: {self.distribution}. "
                "Expected one of: gaussian, poisson, exponential"
            )

        return noise_func(X_)

    def _add_gaussian_noise(self, X: np.ndarray) -> np.ndarray:
        """Add Gaussian noise to the input array."""
        return X + self._rng.normal(0, self.scale, size=X.shape)

    def _add_poisson_noise(self, X: np.ndarray) -> np.ndarray:
        """Add Poisson noise to the input array."""
        return X + self._rng.poisson(X, size=X.shape) * self.scale

    def _add_exponential_noise(self, X: np.ndarray) -> np.ndarray:
        """Add exponential noise to the input array."""
        return X + self._rng.exponential(self.scale, size=X.shape)