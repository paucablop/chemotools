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
        """Fit the transformer to the input data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : None
            Ignored. Present for API consistency.

        Returns
        -------
        self : AddNoise
            Fitted transformer.

        Raises
        ------
        ValueError
            If X is not a 2D array or contains non-finite values.
        """

        # Check that X is a 2D array and has only finite values
        X = validate_data(
            self, X, y="no_validation", ensure_2d=True, reset=True, dtype=np.float64
        )

        # Instantiate the random number generator
        self._rng = np.random.default_rng(self.random_state)

        return self

    def transform(self, X: np.ndarray, y=None) -> np.ndarray:
        """Transform the input data by adding random noise.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data to transform.
        y : None
            Ignored. Present for API consistency.

        Returns
        -------
        X_noisy : ndarray of shape (n_samples, n_features)
            Transformed data with added noise.

        Raises
        ------
        ValueError
            If X has different number of features than the training data,
            or if an invalid noise distribution is specified.
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
