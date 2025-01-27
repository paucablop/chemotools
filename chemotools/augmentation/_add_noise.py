from typing import Literal, Optional

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, OneToOneFeatureMixin
from sklearn.utils.validation import check_is_fitted, validate_data


class AddNoise(TransformerMixin, OneToOneFeatureMixin, BaseEstimator):
    """
    Add normal noise to the input data.
    """

    def __init__(
        self,
        noise_distribution: Literal["gaussian", "poisson", "exponential"] = "gaussian",
        scale: float = 0.0,
        random_state: Optional[int] = None,
    ):
        self.noise_distribution = noise_distribution
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
        check_is_fitted(self, "_is_fitted")

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

        # Check that the number of features is the same as the fitted data
        if X_.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features but got {X_.shape[1]}"
            )

        # Calculate the standard normal variate
        for i, x in enumerate(X_):
            match self.noise_distribution:
                case "gaussian":
                    X_[i] = self._add_gaussian_noise(x)
                case "poisson":
                    X_[i] = self._add_poisson_noise(x)
                case "exponential":
                    X_[i] = self._add_exponential_noise(x)
                case _:
                    raise ValueError("Invalid noise distribution")

        return X_.reshape(-1, 1) if X_.ndim == 1 else X_

    def _add_gaussian_noise(self, x) -> np.ndarray:
        return x + self._rng.normal(0, self.scale, size=x.shape)

    def _add_poisson_noise(self, x) -> np.ndarray:
        return self._rng.poisson(x, size=x.shape) * self.scale

    def _add_exponential_noise(self, x) -> np.ndarray:
        return x + self._rng.exponential(self.scale, size=x.shape)
