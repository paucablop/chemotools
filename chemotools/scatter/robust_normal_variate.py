import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, OneToOneFeatureMixin
from sklearn.utils.validation import check_is_fitted

from chemotools.utils.check_inputs import check_input


class RobustNormalVariate(OneToOneFeatureMixin, BaseEstimator, TransformerMixin):
    """
    A transformer that calculates the robust normal variate of the input data.

    Parameters
    ----------
    percentile : float, optional
        The percentile to use for the robust normal variate. The value should be
        between 0 and 100. The default is 25.

    Attributes
    ----------
    n_features_in_ : int
        The number of features in the input data.

    _is_fitted : bool
        Whether the transformer has been fitted to data.

    Methods
    -------
    fit(X, y=None)
        Fit the transformer to the input data.

    transform(X, y=0, copy=True)
        Transform the input data by calculating the standard normal variate.

    References
    ----------
    Q. Guo, W. Wu, D.L. Massart. The robust normal variate transform for pattern
    recognition with near-infrared data. doi:10.1016/S0003-2670(98)00737-5
    """

    def __init__(self, percentile: float = 25):
        self.percentile = percentile

    def fit(self, X: np.ndarray, y=None) -> "RobustNormalVariate":
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
        self : RobustNormalVariate
            The fitted transformer.
        """
        # Check that X is a 2D array and has only finite values
        X = check_input(X)

        # Set the number of features
        self.n_features_in_ = X.shape[1]

        # Set the fitted attribute to True
        self._is_fitted = True

        return self

    def transform(self, X: np.ndarray, y=None) -> np.ndarray:
        """
        Transform the input data by calculating the standard normal variate.

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
        X = check_input(X)
        X_ = X.copy()

        # Check that the number of features is the same as the fitted data
        if X_.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features but got {X_.shape[1]}"
            )

        # Calculate the standard normal variate
        for i, x in enumerate(X_):
            X_[i] = self._calculate_robust_normal_variate(x)

        return X_.reshape(-1, 1) if X_.ndim == 1 else X_

    def _calculate_robust_normal_variate(self, x) -> np.ndarray:
        percentile = np.percentile(x, self.percentile)
        return (x - percentile) / np.std(x[x <= percentile])
