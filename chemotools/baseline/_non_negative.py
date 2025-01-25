import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, OneToOneFeatureMixin
from sklearn.utils.validation import check_is_fitted, validate_data


class NonNegative(TransformerMixin, OneToOneFeatureMixin, BaseEstimator):
    """
    A transformer that sets all negative values to zero or to abs.

    Parameters
    ----------
    mode : str, optional
        The mode to use for the non-negative values. Can be "zero" or "abs".

    Methods
    -------
    fit(X, y=None)
        Fit the transformer to the input data.

    transform(X, y=0, copy=True)
        Transform the input data by subtracting the constant baseline value.
    """

    def __init__(self, mode: str = "zero"):
        self.mode = mode

    def fit(self, X: np.ndarray, y=None) -> "NonNegative":
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
        self : ConstantBaselineCorrection
            The fitted transformer.
        """
        # Check that X is a 2D array and has only finite values
        X = validate_data(
            self, X, y="no_validation", ensure_2d=True, reset=True, dtype=np.float64
        )
        return self

    def transform(self, X: np.ndarray, y=None) -> np.ndarray:
        """
        Transform the input data by subtracting the constant baseline value.

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

        # Check that the number of features is the same as the fitted data
        if X_.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features but got {X_.shape[1]}"
            )

        # Calculate non-negative values
        for i, x in enumerate(X_):
            if self.mode == "zero":
                X_[i] = np.clip(x, a_min=0, a_max=np.inf)

            if self.mode == "abs":
                X_[i] = np.abs(x)

        return X_.reshape(-1, 1) if X_.ndim == 1 else X_
