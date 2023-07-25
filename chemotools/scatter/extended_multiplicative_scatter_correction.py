import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, OneToOneFeatureMixin
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from chemotools.utils.check_inputs import check_input


class ExtendedMultiplicativeScatterCorrection(
    OneToOneFeatureMixin, BaseEstimator, TransformerMixin
):
    """Extended multiplicative scatter correction (EMSC) is a preprocessing technique for
    removing non linear scatter effects from spectra. It is based on fitting a polynomial
    regression model to the spectrum using a reference spectrum. The reference spectrum
    can be the mean or median spectrum of a set of spectra or a selected reerence.

    Note that this implementation does not include further extensions of the model using
    orthogonal subspace models.

    Parameters
    ----------
    reference : np.ndarray, optional
        The reference spectrum to use for the correction. If None, the mean
        spectrum will be used. The default is None.
    use_mean : bool, optional
        Whether to use the mean spectrum as the reference. The default is True.
    use_median : bool, optional
        Whether to use the median spectrum as the reference. The default is False.
    order : int, optional
        The order of the polynomial to fit to the spectrum. The default is 2.
    weights : np.ndarray, optional
        The weights to use for the weighted EMSC. If None, the standard EMSC
        will be used. The default is None.


    Attributes
    ----------
    reference_ : np.ndarray
        The reference spectrum used for the correction.
    n_features_in_ : int
        The number of features in the training data.

    References
    ----------
    Nils Kristian Afseth, Achim Kohler. Extended multiplicative signal correction
    in vibrational spectroscopy, a tutorial, doi:10.1016/j.chemolab.2012.03.004

    Valeria Tafintseva et al. Correcting replicate variation in spectroscopic data by machine learning and
    model-based pre-processing, doi:10.1016/j.chemolab.2021.104350
    """

    def __init__(
        self,
        reference: np.ndarray = None,
        use_mean: bool = True,
        use_median: bool = False,
        order: int = 2,
        weights: np.ndarray = None,
    ):
        self.reference = reference
        self.use_mean = use_mean
        self.use_median = use_median
        self.order = order
        self.weights = weights

    def fit(self, X: np.ndarray, y=None) -> "ExtendedMultiplicativeScatterCorrection":
        """
        Fit the transformer to the input data. If no reference is provided, the
        mean or median spectrum will be calculated from the input data.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            The input data to fit the transformer to.

        y : None
            Ignored.

        Returns
        -------
        self : MultiplicativeScatterCorrection
            The fitted transformer.
        """
        # Check that X is a 2D array and has only finite values
        X = check_input(X)

        # Set the number of features
        self.n_features_in_ = X.shape[1]

        # Set the fitted attribute to True
        self._is_fitted = True

        # Check that the length of the reference is the same as the number of features
        if self.reference is not None:
            if len(self.reference) != self.n_features_in_:
                raise ValueError(
                    f"Expected {self.n_features_in_} features in reference but got {len(self.reference)}"
                )

        if self.weights is not None:
            if len(self.weights) != self.n_features_in_:
                raise ValueError(
                    f"Expected {self.n_features_in_} features in weights but got {len(self.weights)}"
                )

        # Set the reference
        if self.reference is not None:
            self.reference_ = np.array(self.reference)
            self.indices_ = self._calculate_indices(self.reference_)
            self.A_ = self._calculate_A(self.indices_, self.reference_)
            self.weights_ = np.array(self.weights)
            return self

        if self.use_median:
            self.reference_ = np.median(X, axis=0)
            self.indices_ = self._calculate_indices(X[0])
            self.A_ = self._calculate_A(self.indices_, self.reference_)
            self.weights_ = np.array(self.weights)
            return self

        if self.use_mean:
            self.reference_ = X.mean(axis=0)
            self.indices_ = self._calculate_indices(X[0])
            self.A_ = self._calculate_A(self.indices_, self.reference_)
            self.weights_ = np.array(self.weights)
            return self

        raise ValueError("No reference was provided")

    def transform(self, X: np.ndarray, y=None) -> np.ndarray:
        """
        Transform the input data by applying the multiplicative scatter
        correction.

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

        # Calculate the extended multiplicative scatter correction
        X_ = X.copy()

        if self.weights is None:
            for i, x in enumerate(X_):
                X_[i] = self._calculate_emsc(x)
            return X_.reshape(-1, 1) if X_.ndim == 1 else X_

        if self.weights is not None:
            for i, x in enumerate(X_):
                X_[i] = self._calculate_weighted_emsc(x)
            return X_.reshape(-1, 1) if X_.ndim == 1 else X_

    def _calculate_weighted_emsc(self, x):
        reg = np.linalg.lstsq(
            np.diag(self.weights_) @ self.A_, x * self.weights_, rcond=None
        )[0]
        x_ = (x - np.dot(self.A_[:, 0:-1], reg[0:-1])) / reg[-1]
        return x_

    def _calculate_emsc(self, x):
        reg = np.linalg.lstsq(self.A_, x, rcond=None)[0]
        x_ = (x - np.dot(self.A_[:, 0:-1], reg[0:-1])) / reg[-1]
        return x_

    def _calculate_indices(self, reference):
        return np.linspace(0, len(reference) - 1, len(reference))

    def _calculate_A(self, indices, reference):
        return np.vstack(
            [[np.power(indices, o) for o in range(self.order + 1)], reference]
        ).T
