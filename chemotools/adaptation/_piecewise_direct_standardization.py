"""
Test for PiecewiseDirectStandardization
"""

# author: Ruggero Guerrini

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.utils.validation import check_is_fitted, validate_data


class PiecewiseDirectStandardization(TransformerMixin, BaseEstimator):
    """
    Implement a piecewise direct standardization transformer for
    the calibration transfer
    (domain adaption) application.
    y contains the reference measurements acquired
    on the target instrument.
    X contains the corresponding measurements of the same samples
    acquired on the source instrument.
    The transformer use a moving window PLS regressor to estimate the transformation.
    PLS is chosen as in a small window the variables are strongly collerated,
    making the use of OLS not possible.
    After fitting, new X spectra can be transformed into
    the y space.

    Parameters
    ----------
    window_length : int
        Half-width (w) of the local spectral window used in PDS

    n_components : int
        Number of components to keep for PLS model

    scale : bool, default = True
        Whether to scale X and Y in the PLS model

    Reference
    ---------
    .. [1] Wang, Yongdong., Veltkamp,
        D. J., & Kowalski, B. R. (1991).
        Multivariate instrument standardization.
        Analytical Chemistry, 63(23), 2750–2756.

    .. [2] Bouveresse, E.; Massart, D. L. (1996).
        Improvement of the piecewise
        direct standardisation procedure
        for the transfer of NIR spectra
        for multivariate calibration.
        Chemometrics and Intelligent
        Laboratory Systems, 32(2), 201–213.

    Attributes
    ----------
    n_samples_ : number of samples for both target and source matrix

    n_features_ : number of features for both target and source matrix

    pls_ :

    Examples
    --------
        X = np.random.randn((100,50))
        X_portbale = X*2+5
        PDS = PiecewiseDirectStandardization().fit(X,y)
        X_transf = PDS.transform(X)

    """



    def __init__(
        self,
        window_length: int = 25,
        n_components: int = 2,
        scale: bool = True,
    ):
        self.window_length = window_length
        self.n_components = n_components
        self.scale = scale

    def fit(
        self, X: np.ndarray, y: np.ndarray | None = None
    ) -> "PiecewiseDirectStandardization":
        """
        Fit the PiecewiseDirectStandardization to the input data.

        IMPORTANT
        To preserve the compatibility with scikitlearn, the case
        where y is None or its shape are not equal to X are accepted.
        In this case the PDS is applied to the same data matrix (X -> X)

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            The source data

        y : np.ndarray of shape (n_samples, n_features)
            The target data

        Returns
        -------
        self : PiecewiseDirectStandardization
            The PDS transformer.
        """

        # validate_data
        X = validate_data(self, X, ensure_2d=True, reset=True, dtype=np.float64)

        # To protect from .ndim
        if y is not None:
            y = np.asarray(y)

        # Real case
        if y is not None and y.ndim == 2:
            # Fundamental checks
            if X.shape != y.shape:
                raise ValueError("target and source matrix must have same shape!")
            if self.window_length is None:
                raise ValueError("window_length must be specified")
            if self.n_components is None:
                raise ValueError("n_components must be specified")
            if self.n_components > self.window_length + 1:
                raise ValueError(
                    "n_components must be smaller or equal to window_length"
                )
            _, y = validate_data(
                self,
                X,
                y,
                ensure_2d=True,
                reset=False,
                dtype=np.float64,
                multi_output=True,
            )
            n, p = y.shape
            self.n_samples_ = n
            self.n_features_ = p

            self.pls_ = []
            for i in range(p):
                # close to the edge avoid errors
                l_lim = max(0, i - self.window_length)
                r_lim = min(p, i + self.window_length + 1)
                model = PLSRegression(
                    n_components=self.n_components,
                    scale=self.scale,
                ).fit(X[:, l_lim:r_lim], y[:, i])
                params = {
                    "x_mean_": model._x_mean,
                    "coef_": model.coef_,
                    "intercept_": model.intercept_,
                }
                self.pls_.append(params)
            return self
        else:
            y = X
            n, p = y.shape
            self.n_samples_ = n
            self.n_features_ = p

            self.pls_ = []
            for i in range(p):
                # close to the edge avoid errors
                l_lim = max(0, i - self.window_length)
                r_lim = min(p, i + self.window_length + 1)
                model = PLSRegression(
                    n_components=self.n_components,
                    scale=self.scale,
                ).fit(X[:, l_lim:r_lim], y[:, i])
                params = {
                    "x_mean_": model.x_mean_,
                    "coef_": model.coef_,
                    "intercept_": model.intercept_,
                }
                self.pls_.append(params)
            print("Error Method: You must have two set of data not only one")
            return self

    def transform(self, X_new) -> np.ndarray:
        """
        Use the trained model to transform the source data

        Parameters
        ----------
        X_new : np.ndarray of shape (n_samples, n_features)
            The input data to transform

        Returns
        -------
        X_transformed : np.ndarray of shape (n_samples, n_features)
            The data transformed
        """
        # Check the data
        X_new = validate_data(
            self,
            X_new,
            ensure_2d=True,
            reset=False,
            dtype=np.float64,
        )
        # Verify that the model was trained
        check_is_fitted(self)
        X_transformed = np.zeros(X_new.shape)
        for i in range(self.n_features_):
            # close to the edge avoid errors
            l_lim = max(0, i - self.window_length)
            r_lim = min(self.n_features_, i + self.window_length + 1)

            X = X_new[:, l_lim:r_lim] - self.pls_[i]["x_mean_"]
            X_transformed[:, i] = (
                X @ self.pls_[i]["coef_"].T + self.pls_[i]["intercept_"]
            ).ravel()
        return X_transformed
