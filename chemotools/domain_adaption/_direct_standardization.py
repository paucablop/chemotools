"""
The :mod:'chemotools.domain_adaption:DirectStandardization' module implements a Direct Standardization transformer
"""

# Authors: Ruggero Guerrini
# License: MIT

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class DirectStandardization(BaseEstimator, TransformerMixin):
    """
    Description
    Implement a Direct Standardization transformer for calibration transfer application

    Parameters
    ----------
    None

    Attributes
    ----------
    T : np.ndarray of shape (n_features, n_features)
        The pxp matrix that solver the problem X_bench T = X_portable
        using the method of least squares

    Examples
    --------
        X_bench = np.random.randn((100,50))
        X_portbale = X_bench*2+5
        DS = DirectStandardization().fit(X_bench,X_portable)
        X_bench_transf = DS.transform(X_bench)
    
    """
    def __init__(self):
        pass
    def fit(self, X_bench: np.ndarray, X_portable: np.ndarray) -> DirectStandardization:
        """
        Fit the DirectStandardization to the input data.

        Parameters
        ----------
        X_bench : np.ndarray of shape (n_samples, n_features)
            The bench data
        X_portable : np.ndarray of shape (n_samples, n_features)
            The portable data

        Returns
        -------
        self : DirectStandardization
            The fitted model.
        """
        X_bench = np.asarray(X_bench, dtype=float)
        X_portable = np.asarray(X_portable, dtype=float)
        if X_portable.shape != X_bench.shape:
            raise ValueError ("Portable and bench must have the same dimensions")
        self.T = np.linalg.pinv(X_bench) @ X_portable
        return self
    def transform(self,X_bench) -> np.ndarray:
        """
        Fit the Direct standardization transforme and 
        transform the bench data 

        Parameters
        ----------
        X+bench : np.ndarray of shape (n_samples, n_features)
            The input data to transform
        
        Returns
        -------
        X_transf : np.ndarray of shape (n_samples, n_features)
            The data transformed
        """
        if self.T is None:
            raise RuntimeError("Model not trained. Call train() first.")
        return X_bench @ self.T 