"""
The :mod:'chemotools.domain_adaption:DirectStandardization'
module implements a Direct Standardization transformer
"""

# Authors: Ruggero Guerrini
# License: MIT

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class DirectStandardization(BaseEstimator, TransformerMixin):
    """
    Description
    -----------
    Implement a direct standardization transformer for the calibration
    transfer application.
    X_master contains the reference measurements acquired
    on the master instrument.
    X_slave contains the corresponding measurements of the same samples
    acquired on the slave instrument.
    The transformer estimates a mapping from the slave space to
    the master space.
    After fitting, new X_slave spectra can be transformed into
    the X_master space.

    Parameters
    ----------
    None

    Attributes
    ----------
    T : np.ndarray of shape (n_features, n_features)
        The pxp matrix that solver the problem X_slave T = X_master
        using the method of least squares

    Examples
    --------
        X_slave = np.random.randn((100,50))
        X_portbale = X_slave*2+5
        DS = DirectStandardization().fit(X_slave,X_master)
        X_slave_transf = DS.transform(X_slave)

    """

    def __init__(self):
        pass

    def fit(self, X_slave: np.ndarray, X_master: np.ndarray) -> "DirectStandardization":
        """
        Fit the DirectStandardization to the input data.

        Parameters
        ----------
        X_slave : np.ndarray of shape (n_samples, n_features)
            The slave data
        X_master : np.ndarray of shape (n_samples, n_features)
            The master data

        Returns
        -------
        self : DirectStandardization
            The fitted model.
        """
        X_slave = np.asarray(X_slave, dtype=float)
        X_master = np.asarray(X_master, dtype=float)
        if X_master.shape != X_slave.shape:
            raise ValueError("master and slave must have the same dimensions")
        self.T = np.linalg.pinv(X_slave) @ X_master
        return self

    def transform(self, X_slave) -> np.ndarray:
        """
        Transform the slave data

        Parameters
        ----------
        X_slave : np.ndarray of shape (n_samples, n_features)
            The input data to transform

        Returns
        -------
        X_transf : np.ndarray of shape (n_samples, n_features)
            The data transformed
        """
        if self.T is None:
            raise RuntimeError("Model not trained. Call train() first.")
        return X_slave @ self.T
