import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from chemotools.utils.check_inputs import check_input

class ExtendedMultiplicativeScatterCorrection(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.ref_spec = None
        self.coeffs = None

    def fit(self, X, ref_spec=None):
        if ref_spec is None:
            # Use mean spectrum as reference if none provided
            ref_spec = np.mean(X, axis=0)
        self.ref_spec = ref_spec

        # Calculate the mean spectrum
        mean_spec = np.mean(X, axis=0)

        # Fit a linear model to the reference spectrum
        coeffs = np.polyfit(mean_spec, ref_spec, deg=1)
        self.coeffs = coeffs

    def transform(self, X):
        # Divide the spectra by the linear model
        X_emsc = X / np.polyval(self.coeffs, X.mean(axis=1))
        return X_emsc

    def fit_transform(self, X, ref_spec=None):
        self.fit(X, ref_spec=ref_spec)
        X_emsc = self.transform(X)
        return X_emsc
