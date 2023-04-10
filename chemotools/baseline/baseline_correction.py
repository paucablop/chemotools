
from scipy.interpolate import UnivariateSpline
from scipy.signal import savgol_filter, argrelextrema
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array


import numpy as np
from scipy.sparse import csc_matrix, eye, diags
from scipy.sparse.linalg import spsolve

 


def _drift_correct_spectrum(x: np.ndarray) -> np.ndarray:

    # Drift correct the spectrum based on the rubber band method
    # Can take any array and returns it drift corrected
    # Find the x values at the edges of the spectrum
    y1: float = x[0]
    y2: float = x[-1]

    # Find the max and min wavenumebrs
    x1 = 0
    x2 = len(x)
    x_range = np.linspace(x1, x2, x2)

    # Calculate the straight line initial and end point
    slope = (y2 - y1) / (x2 - x1)
    intercept = y1 - slope * x1
    drift_correction = slope * x_range + intercept

    # Return the drift corrected spectrum
    return x - drift_correction

class TwoPointDriftCorrection(BaseEstimator, TransformerMixin):

    def fit(self, X: np.ndarray, y=0):
        self.shape_ = X.shape
        return self

    def transform(self, X: np.ndarray, y=0, copy=True) -> np.ndarray:
        X_ = check_array_type(X, copy)
        for i, x in enumerate(X_):
            X_[i, :] = _drift_correct_spectrum(x)
        return X_


class BaseLineCorrection(BaseEstimator, TransformerMixin):

    def __init__(self, order, indices) -> None:
        self.order = order
        self.indices = indices

    def fit(self, X, y=None):
        self.shape_ = X.shape
        return self

    def transform(self, X: np.ndarray, y=None, copy=True):
        X_ = check_array_type(X, copy)
        intensity = X_[0][self.indices]
        poly = np.polyfit(self.indices, intensity, self.order)
        baseline = [np.polyval(poly, i) for i in range(0, len(X_[0]))]      
        return X_[0] - baseline


class BaseLineSplineCorrection(BaseEstimator, TransformerMixin):

    def __init__(self, order, indices) -> None:
        self.order = order
        self.indices = indices

    def fit(self, X, y=None):
        self.shape_ = X.shape
        return self

 

    def transform(self, X: np.ndarray, y=None, copy=True):
        X_ = check_array_type(X, copy)
        intensity = X_[0][self.indices]
        spl = UnivariateSpline(self.indices, intensity, k=self.order)
        baseline = spl(range(len(X_[0])))      
        return X_[0] - baseline

 

class AutomaticBaselineCorrection(BaseEstimator, TransformerMixin):

    def __init__(self, spline_order, local_min_order) -> None:
        self.spline_order = spline_order
        self.local_min_order = local_min_order

    def fit(self, X, y=None):
        self.shape_ = X.shape
        self.indices = argrelextrema(X[0], np.less, order=self.local_min_order)[0]