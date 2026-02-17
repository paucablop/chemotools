# -*- coding: utf-8 -*-
"""
Created on Tue Jan 27 09:12:17 2026

@author: Ruggero Guerrini
"""
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.utils.validation import check_X_y, check_array
from sklearn.utils.validation import check_is_fitted
    
class PCR(BaseEstimator, RegressorMixin):
    def __init__(self, n_components=2):
        self.n_components = n_components
    def fit(self, X, y):
        X, y = check_X_y(X,y)        
        if not 1 <= self.n_components <= min(X.shape):
            raise ValueError("n_components must be between 1 and min(X.shape)")
        self.pca_ = PCA(n_components=self.n_components).fit(X)
        T = self.pca_.transform(X)
        self.lr_ = LinearRegression().fit(T, y)
        return self
    def predict(self, X):
        check_is_fitted(self, ["pca_", "lr_"])
        X = check_array(X)
        T = self.pca_.transform(X)
        return self.lr_.predict(T)