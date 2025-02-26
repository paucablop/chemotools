import numpy as np
import pandas as pd
import polars as pl
import pytest

from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


from chemotools.outliers import (
    HotellingT2,
    QResiduals,
    DModX,
    Leverage,
    StudentizedResiduals,
)

from .conftest import _DummyModelResiduals

### TEST BASE CLASS ###
# Invalid model type
def test_invalid_model_raises_error():
    # Arrange & Act & Assert
    with pytest.raises(ValueError, match=".*not a valid model.*"):
        _DummyModelResiduals(1, confidence=0.95)

# PCA
def test_model_instantiation_with_fitted_pca(fitted_pca):
    # Arrange & Act & Assert
    model_residuals = _DummyModelResiduals(fitted_pca, confidence=0.95)
    assert model_residuals.n_features_in_ == fitted_pca.n_features_in_
    assert model_residuals.n_components_ == fitted_pca.n_components_

def test_unfitted_pca_raises_error(unfitted_pca):
    # Arrange & Act & Assert
    with pytest.raises(ValueError, match=".*not fitted.*"):
        _DummyModelResiduals(unfitted_pca, confidence=0.95)

# PLSRegression
def test_model_instantiation_with_fitted_pls(fitted_pls):
    # Arrange & Act & Assert
    model_residuals = _DummyModelResiduals(fitted_pls, confidence=0.95)
    assert model_residuals.n_features_in_ == fitted_pls.n_features_in_
    assert model_residuals.n_components_ == fitted_pls.n_components

def test_unfitted_pls_raises_error(unfitted_pls):
    # Arrange & Act & Assert
    with pytest.raises(ValueError, match=".*not fitted.*"):
        _DummyModelResiduals(unfitted_pls, confidence=0.95)

# Pipeline
def test_model_instantiation_with_fitted_pipeline_pca(fitted_pipeline_pca):
    # Arrange & Act & Assert
    model_residuals = _DummyModelResiduals(fitted_pipeline_pca, confidence=0.95)
    assert model_residuals.n_features_in_ == fitted_pipeline_pca[-1].n_features_in_
    assert model_residuals.n_components_ == fitted_pipeline_pca[-1].n_components_

def test_unfitted_pipeline_raises_error(unfitted_pipeline):
    # Arrange & Act & Assert
    with pytest.raises(ValueError, match=".*not fitted.*"):
        _DummyModelResiduals(unfitted_pipeline, confidence=0.95)

def test_pipeline_with_invalid_model_raises_error(invalid_pipeline):
    # Arrange & Act & Assert
    with pytest.raises(ValueError, match=".*not a valid model.*"):
        _DummyModelResiduals(invalid_pipeline, confidence=0.95)

# Test confidence level
def test_invalid_confidence_raises_error(fitted_pca):
    # Arrange & Act & Assert
    with pytest.raises(ValueError, match="Confidence must be between 0 and 1"):
        _DummyModelResiduals(fitted_pca, confidence=1.5)  # Out of bounds
    with pytest.raises(ValueError, match="Confidence must be between 0 and 1"):
        _DummyModelResiduals(fitted_pca, confidence=-0.5)  # Out of bounds


# Test DModX
def test_dmodx(dummy_data_loader):
    """Test HotellingT2 class with a fitted PCA model."""
    
    # Arrange
    X, _ = dummy_data_loader  # Load dummy data
    
    pca = PCA(n_components=1).fit(X)
    
    dmodx = DModX(model=pca, confidence=0.95)

    test_point_inlier = np.array([[50, 100, 150]])  
    test_point_outlier = np.array([[200, 50, 400]])  


    # Act
    dmodx.fit(X)
    residuals = dmodx.predict_residuals(X)
    prediction_inlier = dmodx.predict_residuals(test_point_inlier)[0]
    prediction_outlier = dmodx.predict_residuals(test_point_outlier)[0]

    # Assert model attributes
    assert dmodx.confidence == 0.95, "Confidence value should be 0.95"
    assert np.isclose(dmodx.critical_value_, 14.124446891825524), "Critical value mismatch"
    assert dmodx.n_features_in_ == 3, "Number of input features should be 3"
    assert dmodx.n_components_ == 1, "Number of model components should be 2"
    assert dmodx.n_samples_ == 100, "Number of samples should be 101"

    # Assert predictions
    assert prediction_inlier < dmodx.critical_value_, "Test point should not be an outlier"
    assert prediction_inlier < np.max(residuals), "Prediction should be within residual range"
    assert np.isclose(prediction_inlier, 0.01684261), "Prediction value mismatch"
    assert prediction_outlier > dmodx.critical_value_, "Test point should be an outlier"
    assert prediction_outlier > np.max(residuals), "Prediction should be outside residual range"
    assert np.isclose(prediction_outlier, 144.55574493), "Prediction value mismatch"




# Test Q Residuals
@pytest.mark.parametrize("method, expected_critical_value", [
    ("chi-square", 0.16965388642221613),
    ("jackson-mudholkar", 0.07479919388489323),  
    ("percentile", 0.11543872873258751)  
])
def test_q_residuals(dummy_data_loader, method, expected_critical_value):
    """Test Q Residuals with a fitted PCA model."""

    # Arrange
    X, _ = dummy_data_loader  
    
    pca = PCA(n_components=2).fit(X)
    
    q_residuals = QResiduals(model=pca, confidence=0.95, method=method)

    test_point_inlier = np.array([[50, 100, 150]])  
    test_point_outlier = np.array([[200, 50, 400]])  

    # Act
    q_residuals.fit(X)
    residuals = q_residuals.predict_residuals(X)
    prediction_inlier = q_residuals.predict_residuals(test_point_inlier)[0]
    prediction_outlier = q_residuals.predict_residuals(test_point_outlier)[0]

    # Assert model attributes
    assert q_residuals.confidence == 0.95, "Confidence value should be 0.95"
    assert np.isclose(q_residuals.critical_value_, expected_critical_value), "Critical value mismatch"
    assert q_residuals.n_features_in_ == 3, "Number of input features should be 3"
    assert q_residuals.n_components_ == 2, "Number of model components should be 2"
    assert q_residuals.n_samples_ == 100, "Number of samples should be 101"

    # Assert predictions
    assert prediction_inlier < q_residuals.critical_value_, "Test point should not be an outlier"
    assert prediction_inlier < np.max(residuals), "Prediction should be within residual range"
    assert np.isclose(prediction_inlier, 0.00050853), "Prediction value mismatch"  
    assert prediction_outlier > q_residuals.critical_value_, "Test point should be an outlier"
    assert prediction_outlier > np.max(residuals), "Prediction should be outside residual range"
    assert np.isclose(prediction_outlier, 10.73161499), "Prediction value mismatch"


@pytest.mark.parametrize("invalid_method", ["percentil", "wrong-method", "jackson mudholkar", "chi square"])  
def test_q_residuals_invalid_method(dummy_data_loader, invalid_method):
    """Test Q Residuals with invalid method inputs."""

    # Arrange
    X, _ = dummy_data_loader  # Load dummy data
    pca = PCA(n_components=2).fit(X)

    # Act & Assert
    with pytest.raises(ValueError, match="Invalid method.*"):
        QResiduals(model=pca, confidence=0.95, method=invalid_method).fit(X) 



# Test Hotelling T2
def test_hotelling_t2(dummy_data_loader):
    """Test HotellingT2 class with a fitted PCA model."""
    
    # Arrange
    X, _ = dummy_data_loader  # Load dummy data
    
    pca = PCA(n_components=2).fit(X)
    
    hotelling_t2 = HotellingT2(model=pca, confidence=0.95)

    test_point_inlier = np.array([[50, 100, 150]])  # Single test sample
    test_point_outlier = np.array([[200, 300, 400]])  # Single test sample


    # Act
    hotelling_t2.fit(X)
    residuals = hotelling_t2.predict_residuals(X)
    prediction_inlier = hotelling_t2.predict_residuals(test_point_inlier)[0]
    prediction_outlier = hotelling_t2.predict_residuals(test_point_outlier)[0]

    # Assert model attributes
    assert hotelling_t2.confidence == 0.95, "Confidence value should be 0.95"
    assert np.isclose(hotelling_t2.critical_value_, 6.2414509854897675), "Critical value mismatch"
    assert hotelling_t2.n_features_in_ == 3, "Number of input features should be 3"
    assert hotelling_t2.n_components_ == 2, "Number of model components should be 2"
    assert hotelling_t2.n_samples_ == 100, "Number of samples should be 101"

    # Assert predictions
    assert prediction_inlier < hotelling_t2.critical_value_, "Test point should not be an outlier"
    assert prediction_inlier < np.max(residuals), "Prediction should be within residual range"
    assert np.isclose(prediction_inlier, 0.0013293), "Prediction value mismatch"
    assert prediction_outlier > hotelling_t2.critical_value_, "Test point should be an outlier"
    assert prediction_outlier > np.max(residuals), "Prediction should be outside residual range"
    assert np.isclose(prediction_outlier, 147.56313415), "Prediction value mismatch"


