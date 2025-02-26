import numpy as np
import pytest

from sklearn.decomposition import PCA


from chemotools.outliers import (
    HotellingT2,
    QResiduals,
    DModX,
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


### TEST OUTLIER DETECTION MODELS ###
# Parametrized test
@pytest.mark.parametrize("model_class, kwargs, n_components, expected_critical_value, expected_prediction_inlier, expected_prediction_outlier", [
    # DModX with different PCA components
    (DModX, {"confidence": 0.95}, 1, 14.124446891825524, 0.01684261, 144.55574493),

    # QResiduals with different methods & PCA components
    (QResiduals, {"confidence": 0.95, "method": "chi-square"}, 2, 0.16965388642221613, 0.00050853, 10.73161499),
    (QResiduals, {"confidence": 0.95, "method": "jackson-mudholkar"}, 2, 0.07479919388489323, 0.00050853, 10.73161499),
    (QResiduals, {"confidence": 0.95, "method": "percentile"}, 2, 0.11543872873258751, 0.00050853, 10.73161499),

    # HotellingT2 with different PCA components
    (HotellingT2, {"confidence": 0.95}, 2, 6.2414509854897675, 0.0013293, 944286.28269795),  # Example for 2 components
])
def test_outlier_detection_models(dummy_data_loader, model_class, kwargs, n_components, expected_critical_value, expected_prediction_inlier, expected_prediction_outlier):
    """Test different outlier detection models with various PCA components and outlier test methods."""
    
    # Arrange
    X, _ = dummy_data_loader  # Load dummy data
    pca = PCA(n_components=n_components).fit(X)  # Dynamic PCA component selection
    
    model = model_class(model=pca, **kwargs)  # Instantiate model with params

    test_point_inlier = np.array([[50, 100, 150]])  
    test_point_outlier = np.array([[200, 50, 400]])  

    # Act
    model.fit(X)
    residuals = model.predict_residuals(X)
    prediction_inlier = model.predict_residuals(test_point_inlier)[0]
    prediction_outlier = model.predict_residuals(test_point_outlier)[0]

    # Assert model attributes
    assert model.confidence == kwargs["confidence"], "Confidence value should match input"
    assert np.isclose(model.critical_value_, expected_critical_value), f"Critical value mismatch for {model_class.__name__} with {n_components} components"
    assert model.n_features_in_ == 3, "Number of input features should be 3"
    assert model.n_components_ == n_components, f"Number of model components should be {n_components}"
    assert model.n_samples_ == 100, "Number of samples should be 100"

    # Assert predictions
    assert prediction_inlier < model.critical_value_, "Test point should not be an outlier"
    assert prediction_inlier < np.max(residuals), "Prediction should be within residual range"
    assert np.isclose(prediction_inlier, expected_prediction_inlier), "Prediction value mismatch"
    assert prediction_outlier > model.critical_value_, "Test point should be an outlier"
    assert prediction_outlier > np.max(residuals), "Prediction should be outside residual range"
    assert np.isclose(prediction_outlier, expected_prediction_outlier), "Prediction value mismatch"
