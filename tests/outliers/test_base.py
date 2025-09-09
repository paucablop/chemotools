import pytest

from tests.conftest import _DummyModelResiduals


# Test functionality
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
