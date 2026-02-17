import pytest

from tests.conftest import _DummyModelResiduals


# Test functionality
# Invalid model type
def test_invalid_model_raises_error(dummy_data_loader):
    # Arrange
    X, y = dummy_data_loader

    # Act & Assert
    with pytest.raises((TypeError, ValueError)):
        _DummyModelResiduals(1, confidence=0.95).fit(X, y)


# PCA
def test_model_instantiation_with_fitted_pca(fitted_pca, dummy_data_loader):
    # Arrange
    X, y = dummy_data_loader

    # Act
    model_residuals = _DummyModelResiduals(fitted_pca, confidence=0.95).fit(X, y)

    # Assert
    assert model_residuals.n_features_in_ == fitted_pca.n_features_in_
    assert model_residuals.n_components_ == fitted_pca.n_components_


def test_unfitted_pca_raises_error(unfitted_pca, dummy_data_loader):
    # Arrange
    X, y = dummy_data_loader

    # Act & Assert
    with pytest.raises(ValueError, match=".*not fitted.*"):
        _DummyModelResiduals(unfitted_pca, confidence=0.95).fit(X, y)


# PLSRegression
def test_model_instantiation_with_fitted_pls(fitted_pls, dummy_data_loader):
    # Arrange
    X, y = dummy_data_loader

    # Act
    model_residuals = _DummyModelResiduals(fitted_pls, confidence=0.95).fit(X, y)

    # Assert
    assert model_residuals.n_features_in_ == fitted_pls.n_features_in_
    assert model_residuals.n_components_ == fitted_pls.n_components


def test_unfitted_pls_raises_error(unfitted_pls, dummy_data_loader):
    # Arrange
    X, y = dummy_data_loader

    # Act & Assert
    with pytest.raises(ValueError, match=".*not fitted.*"):
        _DummyModelResiduals(unfitted_pls, confidence=0.95).fit(X, y)


# Pipeline
def test_model_instantiation_with_fitted_pipeline_pca(
    fitted_pipeline_pca, dummy_data_loader
):
    # Arrange
    X, y = dummy_data_loader

    # Act
    model_residuals = _DummyModelResiduals(fitted_pipeline_pca, confidence=0.95).fit(
        X, y
    )

    # Assert
    assert model_residuals.n_features_in_ == fitted_pipeline_pca[-1].n_features_in_
    assert model_residuals.n_components_ == fitted_pipeline_pca[-1].n_components_


def test_unfitted_pipeline_raises_error(unfitted_pipeline, dummy_data_loader):
    # Arrange
    X, y = dummy_data_loader

    # Act & Assert
    with pytest.raises(ValueError, match=".*not fitted.*"):
        _DummyModelResiduals(unfitted_pipeline, confidence=0.95).fit(X, y)


def test_pipeline_with_invalid_model_raises_error(invalid_pipeline, dummy_data_loader):
    # Arrange
    X, y = dummy_data_loader

    # Act & Assert
    with pytest.raises(
        TypeError,
        match=(
            ".*Model must be _BasePCA, _PLS, or a Pipeline"
            " ending with one of these types.*"
        ),
    ):
        _DummyModelResiduals(invalid_pipeline, confidence=0.95).fit(X, y)


# Test confidence level
def test_invalid_confidence_raises_error(fitted_pca, dummy_data_loader):
    # Arrange
    X, y = dummy_data_loader

    # Act & Assert
    with pytest.raises(ValueError, match="confidence"):
        _DummyModelResiduals(fitted_pca, confidence=1.5).fit(X, y)  # Out of bounds
    with pytest.raises(ValueError, match="confidence"):
        _DummyModelResiduals(fitted_pca, confidence=-0.5).fit(X, y)  # Out of bounds
