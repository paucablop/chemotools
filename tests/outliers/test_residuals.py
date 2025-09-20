import numpy as np
import pytest

from sklearn.decomposition import PCA


from chemotools.outliers import (
    DModX,
    HotellingT2,
    QResiduals,
)


# Test functionality
# Parametrized test
@pytest.mark.parametrize(
    "model_class, kwargs, n_components, expected_critical_value, expected_prediction_inlier, expected_prediction_outlier",
    [
        # DModX with different PCA components
        (
            DModX,
            {"confidence": 0.95},
            1,
            1.7576128734793073,
            0.08797542463276586,
            755.0702354976111,
        ),
        # QResiduals with different methods & PCA components
        (
            QResiduals,
            {"confidence": 0.95, "method": "chi-square"},
            2,
            0.16965388642221613,
            0.00050853,
            10.73161499,
        ),
        (
            QResiduals,
            {"confidence": 0.95, "method": "jackson-mudholkar"},
            2,
            0.07479919388489323,
            0.00050853,
            10.73161499,
        ),
        (
            QResiduals,
            {"confidence": 0.95, "method": "percentile"},
            2,
            0.11543872873258751,
            0.00050853,
            10.73161499,
        ),
        # HotellingT2 with different PCA components
        (
            HotellingT2,
            {"confidence": 0.95},
            2,
            6.2414509854897675,
            0.0013293,
            944286.28269795,
        ),  # Example for 2 components
    ],
)
def test_outlier_detection_models(
    dummy_data_loader,
    model_class,
    kwargs,
    n_components,
    expected_critical_value,
    expected_prediction_inlier,
    expected_prediction_outlier,
):
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

    inlier_flag = model.predict(test_point_inlier)
    outlier_flag = model.predict(test_point_outlier)

    # Assert model attributes
    assert model.confidence == kwargs["confidence"], (
        "Confidence value should match input"
    )
    assert np.isclose(model.critical_value_, expected_critical_value), (
        f"Critical value mismatch for {model_class.__name__} with {n_components} components"
    )
    assert model.n_features_in_ == 3, "Number of input features should be 3"
    assert model.n_components_ == n_components, (
        f"Number of model components should be {n_components}"
    )
    assert model.n_samples_ == 100, "Number of samples should be 100"

    # Assert predictions
    assert prediction_inlier < model.critical_value_, (
        "Test point should not be an outlier"
    )
    assert prediction_inlier < np.max(residuals), (
        "Prediction should be within residual range"
    )
    assert np.isclose(prediction_inlier, expected_prediction_inlier), (
        "Prediction value mismatch"
    )
    assert prediction_outlier > model.critical_value_, "Test point should be an outlier"
    assert prediction_outlier > np.max(residuals), (
        "Prediction should be outside residual range"
    )
    assert np.isclose(prediction_outlier, expected_prediction_outlier), (
        "Prediction value mismatch"
    )

    # Assert outlier flags
    assert inlier_flag == 1, "Inlier flag should be 1"
    assert outlier_flag == -1, "Outlier flag should be -1"
