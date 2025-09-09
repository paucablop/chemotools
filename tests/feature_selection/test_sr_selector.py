import numpy as np
import pytest

from sklearn.exceptions import NotFittedError

from chemotools.feature_selection import SRSelector


# Test compliance with scikit-learn
# TODO: Design test for SRSelector


# Test functionality
def test_sr_selector_with_fitted_pls(fitted_pls, dummy_data_loader):
    """Test SRSelector with fitted PLS model.
    Assertion values are taken from reference implementation.
    """
    # Arrange
    X, y = dummy_data_loader
    sr_selector = SRSelector(model=fitted_pls, threshold=1)

    # Act
    sr_selector.fit(X, y)

    # Assert
    assert np.isclose(sr_selector.feature_scores_[0], 130018.58664317118)
    assert np.isclose(sr_selector.feature_scores_[1], 592301.5646850762)
    assert np.isclose(sr_selector.feature_scores_[2], 961737.1359068436)


def test_sr_selector_with_fitted_pca(fitted_pca, dummy_data_loader):
    """Test SRSelector with fitted PCA model.
    Should raise TypeError since PCA is not an accepted model type.
    """
    # Arrange, Act & Assert
    with pytest.raises(TypeError, match=".*not a valid model.*"):
        SRSelector(model=fitted_pca, threshold=1)


def test_sr_selector_with_unfitted_model(unfitted_pls, dummy_data_loader):
    """Test SRSelector with unfitted model.
    Should raise NotFittedError since the model is not fitted.
    """
    # Arrange
    X, y = dummy_data_loader

    # Act & Assert
    with pytest.raises(NotFittedError, match=".*not fitted.*"):
        SRSelector(model=unfitted_pls, threshold=1)
