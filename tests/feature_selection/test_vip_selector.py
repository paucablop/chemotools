import numpy as np

from chemotools.feature_selection import VIPSelector


# Test compliance with scikit-learn
# TODO: Design test for VIPSelector


# Test functionality
def test_vip_selector_with_fitted_pls(fitted_pls, dummy_data_loader):
    """Test VIPSelector with fitted PLS model.
    Assertion values are taken from reference implementation.
    """
    # Arrange
    X, y = dummy_data_loader
    vip_selector = VIPSelector(model=fitted_pls, threshold=1)

    # Act
    vip_selector.fit(X, y)

    # Assert
    assert np.isclose(vip_selector.feature_scores_[0], 0.9999924871943124)
    assert np.isclose(vip_selector.feature_scores_[1], 1.0000023244107892)
    assert np.isclose(vip_selector.feature_scores_[2], 1.0000051883505163)
