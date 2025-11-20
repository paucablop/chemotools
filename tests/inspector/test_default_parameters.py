"""Tests for inspector default parameters with varying component counts."""

import numpy as np
import pytest
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from chemotools.inspector import PCAInspector, PLSRegressionInspector


@pytest.fixture
def sample_data():
    """Generate sample data for testing."""
    np.random.seed(42)
    n_samples = 50
    n_features = 100
    X = np.random.randn(n_samples, n_features)
    y = np.random.randn(n_samples)
    return X, y


class TestPCAInspectorDefaultParameters:
    """Test PCA inspector default parameters adapt to component count."""

    @pytest.mark.parametrize("n_components", [1, 2, 3, 5, 10])
    def test_inspect_with_default_parameters(self, sample_data, n_components):
        """Test that inspect() works with default parameters for various component counts."""
        X, y = sample_data

        model = make_pipeline(StandardScaler(), PCA(n_components=n_components))
        model.fit(X)

        inspector = PCAInspector(model, X, y)

        # This should not raise any errors
        figs = inspector.inspect()

        # Verify we got some figures back
        assert len(figs) > 0
        assert "loadings" in figs
        assert "variance" in figs
        assert "distances" in figs

        # For 1-2 components, we should have 1 scores plot
        # For 3+ components, we should have 2 scores plots
        if n_components <= 2:
            assert "scores_1" in figs
            assert "scores_2" not in figs
        else:
            assert "scores_1" in figs
            assert "scores_2" in figs

    def test_single_component_plots_1d_scores(self, sample_data):
        """Test that single-component model creates 1D scores plot."""
        X, y = sample_data

        model = make_pipeline(StandardScaler(), PCA(n_components=1))
        model.fit(X)

        inspector = PCAInspector(model, X, y)
        figs = inspector.inspect()

        # Should succeed and create a 1D plot
        assert "scores_1" in figs

    def test_manual_invalid_components_raises_helpful_error(self, sample_data):
        """Test that manually specifying invalid components raises helpful error."""
        X, y = sample_data

        model = make_pipeline(StandardScaler(), PCA(n_components=2))
        model.fit(X)

        inspector = PCAInspector(model, X, y)

        # Trying to plot component 2 (index) when only components 0 and 1 exist
        with pytest.raises(ValueError, match="Component index 2 is invalid"):
            inspector.inspect(components_scores=(0, 2))

        # Trying to plot component 5 in loadings when only 2 components exist
        with pytest.raises(ValueError, match="Component index 5 is out of bounds"):
            inspector.inspect(loadings_components=[0, 1, 5])


class TestPLSRegressionInspectorDefaultParameters:
    """Test PLS regression inspector default parameters adapt to component count."""

    @pytest.mark.parametrize("n_components", [1, 2, 3, 5, 10])
    def test_inspect_with_default_parameters(self, sample_data, n_components):
        """Test that inspect() works with default parameters for various component counts."""
        X, y = sample_data

        model = make_pipeline(
            StandardScaler(), PLSRegression(n_components=n_components)
        )
        model.fit(X, y)

        inspector = PLSRegressionInspector(model, X, y)

        # This should not raise any errors
        figs = inspector.inspect()

        # Verify we got some figures back
        assert len(figs) > 0
        assert "loadings_x" in figs
        assert "loadings_weights" in figs
        assert "loadings_rotations" in figs
        assert "regression_coefficients" in figs
        assert "distances_hotelling_q" in figs
        assert "distances_leverage_studentized" in figs

        # For 1-2 components, we should have 1 scores plot
        # For 3+ components, we should have 2 scores plots
        if n_components <= 2:
            assert "scores_1" in figs
            assert "scores_2" not in figs
        else:
            assert "scores_1" in figs
            assert "scores_2" in figs

    def test_single_component_plots_1d_scores(self, sample_data):
        """Test that single-component model creates 1D scores plot."""
        X, y = sample_data

        model = make_pipeline(StandardScaler(), PLSRegression(n_components=1))
        model.fit(X, y)

        inspector = PLSRegressionInspector(model, X, y)
        figs = inspector.inspect()

        # Should succeed and create a 1D plot
        assert "scores_1" in figs

    def test_manual_invalid_components_raises_helpful_error(self, sample_data):
        """Test that manually specifying invalid components raises helpful error."""
        X, y = sample_data

        model = make_pipeline(StandardScaler(), PLSRegression(n_components=2))
        model.fit(X, y)

        inspector = PLSRegressionInspector(model, X, y)

        # Trying to plot component 2 (index) when only components 0 and 1 exist
        with pytest.raises(ValueError, match="Component index 2 is invalid"):
            inspector.inspect(components_scores=(0, 2))

        # Trying to plot component 5 in loadings when only 2 components exist
        with pytest.raises(ValueError, match="Component index 5 is out of bounds"):
            inspector.inspect(loadings_components=[0, 1, 5])
