"""Tests for ExternalParameterOrthogonalization."""

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.utils.estimator_checks import check_estimator

from chemotools.cross_decomposition import ExternalParameterOrthogonalization

# --- Fixtures ---


@pytest.fixture()
def epo_dataset():
    """Create a dataset with known signal and external nuisance directions."""
    rng = np.random.default_rng(0)
    n_samples, n_features = 40, 10

    signal_scores = rng.normal(size=n_samples)
    nuisance_scores = rng.normal(size=n_samples)

    signal_vector = rng.normal(size=n_features)
    signal_vector /= np.linalg.norm(signal_vector)

    nuisance_vector = rng.normal(size=n_features)
    nuisance_vector -= signal_vector * (signal_vector @ nuisance_vector)
    nuisance_vector /= np.linalg.norm(nuisance_vector)

    X = (
        2.0 * np.outer(signal_scores, signal_vector)
        + 1.5 * np.outer(nuisance_scores, nuisance_vector)
        + 1e-3 * rng.normal(size=(n_samples, n_features))
    )
    X_external = (
        2.0 * np.outer(signal_scores, signal_vector)
        + 8.0 * np.outer(nuisance_scores, nuisance_vector)
        + 1e-3 * rng.normal(size=(n_samples, n_features))
    )

    return X, X_external, nuisance_vector


@pytest.fixture()
def paired_epo_dataset():
    """Create a paired-sample dataset with known nuisance direction."""
    rng = np.random.default_rng(0)
    n_samples, n_features = 12, 8
    sample_ids = np.repeat(np.arange(n_samples), 2)
    condition = np.tile(np.array([-1.0, 1.0]), n_samples)

    signal_vector = rng.normal(size=n_features)
    signal_vector /= np.linalg.norm(signal_vector)
    nuisance_vector = rng.normal(size=n_features)
    nuisance_vector -= signal_vector * (signal_vector @ nuisance_vector)
    nuisance_vector /= np.linalg.norm(nuisance_vector)

    latent_signal = rng.normal(size=n_samples)
    base = 3.0 * np.outer(latent_signal, signal_vector)
    base = np.repeat(base, 2, axis=0)

    X = base + 0.5 * np.outer(condition, nuisance_vector)
    X_external = base + 4.0 * np.outer(condition, nuisance_vector)

    return X, X_external, sample_ids, nuisance_vector


# Test compliance with scikit-learn


def test_compliance_epo():
    """Check sklearn estimator compliance for the EPO transformer."""
    # Arrange
    transformer = ExternalParameterOrthogonalization()

    # Act & Assert
    check_estimator(transformer)


# Test functionality


class TestEPOFitTransform:
    """Test basic fit / transform behaviour."""

    def test_fit_transform_matches_fit_then_transform(self, epo_dataset):
        """Ensure `fit_transform()` matches `fit()` followed by `transform()`."""
        # Arrange
        X, X_external, _ = epo_dataset

        # Act
        fit_transformer = ExternalParameterOrthogonalization(n_components=1)
        Xt_fit_transform = fit_transformer.fit_transform(X, X_external=X_external)

        transformer = ExternalParameterOrthogonalization(n_components=1)
        transformer.fit(X, X_external=X_external)
        Xt_transform = transformer.transform(X)

        # Assert
        np.testing.assert_allclose(Xt_fit_transform, Xt_transform)

    def test_transform_before_fit_raises_not_fitted_error(self):
        """Ensure calling `transform()` before `fit()` raises `NotFittedError`."""
        # Arrange
        X = np.ones((4, 3), dtype=float)
        transformer = ExternalParameterOrthogonalization()

        # Act / Assert
        with pytest.raises(NotFittedError):
            transformer.transform(X)

    def test_fit_without_external_leaves_data_unchanged(self):
        """Default to a no-op transform when `X_external` is not provided."""
        # Arrange
        rng = np.random.default_rng(0)
        X = rng.normal(size=(8, 5))
        transformer = ExternalParameterOrthogonalization(n_components=1)

        # Act
        Xt = transformer.fit_transform(X)

        # Assert
        np.testing.assert_allclose(Xt, X)
        np.testing.assert_allclose(transformer.P_epo_, np.eye(X.shape[1]))
        np.testing.assert_allclose(transformer.mean_X_, X.mean(axis=0))


class TestEPONuisanceRemoval:
    """Test that EPO effectively removes nuisance variation."""

    def test_transform_reduces_known_nuisance_variation(self, epo_dataset):
        """Reduce variance along a known nuisance direction after EPO transform."""
        # Arrange
        X, X_external, nuisance_vector = epo_dataset
        transformer = ExternalParameterOrthogonalization(n_components=1)
        X_centered = X - X.mean(axis=0)
        before = np.std(X_centered @ nuisance_vector)

        # Act
        Xt = transformer.fit_transform(X, X_external=X_external)
        Xt_centered = Xt - Xt.mean(axis=0)
        after = np.std(Xt_centered @ nuisance_vector)

        # Assert
        assert after < before * 0.25

    def test_sample_ids_reduce_within_sample_external_differences(
        self, paired_epo_dataset
    ):
        """Use `sample_ids` to suppress paired external-condition differences."""
        # Arrange
        X, X_external, sample_ids, nuisance_vector = paired_epo_dataset
        transformer = ExternalParameterOrthogonalization(n_components=1)
        before = np.mean(
            [abs((X[i] - X[i + 1]) @ nuisance_vector) for i in range(0, len(X), 2)]
        )

        # Act
        Xt = transformer.fit_transform(X, X_external=X_external, sample_ids=sample_ids)
        after = np.mean(
            [abs((Xt[i] - Xt[i + 1]) @ nuisance_vector) for i in range(0, len(Xt), 2)]
        )

        # Assert
        assert after < before * 0.1
