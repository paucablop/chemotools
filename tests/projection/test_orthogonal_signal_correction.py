"""Tests for OrthogonalSignalCorrection."""

import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning, NotFittedError
from sklearn.utils.estimator_checks import check_estimator

import chemotools.projection._orthogonal_signal_correction as osc_module
from chemotools.projection import OrthogonalSignalCorrection


def _make_osc_dataset(
    *,
    n_samples: int = 40,
    n_features: int = 12,
    signal_scale: float = 3.0,
    nuisance_scale: float = 8.0,
    noise_scale: float = 1e-2,
    seed: int = 0,
):
    """Create a stable synthetic dataset with signal and nuisance variation."""
    rng = np.random.default_rng(seed)

    y = np.linspace(-1.0, 1.0, n_samples)
    y_centered = y - y.mean()

    nuisance_scores = rng.normal(size=n_samples)
    nuisance_scores -= y_centered * (
        (y_centered @ nuisance_scores) / (y_centered @ y_centered)
    )
    nuisance_scores /= np.linalg.norm(nuisance_scores)

    signal_vector = rng.normal(size=n_features)
    signal_vector /= np.linalg.norm(signal_vector)

    nuisance_vector = rng.normal(size=n_features)
    nuisance_vector -= signal_vector * (signal_vector @ nuisance_vector)
    nuisance_vector /= np.linalg.norm(nuisance_vector)

    X = (
        signal_scale * np.outer(y_centered, signal_vector)
        + nuisance_scale * np.outer(nuisance_scores, nuisance_vector)
        + noise_scale * rng.normal(size=(n_samples, n_features))
    )
    y_multi = np.column_stack([y, y**2])

    return X, y, y_multi, nuisance_vector


# Test compliance with scikit-learn
def test_compliance_osc():
    """Check sklearn estimator compliance for the OSC transformer."""
    # Arrange
    transformer = OrthogonalSignalCorrection()

    # Act & Assert
    # n_iter_ is per-component (array), but sklearn's check_transformer_n_iter
    # only handles this for its own hardcoded CROSS_DECOMPOSITION list.
    check_estimator(
        transformer,
        expected_failed_checks={
            "check_transformer_n_iter": "n_iter_ is stored per component (array) "
            "instead of scalar"
        },
    )


@pytest.mark.parametrize(
    ("method", "n_components"),
    [("wold", 2), ("sjoblom", 2), ("fearn", 2)],
)
@pytest.mark.parametrize(
    "y_factory",
    [
        lambda rng, n_samples: rng.normal(size=n_samples),
        lambda rng, n_samples: rng.normal(size=(n_samples, 2)),
    ],
)
def test_osc_methods_preserve_expected_shapes(method, n_components, y_factory):
    """Ensure each OSC method produces finite outputs with expected shapes."""
    # Arrange
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 6))
    y = y_factory(rng, X.shape[0])
    transformer = OrthogonalSignalCorrection(
        method=method,
        n_components=n_components,
    )

    # Act
    Xt = transformer.fit_transform(X, y)

    # Assert
    assert Xt.shape == X.shape
    assert transformer.scores_.shape == (X.shape[0], n_components)
    assert transformer.weights_.shape == (X.shape[1], n_components)
    assert transformer.loadings_.shape == (X.shape[1], n_components)
    assert transformer.n_iter_.shape == (n_components,)
    assert np.isfinite(transformer.scores_).all()
    assert np.isfinite(transformer.weights_).all()
    assert np.isfinite(transformer.loadings_).all()
    assert np.isfinite(transformer.n_iter_).all()


@pytest.mark.parametrize(
    ("method", "n_components"),
    [("wold", 2), ("sjoblom", 2), ("fearn", 2)],
)
def test_fit_transform_matches_fit_then_transform(method, n_components):
    """Verify `fit_transform()` matches `fit()` followed by `transform()`."""
    # Arrange
    X, y, _, _ = _make_osc_dataset()

    # Act
    fit_transformer = OrthogonalSignalCorrection(
        method=method,
        n_components=n_components,
    )
    Xt_fit_transform = fit_transformer.fit_transform(X, y)

    transformer = OrthogonalSignalCorrection(
        method=method,
        n_components=n_components,
    )
    transformer.fit(X, y)
    Xt_transform = transformer.transform(X)

    # Assert
    np.testing.assert_allclose(Xt_fit_transform, Xt_transform)


def test_transform_before_fit_raises_not_fitted_error():
    """Ensure calling `transform()` before `fit()` raises `NotFittedError`."""
    # Arrange
    X = np.ones((4, 3), dtype=float)
    transformer = OrthogonalSignalCorrection()

    # Act / Assert
    with pytest.raises(NotFittedError):
        transformer.transform(X)


def test_fit_rejects_single_sample():
    """Reject datasets with fewer than two samples."""
    # Arrange
    X = np.array([[1.0, 2.0, 3.0]])
    y = np.array([1.0])
    transformer = OrthogonalSignalCorrection()

    # Act / Assert
    with pytest.raises(ValueError, match="At least 2 samples are required"):
        transformer.fit(X, y)


@pytest.mark.parametrize("method", ["wold", "sjoblom"])
def test_iterative_methods_warn_when_not_converged(method):
    """Emit a convergence warning when iterative OSC methods hit `max_iter`."""
    # Arrange
    X, y, _, _ = _make_osc_dataset()
    transformer = OrthogonalSignalCorrection(
        method=method,
        n_components=1,
        max_iter=1,
        tol=0.0,
    )

    # Act / Assert
    with pytest.warns(ConvergenceWarning, match="did not converge"):
        transformer.fit(X, y)

    assert transformer.n_iter_.shape == (1,)
    assert transformer.n_iter_[0] == 1


@pytest.mark.parametrize("method", ["wold", "sjoblom"])
def test_iterative_methods_scores_are_approximately_orthogonal_to_centered_y(method):
    """Learn scores that are nearly orthogonal to the centered target."""
    # Arrange
    X, y, _, _ = _make_osc_dataset(
        signal_scale=1.0,
        nuisance_scale=20.0,
        noise_scale=1e-6,
    )
    transformer = OrthogonalSignalCorrection(method=method, n_components=1)

    # Act
    transformer.fit(X, y)
    score = transformer.scores_[:, 0]
    y_centered = y - y.mean()
    relative_projection = abs(score @ y_centered) / (
        np.linalg.norm(score) * np.linalg.norm(y_centered)
    )

    # Assert
    assert relative_projection < 1e-6


@pytest.mark.parametrize("method", ["wold", "sjoblom", "fearn"])
def test_transform_reduces_known_orthogonal_variation(method):
    """Reduce variance along a known nuisance direction after OSC transform."""
    # Arrange
    X, y, _, nuisance_vector = _make_osc_dataset()
    n_components = 2 if method == "fearn" else 1
    transformer = OrthogonalSignalCorrection(method=method, n_components=n_components)
    X_centered = X - X.mean(axis=0)
    before = np.std(X_centered @ nuisance_vector)

    # Act
    Xt = transformer.fit_transform(X, y)
    Xt_centered = Xt - Xt.mean(axis=0)
    after = np.std(Xt_centered @ nuisance_vector)

    # Assert
    assert after < before * 0.25


def test_fearn_supports_multiple_components():
    """Allow Fearn OSC to extract more than one orthogonal component."""
    # Arrange
    X, y, _, _ = _make_osc_dataset()
    transformer = OrthogonalSignalCorrection(method="fearn", n_components=2)

    # Act
    Xt = transformer.fit_transform(X, y)

    # Assert
    assert Xt.shape == X.shape
    assert transformer.scores_.shape == (X.shape[0], 2)
    assert transformer.weights_.shape == (X.shape[1], 2)
    assert transformer.loadings_.shape == (X.shape[1], 2)
    np.testing.assert_array_equal(transformer.n_iter_, np.array([1, 1]))


def test_wold_raises_for_zero_norm_orthogonal_score_vector():
    """Raise when Wold encounters an initially zero orthogonal score vector."""
    # Arrange
    X = np.zeros((6, 4), dtype=float)
    y = np.linspace(0.0, 1.0, 6)
    transformer = OrthogonalSignalCorrection(method="wold", n_components=1)

    # Act / Assert
    with pytest.raises(ValueError, match="zero-norm orthogonal score vector"):
        transformer._wold_method(X, y)


def test_wold_raises_for_zero_norm_weight_vector(monkeypatch):
    """Raise when Wold computes a zero-norm weight vector inside the loop."""
    # Arrange
    X, y, _, _ = _make_osc_dataset()
    transformer = OrthogonalSignalCorrection(method="wold", n_components=1)
    original_norm = osc_module.np.linalg.norm

    def fake_norm(x, *args, **kwargs):
        arr = np.asarray(x)
        if arr.ndim == 1 and arr.shape[0] == X.shape[1]:
            return 0.0
        return original_norm(x, *args, **kwargs)

    monkeypatch.setattr(osc_module.np.linalg, "norm", fake_norm)

    # Act / Assert
    with pytest.raises(ValueError, match="zero-norm weight vector"):
        transformer._wold_method(X, y)


def test_wold_raises_for_zero_norm_orthogonal_score_after_convergence(monkeypatch):
    """Raise when Wold's final orthogonal score degenerates after convergence."""
    # Arrange
    X, y, _, _ = _make_osc_dataset()
    transformer = OrthogonalSignalCorrection(method="wold", n_components=1, tol=np.inf)
    original_isclose = osc_module.np.isclose
    scalar_calls = {"count": 0}

    def fake_isclose(a, b, *args, **kwargs):
        if np.isscalar(a) and np.isscalar(b):
            scalar_calls["count"] += 1
            if scalar_calls["count"] == 3:
                return True
        return original_isclose(a, b, *args, **kwargs)

    monkeypatch.setattr(osc_module.np, "isclose", fake_isclose)

    # Act / Assert
    with pytest.raises(
        ValueError,
        match="zero-norm orthogonal score vector after convergence",
    ):
        transformer._wold_method(X, y)


def test_wold_raises_for_zero_norm_weight_after_convergence(monkeypatch):
    """Raise when Wold's final weight vector degenerates after convergence."""
    # Arrange
    X, y, _, _ = _make_osc_dataset()
    transformer = OrthogonalSignalCorrection(method="wold", n_components=1, tol=np.inf)
    original_isclose = osc_module.np.isclose
    scalar_calls = {"count": 0}

    def fake_isclose(a, b, *args, **kwargs):
        if np.isscalar(a) and np.isscalar(b):
            scalar_calls["count"] += 1
            if scalar_calls["count"] == 4:
                return True
        return original_isclose(a, b, *args, **kwargs)

    monkeypatch.setattr(osc_module.np, "isclose", fake_isclose)

    # Act / Assert
    with pytest.raises(
        ValueError,
        match="zero-norm weight vector after convergence",
    ):
        transformer._wold_method(X, y)


def test_sjoblom_raises_for_zero_norm_orthogonal_score_vector():
    """Raise when Sjöblom encounters an initially zero orthogonal score vector."""
    # Arrange
    X = np.zeros((6, 4), dtype=float)
    y = np.linspace(0.0, 1.0, 6)
    transformer = OrthogonalSignalCorrection(method="sjoblom", n_components=1)

    # Act / Assert
    with pytest.raises(ValueError, match="zero-norm orthogonal score vector"):
        transformer._sjoblom_method(X, y)


def test_sjoblom_raises_for_zero_norm_weight_vector(monkeypatch):
    """Raise when Sjöblom computes a zero-norm weight vector inside the loop."""
    # Arrange
    X, y, _, _ = _make_osc_dataset()
    transformer = OrthogonalSignalCorrection(method="sjoblom", n_components=1)
    original_norm = osc_module.np.linalg.norm

    def fake_norm(x, *args, **kwargs):
        arr = np.asarray(x)
        if arr.ndim == 1 and arr.shape[0] == X.shape[1]:
            return 0.0
        return original_norm(x, *args, **kwargs)

    monkeypatch.setattr(osc_module.np.linalg, "norm", fake_norm)

    # Act / Assert
    with pytest.raises(ValueError, match="zero-norm weight vector"):
        transformer._sjoblom_method(X, y)


def test_sjoblom_raises_for_zero_norm_final_score_vector(monkeypatch):
    """Raise when Sjöblom's final score vector degenerates after projection."""
    # Arrange
    X, y, _, _ = _make_osc_dataset()
    transformer = OrthogonalSignalCorrection(
        method="sjoblom",
        n_components=1,
        tol=np.inf,
    )
    original_isclose = osc_module.np.isclose
    scalar_calls = {"count": 0}

    def fake_isclose(a, b, *args, **kwargs):
        if np.isscalar(a) and np.isscalar(b):
            scalar_calls["count"] += 1
            if scalar_calls["count"] == 3:
                return True
        return original_isclose(a, b, *args, **kwargs)

    monkeypatch.setattr(osc_module.np, "isclose", fake_isclose)

    # Act / Assert
    with pytest.raises(ValueError, match="zero-norm final score vector"):
        transformer._sjoblom_method(X, y)


def test_fearn_raises_for_zero_norm_weight_vector(monkeypatch):
    """Raise when Fearn's SVD returns a zero-norm weight vector."""
    # Arrange
    X, y, _, _ = _make_osc_dataset()
    transformer = OrthogonalSignalCorrection(method="fearn", n_components=1)

    def fake_svd(z, full_matrices=False):
        u = np.zeros((z.shape[0], min(z.shape)))
        s = np.zeros(min(z.shape))
        vt = np.zeros((min(z.shape), z.shape[1]))
        return u, s, vt

    monkeypatch.setattr(osc_module, "svd", fake_svd)

    # Act / Assert
    with pytest.raises(ValueError, match="zero-norm weight vector"):
        transformer._fearn_method(X, y)


def test_fearn_raises_for_zero_norm_score_vector():
    """Raise when Fearn produces zero scores from degenerate input data."""
    # Arrange
    X = np.zeros((6, 4), dtype=float)
    y = np.linspace(0.0, 1.0, 6)
    transformer = OrthogonalSignalCorrection(method="fearn", n_components=1)

    # Act / Assert
    with pytest.raises(ValueError, match="zero-norm score vector"):
        transformer._fearn_method(X, y)


def test_raises_error_for_invalid_method():
    """Raise when an invalid method name is provided."""
    # Arrange
    X, y, _, _ = _make_osc_dataset()
    transformer = OrthogonalSignalCorrection(method="invalid_method", n_components=1)

    # Act / Assert
    with pytest.raises(
        ValueError, match="The 'method' parameter of OrthogonalSignalCorrection"
    ):
        transformer.fit(X, y)


@pytest.mark.parametrize("method", ["wold", "sjoblom", "fearn"])
def test_osc_has_correct_variance_ratios(method):
    """Test that OrthogonalSignalCorrection has correct variance ratios."""
    # Arrange
    X, y, _, _ = _make_osc_dataset()
    transformer = OrthogonalSignalCorrection(method=method, n_components=1)
    transformer.fit(X, y)

    # Act
    _ = transformer.transform(X, y)

    # Assert
    assert 0.0 <= transformer.retained_variance_ratio_ <= 1.0
    assert 0.0 <= transformer.removed_variance_ratio_ <= 1.0
    assert np.isclose(
        transformer.retained_variance_ratio_ + transformer.removed_variance_ratio_, 1.0
    )
