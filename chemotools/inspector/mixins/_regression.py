"""Regression-specific utilities shared across inspector implementations."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

from chemotools.outliers import Leverage, StudentizedResiduals

if TYPE_CHECKING:  # pragma: no cover
    from typing import Protocol

    from chemotools.inspector._base import ModelTypes

    class _RegressionInspectorProto(Protocol):
        datasets_: Dict[str, Any]

        @property
        def model(self) -> ModelTypes:  # pragma: no cover
            ...

        @property
        def confidence(self) -> float:  # pragma: no cover
            ...

        def _get_raw_data(
            self, dataset: str
        ) -> Tuple[np.ndarray, Optional[np.ndarray]]:  # pragma: no cover
            ...


class RegressionMixin:
    """Provide regression diagnostics independent of latent-space plotting."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._predictions_cache: Dict[str, np.ndarray] = {}
        self._rmse_cache: Dict[str, float] = {}
        self._r2_cache: Dict[str, float] = {}
        self._leverage_detector: Optional[Leverage] = None
        self._studentized_detector: Optional[StudentizedResiduals] = None

    def _regression_inspector(self) -> "_RegressionInspectorProto":
        return self  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _regression_dataset_exists(self, dataset: str) -> bool:
        inspector = self._regression_inspector()
        datasets = getattr(inspector, "datasets_", {})
        return dataset in datasets

    def _get_regression_raw_data(self, dataset: str) -> Tuple[np.ndarray, np.ndarray]:
        inspector = self._regression_inspector()
        X, y = inspector._get_raw_data(dataset)
        if y is None:
            raise ValueError(f"Target values not available for dataset '{dataset}'.")
        return X, y

    def _get_predictions(self, dataset: str) -> np.ndarray:
        if dataset not in self._predictions_cache:
            X, _ = self._get_regression_raw_data(dataset)
            inspector = self._regression_inspector()
            y_pred = inspector.model.predict(X)
            y_pred = np.asarray(y_pred)
            if y_pred.ndim == 2 and y_pred.shape[1] == 1:
                y_pred = y_pred.ravel()
            self._predictions_cache[dataset] = y_pred
        return self._predictions_cache[dataset]

    def _calculate_rmse(self, dataset: str) -> float:
        _, y_true = self._get_regression_raw_data(dataset)
        y_pred = self._get_predictions(dataset)
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        self._rmse_cache[dataset] = rmse
        return rmse

    def _calculate_r2(self, dataset: str) -> float:
        _, y_true = self._get_regression_raw_data(dataset)
        y_pred = self._get_predictions(dataset)
        score = float(r2_score(y_true, y_pred))
        self._r2_cache[dataset] = score
        return score

    def _optional_rmse(self, dataset: str) -> Optional[float]:
        if not self._regression_dataset_exists(dataset):
            return None
        if dataset not in self._rmse_cache:
            self._calculate_rmse(dataset)
        return self._rmse_cache[dataset]

    def _optional_r2(self, dataset: str) -> Optional[float]:
        if not self._regression_dataset_exists(dataset):
            return None
        if dataset not in self._r2_cache:
            self._calculate_r2(dataset)
        return self._r2_cache[dataset]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def regression_rmse(self, dataset: str) -> float:
        """Return RMSE for the specified dataset, computing it on demand."""
        if dataset not in self._rmse_cache:
            self._calculate_rmse(dataset)
        return self._rmse_cache[dataset]

    def regression_r2(self, dataset: str) -> float:
        """Return R² score for the specified dataset, computing it on demand."""
        if dataset not in self._r2_cache:
            self._calculate_r2(dataset)
        return self._r2_cache[dataset]

    @property
    def RMSE_train(self) -> float:
        """Return RMSE on training data."""
        return self.regression_rmse("train")

    @property
    def RMSE_test(self) -> Optional[float]:
        """Return RMSE on test data, or ``None`` when unavailable."""
        return self._optional_rmse("test")

    @property
    def RMSE_val(self) -> Optional[float]:
        """Return RMSE on validation data, or ``None`` when unavailable."""
        return self._optional_rmse("val")

    @property
    def R2_train(self) -> float:
        """Return R² score on training data."""
        return self.regression_r2("train")

    @property
    def R2_test(self) -> Optional[float]:
        """Return R² score on test data, or ``None`` when unavailable."""
        return self._optional_r2("test")

    @property
    def R2_val(self) -> Optional[float]:
        """Return R² score on validation data, or ``None`` when unavailable."""
        return self._optional_r2("val")

    @property
    def leverage_detector(self) -> Leverage:
        """Return a fitted leverage detector cached for reuse."""
        if self._leverage_detector is None:
            inspector = self._regression_inspector()
            detector = Leverage(inspector.model, confidence=inspector.confidence)
            X_train, y_train = self._get_regression_raw_data("train")
            detector.fit(X_train, y_train)
            self._leverage_detector = detector
        return self._leverage_detector

    @property
    def studentized_detector(self) -> StudentizedResiduals:
        """Return a fitted studentized residuals detector cached for reuse."""
        if self._studentized_detector is None:
            inspector = self._regression_inspector()
            detector = StudentizedResiduals(
                inspector.model, confidence=inspector.confidence
            )
            X_train, y_train = self._get_regression_raw_data("train")
            detector.fit(X_train, y_train)
            self._studentized_detector = detector
        return self._studentized_detector
