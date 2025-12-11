"""Regression-specific utilities shared across inspector implementations."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING, Union, Sequence

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

from chemotools.inspector.helpers import _regression as _regression_plots
from .utils import normalize_datasets
from .summaries import RegressionMetrics, RegressionSummary

if TYPE_CHECKING:  # pragma: no cover
    from typing import Protocol, Literal
    from matplotlib.figure import Figure

    from chemotools.inspector.core.base import ModelTypes

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

        def _get_predictions(self, dataset: str) -> np.ndarray:  # pragma: no cover
            ...

        @property
        def estimator(self) -> Any:  # pragma: no cover
            ...

        def _get_preprocessed_data(
            self, dataset: str
        ) -> np.ndarray:  # pragma: no cover
            ...


class RegressionMixin:
    """Provide regression diagnostics independent of latent-space plotting."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._predictions_cache: Dict[str, np.ndarray] = {}
        self._rmse_cache: Dict[str, float] = {}
        self._r2_cache: Dict[str, float] = {}
        self._bias_cache: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Private methods
    # ------------------------------------------------------------------
    def _regression_inspector(self) -> "_RegressionInspectorProto":
        return self  # type: ignore[return-value]

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

    def _calculate_bias(self, dataset: str) -> float:
        _, y_true = self._get_regression_raw_data(dataset)
        y_pred = self._get_predictions(dataset)

        # Ensure shapes match for subtraction to avoid broadcasting errors
        y_true = np.asarray(y_true)
        if y_true.shape != y_pred.shape:
            if y_true.ndim == 2 and y_true.shape[1] == 1 and y_pred.ndim == 1:
                y_true = y_true.ravel()
            elif y_pred.ndim == 2 and y_pred.shape[1] == 1 and y_true.ndim == 1:
                y_pred = y_pred.ravel()

        # Bias = Mean(y_pred - y_true)
        bias = float(np.mean(y_pred - y_true))
        self._bias_cache[dataset] = bias
        return bias

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

    def _get_datasets_data(
        self, dataset: Union[str, Sequence[str]]
    ) -> Dict[str, Dict[str, np.ndarray]]:
        datasets = normalize_datasets(dataset)
        datasets_data = {}
        for name in datasets:
            if not self._regression_dataset_exists(name):
                continue
            X, y_true = self._get_regression_raw_data(name)
            y_pred = self._get_predictions(name)
            datasets_data[name] = {"y_true": y_true, "y_pred": y_pred, "y": y_true}
        return datasets_data

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def regression_rmse(self, dataset: str) -> float:
        """Return RMSE for the specified dataset.

        Parameters
        ----------
        dataset : str
            Dataset name ('train', 'test', or 'val').

        Returns
        -------
        float
            Root mean squared error.

        Examples
        --------
        >>> inspector.regression_rmse("train")
        0.523
        """
        if dataset not in self._rmse_cache:
            self._calculate_rmse(dataset)
        return self._rmse_cache[dataset]

    def regression_r2(self, dataset: str) -> float:
        """Return R² score for the specified dataset.

        Parameters
        ----------
        dataset : str
            Dataset name ('train', 'test', or 'val').

        Returns
        -------
        float
            Coefficient of determination.

        Examples
        --------
        >>> inspector.regression_r2("train")
        0.95
        """
        if dataset not in self._r2_cache:
            self._calculate_r2(dataset)
        return self._r2_cache[dataset]

    def regression_bias(self, dataset: str) -> float:
        """Return prediction bias (mean error) for the specified dataset.

        Parameters
        ----------
        dataset : str
            Dataset name ('train', 'test', or 'val').

        Returns
        -------
        float
            Mean prediction error (y_pred - y_true).

        Examples
        --------
        >>> inspector.regression_bias("test")
        -0.012
        """
        if dataset not in self._bias_cache:
            self._calculate_bias(dataset)
        return self._bias_cache[dataset]

    def regression_summary(self) -> RegressionSummary:
        """Return a summary of prediction metrics (RMSE, R2) for all available datasets.

        Returns
        -------
        summary : RegressionSummary
            Object containing RegressionMetrics for 'train', 'test', and 'val' datasets.
        """
        metrics = {}
        inspector = self._regression_inspector()
        datasets = getattr(inspector, "datasets_", {})

        for name in ["train", "test", "val"]:
            if name not in datasets:
                continue

            # Only calculate if target values are available
            _, y = inspector._get_raw_data(name)
            if y is not None:
                metrics[name] = RegressionMetrics(
                    rmse=self.regression_rmse(name),
                    r2=self.regression_r2(name),
                    bias=self.regression_bias(name),
                )

        return RegressionSummary(
            train=metrics["train"],
            test=metrics.get("test"),
            val=metrics.get("val"),
        )

    def create_predicted_vs_actual_plot(
        self,
        dataset: Union[str, Sequence[str]] = "train",
        color_by: Optional[Union[str, Dict[str, np.ndarray]]] = "y",
        figsize: Tuple[float, float] = (6, 6),
        annotate_by: Optional[Union[str, Dict[str, np.ndarray]]] = None,
        color_mode: Optional[Literal["continuous", "categorical"]] = None,
    ) -> "Figure":
        """Create predicted vs actual plot.

        Parameters
        ----------
        dataset : str or list of str, default='train'
            Dataset(s) to plot.
        color_by : str or dict, default='y'
            How to color points ('y', 'sample_index', or custom dict).
        figsize : tuple, default=(6, 6)
            Figure dimensions (width, height) in inches.
        annotate_by : str or dict, optional
            Labels for annotating individual points.
        color_mode : {'continuous', 'categorical'}, optional
            Coloring mode for points.

        Returns
        -------
        Figure
            Matplotlib figure with predicted vs actual scatter plot.

        Examples
        --------
        >>> fig = inspector.create_predicted_vs_actual_plot()
        >>> fig = inspector.create_predicted_vs_actual_plot(["train", "test"])
        """
        datasets_data = self._get_datasets_data(dataset)
        return _regression_plots.create_predicted_vs_actual_plot(
            datasets_data=datasets_data,
            color_by=color_by,
            figsize=figsize,
            annotate_by=annotate_by,
            color_mode=color_mode,
        )

    def create_residuals_plot(
        self,
        dataset: Union[str, Sequence[str]] = "train",
        color_by: Optional[Union[str, Dict[str, np.ndarray]]] = "y",
        figsize: Tuple[float, float] = (8, 4),
        annotate_by: Optional[Union[str, Dict[str, np.ndarray]]] = None,
        color_mode: Optional[Literal["continuous", "categorical"]] = None,
    ) -> "Figure":
        """Create residuals plot.

        Parameters
        ----------
        dataset : str or list of str, default='train'
            Dataset(s) to plot.
        color_by : str or dict, default='y'
            How to color points ('y', 'sample_index', or custom dict).
        figsize : tuple, default=(8, 4)
            Figure dimensions (width, height) in inches.
        annotate_by : str or dict, optional
            Labels for annotating individual points.
        color_mode : {'continuous', 'categorical'}, optional
            Coloring mode for points.

        Returns
        -------
        Figure
            Matplotlib figure with residuals vs predicted plot.

        Examples
        --------
        >>> fig = inspector.create_residuals_plot()
        >>> fig = inspector.create_residuals_plot("test", color_by="sample_index")
        """
        datasets_data = self._get_datasets_data(dataset)
        return _regression_plots.create_y_residual_plot(
            datasets_data=datasets_data,
            color_by=color_by,
            figsize=figsize,
            annotate_by=annotate_by,
            color_mode=color_mode,
        )

    def create_qq_plot(
        self,
        dataset: Union[str, Sequence[str]] = "train",
        figsize: Tuple[float, float] = (6, 6),
    ) -> "Figure":
        """Create Q-Q plot for residual normality assessment.

        Parameters
        ----------
        dataset : str or list of str, default='train'
            Dataset(s) to plot.
        figsize : tuple, default=(6, 6)
            Figure dimensions (width, height) in inches.

        Returns
        -------
        Figure
            Matplotlib figure with Q-Q plot.

        Examples
        --------
        >>> fig = inspector.create_qq_plot()
        """
        datasets_data = self._get_datasets_data(dataset)
        inspector = self._regression_inspector()
        return _regression_plots.create_qq_plot(
            datasets_data=datasets_data,
            figsize=figsize,
            confidence=inspector.confidence,
        )

    def create_residual_distribution_plot(
        self,
        dataset: Union[str, Sequence[str]] = "train",
        figsize: Tuple[float, float] = (8, 4),
    ) -> "Figure":
        """Create residual distribution histogram.

        Parameters
        ----------
        dataset : str or list of str, default='train'
            Dataset(s) to plot.
        figsize : tuple, default=(8, 4)
            Figure dimensions (width, height) in inches.

        Returns
        -------
        Figure
            Matplotlib figure with residual histogram.

        Examples
        --------
        >>> fig = inspector.create_residual_distribution_plot()
        """
        datasets_data = self._get_datasets_data(dataset)
        return _regression_plots.create_residual_distribution_plot(
            datasets_data=datasets_data,
            figsize=figsize,
        )
