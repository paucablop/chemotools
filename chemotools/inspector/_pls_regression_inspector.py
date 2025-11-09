"""PLS Regression Inspector for model diagnostics and visualization."""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING, Union
import numpy as np
from sklearn.cross_decomposition._pls import _PLS
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

if TYPE_CHECKING:
    import matplotlib.figure

from chemotools.outliers import HotellingT2, QResiduals, Leverage, StudentizedResiduals

from ._base import _BaseInspector
from ._utils import (
    normalize_datasets,
    normalize_components,
    get_xlabel_for_features,
)
from ._plot_core import (
    create_variance_plot,
    create_loadings_plot,
    create_scores_plot_single_dataset,
    create_scores_plot_multi_dataset,
)
from ._plot_diagnostics import create_model_distances_plot
from ._plot_spectra import (
    create_spectra_plots_single_dataset,
    create_spectra_plots_multi_dataset,
)
from ._plot_regression import (
    create_predicted_vs_actual_plot,
    create_y_residual_plot,
    create_qq_plot,
    create_residual_distribution_plot,
    create_regression_distances_plot,
)

SummaryStep = Dict[str, Union[int, str]]
SummaryValue = Union[
    str, int, float, Dict[str, Any], List[SummaryStep], np.ndarray, None
]


class PLSRegressionInspector(_BaseInspector):
    """Inspector for PLS Regression model diagnostics and visualization.

    This class provides a unified interface for inspecting PLS regression models by
    creating multiple independent diagnostic plots. Instead of complex dashboards with
    many subplots, each method produces several separate figure windows that are easier
    to customize, save, and interact with individually.

    The inspector provides convenience methods that create multiple independent plots:
    - inspect(): Creates all diagnostic plots (scores, loadings, explained variance,
        regression diagnostics, and distance plots)
    - inspect_spectra(): Creates raw and preprocessed spectra plots (if preprocessing exists)

    Parameters
    ----------
    model : _PLS or Pipeline
        Fitted PLS model or pipeline ending with PLS
    X_train : array-like of shape (n_samples, n_features)
        Training data
    y_train : array-like of shape (n_samples,)
        Training targets (required for supervised PLS)
    X_test : array-like of shape (n_samples, n_features), optional
        Test data
    y_test : array-like of shape (n_samples,), optional
        Test targets
    X_val : array-like of shape (n_samples, n_features), optional
        Validation data
    y_val : array-like of shape (n_samples,), optional
        Validation targets
    wavenumbers : array-like of shape (n_features,), optional
        Feature names (e.g., wavenumbers for spectroscopy)
        If None, uses feature indices
    confidence : float, default=0.95
        Confidence level for outlier detection limits (Hotelling's T², Q residuals,
        leverage, and studentized residuals). Must be between 0 and 1.

    Attributes
    ----------
    model : _PLS or Pipeline
        The original model passed to the inspector
    estimator : _PLS
        The PLS estimator
    transformer : Pipeline or None
        Preprocessing pipeline before PLS (if model was a Pipeline)
    nr_components : int
        Number of latent variables
    nr_features : int
        Number of features in original data
    nr_samples : dict
        Number of samples in each dataset
    wavenumbers : ndarray
        Feature names/indices
    confidence : float
        Confidence level for outlier detection
    RMSE_train : float
        Root mean squared error on training data
    RMSE_test : float or None
        Root mean squared error on test data (if available)
    RMSE_val : float or None
        Root mean squared error on validation data (if available)
    R2_train : float
        R² score on training data
    R2_test : float or None
        R² score on test data (if available)
    R2_val : float or None
        R² score on validation data (if available)
    hotelling_t2_limit : float
        Critical value for Hotelling's T² statistic (computed on training data)
    q_residuals_limit : float
        Critical value for Q residuals statistic (computed on training data)

    Examples
    --------
    >>> from sklearn.cross_decomposition import PLSRegression
    >>> from sklearn.pipeline import make_pipeline
    >>> from sklearn.preprocessing import StandardScaler
    >>> from chemotools.datasets import load_fermentation_train
    >>> from chemotools.inspector import PLSRegressionInspector
    >>>
    >>> # Load data
    >>> X, y = load_fermentation_train()
    >>>
    >>> # Create and fit pipeline
    >>> pipeline = make_pipeline(
    ...     StandardScaler(),
    ...     PLSRegression(n_components=5)
    ... )
    >>> pipeline.fit(X, y)
    >>>
    >>> # Create inspector
    >>> inspector = PLSRegressionInspector(pipeline, X, y, wavenumbers=X.columns)
    >>>
    >>> # Print summary
    >>> inspector.summary()
    >>>
    >>> # Create all diagnostic plots
    >>> inspector.inspect()  # Creates scores, loadings, variance, regression plots
    >>>
    >>> # Compare preprocessing
    >>> inspector.inspect_spectra()
    >>>
    >>> # Access underlying data for custom analysis
    >>> x_scores = inspector.get_x_scores('train')
    >>> y_scores = inspector.get_y_scores('train')
    >>> x_loadings = inspector.get_x_loadings([0, 1, 2])
    >>> coeffs = inspector.get_regression_coefficients()
    """

    def __init__(
        self,
        model: Union[_PLS, Pipeline],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        wavenumbers: Optional[Sequence] = None,
        confidence: float = 0.95,
    ):
        super().__init__(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            X_val=X_val,
            y_val=y_val,
            supervised=True,
            feature_names=wavenumbers,
        )

        if not 0 < confidence < 1:
            raise ValueError(f"confidence must be between 0 and 1, got {confidence}")
        self._confidence = confidence

        if self.feature_names is not None:
            self._wavenumbers = np.array(self.feature_names, copy=True)
        else:
            self._wavenumbers = np.arange(self.n_features_in_)

        self._x_scores_cache: Dict[str, np.ndarray] = {}
        self._y_scores_cache: Dict[str, np.ndarray] = {}
        self._predictions_cache: Dict[str, np.ndarray] = {}
        self._hotelling_t2_limit: Optional[float] = None
        self._q_residuals_limit: Optional[float] = None
        self._leverage_detector: Optional[Leverage] = None
        self._studentized_detector: Optional[StudentizedResiduals] = None

        self._RMSE_train = self._calculate_rmse("train")
        self._RMSE_test = self._calculate_rmse("test") if X_test is not None else None
        self._RMSE_val = self._calculate_rmse("val") if X_val is not None else None

        self._R2_train = self._calculate_r2("train")
        self._R2_test = self._calculate_r2("test") if X_test is not None else None
        self._R2_val = self._calculate_r2("val") if X_val is not None else None

    # ==================================================================================
    # Properties
    # ==================================================================================

    @property
    def model(self) -> Union[_PLS, Pipeline]:
        """Return the original model (PLS or Pipeline)."""
        return super().model

    @property
    def estimator(self) -> _PLS:
        """Return the PLS estimator."""
        return self.estimator_

    @property
    def transformer(self) -> Optional[Pipeline]:
        """Return the preprocessing pipeline (if available)."""
        return super().transformer

    @property
    def nr_components(self) -> int:
        """Return the number of latent variables."""
        return self.n_components_

    @property
    def nr_features(self) -> int:
        """Return the number of features in original data."""
        return self.n_features_in_

    @property
    def nr_samples(self) -> Dict[str, int]:
        """Return the number of samples in each dataset."""
        return {name: dataset.X.shape[0] for name, dataset in self.datasets_.items()}

    @property
    def wavenumbers(self) -> np.ndarray:
        """Return the feature names/indices."""
        return self._wavenumbers

    @property
    def confidence(self) -> float:
        """Return the confidence level for outlier detection."""
        return self._confidence

    @property
    def RMSE_train(self) -> float:
        """Return RMSE on training data."""
        return self._RMSE_train

    @property
    def RMSE_test(self) -> Optional[float]:
        """Return RMSE on test data (if available)."""
        return self._RMSE_test

    @property
    def RMSE_val(self) -> Optional[float]:
        """Return RMSE on validation data (if available)."""
        return self._RMSE_val

    @property
    def R2_train(self) -> float:
        """Return R² score on training data."""
        return self._R2_train

    @property
    def R2_test(self) -> Optional[float]:
        """Return R² score on test data (if available)."""
        return self._R2_test

    @property
    def R2_val(self) -> Optional[float]:
        """Return R² score on validation data (if available)."""
        return self._R2_val

    @property
    def hotelling_t2_limit(self) -> float:
        """Return the Hotelling's T² critical value at the specified confidence level."""
        if self._hotelling_t2_limit is None:
            hotelling = HotellingT2(self.model, confidence=self._confidence)
            X_train, _ = self._get_raw_data("train")
            hotelling.fit(X_train)
            self._hotelling_t2_limit = hotelling.critical_value_
        return self._hotelling_t2_limit

    @property
    def q_residuals_limit(self) -> float:
        """Return the Q residuals critical value at the specified confidence level."""
        if self._q_residuals_limit is None:
            q_detector = QResiduals(self.model, confidence=self._confidence)
            X_train, _ = self._get_raw_data("train")
            q_detector.fit(X_train)
            self._q_residuals_limit = q_detector.critical_value_
        return self._q_residuals_limit

    @property
    def leverage_detector(self) -> Leverage:
        """Return the fitted Leverage detector.

        The detector is fitted on training data and cached for reuse.
        """
        if self._leverage_detector is None:
            self._leverage_detector = Leverage(self.model, confidence=self._confidence)
            X_train, y_train = self._get_raw_data("train")
            self._leverage_detector.fit(X_train, y_train)
        return self._leverage_detector

    @property
    def studentized_detector(self) -> StudentizedResiduals:
        """Return the fitted StudentizedResiduals detector.

        The detector is fitted on training data and cached for reuse.
        """
        if self._studentized_detector is None:
            self._studentized_detector = StudentizedResiduals(
                self.model, confidence=self._confidence
            )
            X_train, y_train = self._get_raw_data("train")
            self._studentized_detector.fit(X_train, y_train)
        return self._studentized_detector

    # ==================================================================================
    # Private Methods
    # ==================================================================================

    def _get_raw_data(self, dataset: str) -> Tuple[np.ndarray, np.ndarray]:
        """Get raw X and y data for specified dataset."""
        X, y = super()._get_raw_data(dataset)
        if y is None:
            raise ValueError(f"Target values not available for dataset '{dataset}'.")
        return X, y

    def _get_preprocessed_data(self, dataset: str) -> np.ndarray:
        """Get preprocessed X data for specified dataset."""
        return super()._get_preprocessed_data(dataset)

    def _get_predictions(self, dataset: str) -> np.ndarray:
        """Get predictions for specified dataset."""
        if dataset not in self._predictions_cache:
            X, _ = self._get_raw_data(dataset)
            y_pred = self.model.predict(X)
            if y_pred.ndim > 1:
                y_pred = y_pred.ravel()
            self._predictions_cache[dataset] = y_pred
        return self._predictions_cache[dataset]

    def _calculate_rmse(self, dataset: str) -> float:
        """Calculate RMSE for specified dataset."""
        _, y_true = self._get_raw_data(dataset)
        y_pred = self._get_predictions(dataset)
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))

    def _calculate_r2(self, dataset: str) -> float:
        """Calculate R² score for specified dataset."""
        _, y_true = self._get_raw_data(dataset)
        y_pred = self._get_predictions(dataset)
        return float(r2_score(y_true, y_pred))

    def _get_preprocessed_wavenumbers(self) -> np.ndarray:
        """Get wavenumbers after feature selection."""
        return self._get_preprocessed_feature_names()

    # ==================================================================================
    # Public Methods
    # ==================================================================================

    def get_x_scores(self, dataset: str = "train") -> np.ndarray:
        """Get PLS X-scores for specified dataset.

        Parameters
        ----------
        dataset : {'train', 'test', 'val'}, default='train'
            Which dataset to get scores for

        Returns
        -------
        x_scores : ndarray of shape (n_samples, n_components)
            PLS X-scores (latent variables from X)
        """
        if dataset not in self._x_scores_cache:
            X_preprocessed = self._get_preprocessed_data(dataset)
            x_scores = self.estimator.transform(X_preprocessed)
            self._x_scores_cache[dataset] = x_scores
        return self._x_scores_cache[dataset]

    def get_y_scores(self, dataset: str = "train") -> np.ndarray:
        """Get PLS Y-scores for specified dataset.

        Parameters
        ----------
        dataset : {'train', 'test', 'val'}, default='train'
            Which dataset to get scores for

        Returns
        -------
        y_scores : ndarray of shape (n_samples, n_components)
            PLS Y-scores (latent variables from Y)
        """
        if dataset not in self._y_scores_cache:
            X_preprocessed = self._get_preprocessed_data(dataset)
            _, y = self._get_raw_data(dataset)

            # Use transform with Y to get Y-scores
            _, y_scores = self.estimator.transform(X_preprocessed, y)
            self._y_scores_cache[dataset] = y_scores
        return self._y_scores_cache[dataset]

    def get_x_loadings(
        self, components: Optional[Union[int, Sequence[int]]] = None
    ) -> np.ndarray:
        """Get PLS X-loadings.

        Parameters
        ----------
        components : int, list of int, or None, default=None
            Which components to return. If None, returns all components.

        Returns
        -------
        x_loadings : ndarray of shape (n_features, n_components_selected)
            PLS X-loadings
        """
        loadings = self.estimator.x_loadings_

        if components is not None:
            if isinstance(components, int):
                components = [components]
            loadings = loadings[:, components]

        return loadings

    def get_x_weights(
        self, components: Optional[Union[int, Sequence[int]]] = None
    ) -> np.ndarray:
        """Get PLS X-weights.

        Parameters
        ----------
        components : int, list of int, or None, default=None
            Which components to return. If None, returns all components.

        Returns
        -------
        x_weights : ndarray of shape (n_features, n_components_selected)
            PLS X-weights
        """
        weights = self.estimator.x_weights_

        if components is not None:
            if isinstance(components, int):
                components = [components]
            weights = weights[:, components]

        return weights

    def get_x_rotations(
        self, components: Optional[Union[int, Sequence[int]]] = None
    ) -> np.ndarray:
        """Get PLS X-rotations.

        Parameters
        ----------
        components : int, list of int, or None, default=None
            Which components to return. If None, returns all components.

        Returns
        -------
        x_rotations : ndarray of shape (n_features, n_components_selected)
            PLS X-rotations
        """
        rotations = self.estimator.x_rotations_

        if components is not None:
            if isinstance(components, int):
                components = [components]
            rotations = rotations[:, components]

        return rotations

    def get_regression_coefficients(self) -> np.ndarray:
        """Get PLS regression coefficients (regression vector).

        Returns
        -------
        coef : ndarray of shape (n_features,) or (n_features, n_targets)
            PLS regression coefficients
        """
        coef = self.estimator.coef_
        # sklearn PLS stores coef_ as (n_targets, n_features)
        # Transpose to get (n_features, n_targets) for consistency
        coef = coef.T
        # For univariate targets, flatten to 1D
        if coef.shape[1] == 1:
            coef = coef.ravel()
        return coef

    def get_explained_x_variance_ratio(self) -> Optional[np.ndarray]:
        """Get explained variance ratio in X-space for all components.

        Returns
        -------
        explained_x_variance_ratio : ndarray of shape (n_components,) or None
            Explained variance ratio in X-space, or None if not available
        """
        if hasattr(self.estimator, "explained_x_variance_ratio_"):
            return self.estimator.explained_x_variance_ratio_
        return None

    def get_explained_y_variance_ratio(self) -> Optional[np.ndarray]:
        """Get explained variance ratio in Y-space for all components.

        Returns
        -------
        explained_y_variance_ratio : ndarray of shape (n_components,) or None
            Explained variance ratio in Y-space, or None if not available
        """
        if hasattr(self.estimator, "explained_y_variance_ratio_"):
            return self.estimator.explained_y_variance_ratio_
        return None

    def summary(self) -> Dict[str, SummaryValue]:
        """Get a summary dictionary of the PLS regression model.

        Returns
        -------
        summary : dict
            Dictionary containing model information
        """
        summary_dict: Dict[str, SummaryValue] = {
            "model_type": type(self.estimator).__name__,
            "has_preprocessing": self.transformer is not None,
            "nr_features": self.nr_features,
            "nr_components": self.nr_components,
            "nr_samples": self.nr_samples.copy(),
            "RMSE": {
                "train": self.RMSE_train,
                "test": self.RMSE_test,
                "val": self.RMSE_val,
            },
            "R2": {
                "train": self.R2_train,
                "test": self.R2_test,
                "val": self.R2_val,
            },
        }

        # Add variance info if available
        x_var = self.get_explained_x_variance_ratio()
        y_var = self.get_explained_y_variance_ratio()

        if x_var is not None:
            summary_dict["explained_x_variance_ratio"] = x_var
            summary_dict["total_x_variance"] = x_var.sum() * 100

        if y_var is not None:
            summary_dict["explained_y_variance_ratio"] = y_var
            summary_dict["total_y_variance"] = y_var.sum() * 100

        # Add preprocessing steps if available
        if self.transformer is not None:
            preprocessing_steps = [
                {"step": i, "name": name, "type": type(transform).__name__}
                for i, (name, transform) in enumerate(self.transformer.steps, 1)
            ]
            summary_dict["preprocessing_steps"] = preprocessing_steps
        else:
            summary_dict["preprocessing_steps"] = []

        return summary_dict

    def inspect(
        self,
        dataset: Union[str, Sequence[str]] = "train",
        components_scores: Union[Tuple[int, int], Sequence[Tuple[int, int]]] = (
            (0, 1),
            (1, 2),
        ),
        loadings_components: Union[int, Sequence[int]] = [0, 1, 2],
        variance_threshold: float = 0.95,
        color_by_y: bool = True,
        annotate_by: Optional[Union[str, Dict[str, np.ndarray]]] = None,
        scores_figsize: Tuple[float, float] = (6, 6),
        loadings_figsize: Tuple[float, float] = (10, 5),
        variance_figsize: Tuple[float, float] = (10, 5),
        spectra_figsize: Tuple[float, float] = (12, 5),
        distances_figsize: Tuple[float, float] = (8, 6),
        regression_figsize: Tuple[float, float] = (8, 6),
    ) -> Dict[str, matplotlib.figure.Figure]:
        """Create multiple independent PLS diagnostic plots.

        This method creates separate figure windows for:
        - One or more scores plots (X-scores, default: LV1 vs LV2 and LV2 vs LV3)
        - Multiple loadings plots (X-loadings, X-weights, X-rotations, coefficients)
        - Explained variance plots (X and Y spaces, if available)
        - Raw and preprocessed spectra plots (if preprocessing exists)
        - Regression diagnostic plots (predicted vs actual, residuals, Q-Q, distribution)
        - Distance plots (Hotelling's T² vs Q residuals, Leverage vs Studentized residuals)

        Parameters
        ----------
        dataset : Union[str, Sequence[str]], default='train'
            Dataset(s) to inspect
        components_scores : int, tuple, or sequence, default=((0, 1), (1, 2))
            Component(s) for scores plots
        loadings_components : int or sequence of int, default=[0, 1, 2]
            Which components to show in loadings plots
        variance_threshold : float, default=0.95
            Threshold line for explained variance plot
        color_by_y : bool, default=True
            Whether to color scores by y values (if available)
        annotate_by : str or dict, optional
            Annotations for score plot points
        scores_figsize : tuple of float, default=(6, 6)
            Figure size for each scores plot
        loadings_figsize : tuple of float, default=(10, 5)
            Figure size for loadings plots
        variance_figsize : tuple of float, default=(10, 5)
            Figure size for variance plots
        spectra_figsize : tuple of float, default=(12, 5)
            Figure size for spectra plots
        distances_figsize : tuple of float, default=(8, 6)
            Figure size for distances plots
        regression_figsize : tuple of float, default=(8, 6)
            Figure size for regression plots

        Returns
        -------
        figures : dict
            Dictionary containing all created figures
        """
        figures = {}

        # Normalize inputs
        datasets = normalize_datasets(dataset)
        components_list = normalize_components(components_scores)
        use_suffix = len(datasets) > 1

        # Get xlabel
        xlabel = get_xlabel_for_features(self._wavenumbers is not None)
        preprocessed_wavenumbers = self._get_preprocessed_wavenumbers()

        # ============================================================================
        # Variance plots (X and Y spaces)
        # ============================================================================
        x_var = self.get_explained_x_variance_ratio()
        if x_var is not None:
            figures["variance_x"] = create_variance_plot(
                explained_variance_ratio=x_var,
                variance_threshold=variance_threshold,
                figsize=variance_figsize,
            )
            figures["variance_x"].axes[0].set_title(
                "Explained Variance in X-space", fontsize=12, fontweight="bold"
            )

        y_var = self.get_explained_y_variance_ratio()
        if y_var is not None:
            figures["variance_y"] = create_variance_plot(
                explained_variance_ratio=y_var,
                variance_threshold=variance_threshold,
                figsize=variance_figsize,
            )
            figures["variance_y"].axes[0].set_title(
                "Explained Variance in Y-space", fontsize=12, fontweight="bold"
            )

        # ============================================================================
        # Loadings plots
        # ============================================================================
        # X-loadings
        figures["loadings_x"] = create_loadings_plot(
            loadings=self.get_x_loadings(),
            feature_names=preprocessed_wavenumbers,
            loadings_components=loadings_components,
            xlabel=xlabel,
            figsize=loadings_figsize,
            component_label="LV",
        )
        figures["loadings_x"].axes[0].set_title(
            "X-Loadings", fontsize=12, fontweight="bold"
        )

        # X-weights
        figures["loadings_weights"] = create_loadings_plot(
            loadings=self.get_x_weights(),
            feature_names=preprocessed_wavenumbers,
            loadings_components=loadings_components,
            xlabel=xlabel,
            figsize=loadings_figsize,
            component_label="LV",
        )
        figures["loadings_weights"].axes[0].set_title(
            "X-Weights", fontsize=12, fontweight="bold"
        )

        # X-rotations
        figures["loadings_rotations"] = create_loadings_plot(
            loadings=self.get_x_rotations(),
            feature_names=preprocessed_wavenumbers,
            loadings_components=loadings_components,
            xlabel=xlabel,
            figsize=loadings_figsize,
            component_label="LV",
        )
        figures["loadings_rotations"].axes[0].set_title(
            "X-Rotations", fontsize=12, fontweight="bold"
        )

        # Regression coefficients
        coef = self.get_regression_coefficients()
        # Create a loadings-style plot for coefficients (single "component")
        figures["regression_coefficients"] = create_loadings_plot(
            loadings=coef.reshape(-1, 1),  # Shape as (n_features, 1)
            feature_names=preprocessed_wavenumbers,
            loadings_components=[0],  # Only one "component"
            xlabel=xlabel,
            figsize=loadings_figsize,
            component_label="LV",
        )
        figures["regression_coefficients"].axes[0].set_title(
            "Regression Coefficients", fontsize=12, fontweight="bold"
        )
        # Update legend to show "Coefficients" instead of "LV 1"
        ax = figures["regression_coefficients"].axes[0]
        handles, _ = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, ["Coefficients"], loc="best")

        # ============================================================================
        # Scores plots (X-scores)
        # ============================================================================
        if use_suffix:
            # Multiple datasets
            scores_datasets: Dict[str, Dict[str, Optional[np.ndarray]]] = {}
            for ds in datasets:
                _, y_values = self._get_raw_data(ds)
                scores_datasets[ds] = {
                    "scores": self.get_x_scores(ds),
                    "y": y_values,
                }

            # Get explained variance for axis labels
            explained_var = (
                x_var
                if x_var is not None
                else np.zeros(self.nr_components, dtype=float)
            )

            for i, component_spec in enumerate(components_list, start=1):
                fig = create_scores_plot_multi_dataset(
                    component_spec=component_spec,
                    datasets_data=scores_datasets,
                    explained_var=explained_var,
                    color_by_y=color_by_y,
                    annotate_by=annotate_by,
                    figsize=scores_figsize,
                    component_label="LV",
                )
                figures[f"scores_{i}"] = fig
        else:
            # Single dataset
            ds = datasets[0]
            scores = self.get_x_scores(ds)
            _, y = self._get_raw_data(ds)
            explained_var = (
                x_var
                if x_var is not None
                else np.zeros(self.nr_components, dtype=float)
            )

            for i, component_spec in enumerate(components_list, start=1):
                fig = create_scores_plot_single_dataset(
                    component_spec=component_spec,
                    scores=scores,
                    y=y,
                    explained_var=explained_var,
                    dataset_name=ds,
                    color_by_y=color_by_y,
                    annotate_by=annotate_by,
                    figsize=scores_figsize,
                    component_label="LV",
                )
                figures[f"scores_{i}"] = fig

        # ============================================================================
        # Distance plots
        # ============================================================================
        # Hotelling T² vs Q residuals
        distance_datasets: Dict[str, Dict[str, Optional[np.ndarray]]] = {}
        for ds in datasets:
            X, y = self._get_raw_data(ds)
            distance_datasets[ds] = {"X": X, "y": y}

        fig_distances = create_model_distances_plot(
            datasets_data=distance_datasets,
            model=self._model,
            confidence=self._confidence,
            color_by_y=color_by_y,
            figsize=distances_figsize,
        )
        figures["distances_hotelling_q"] = fig_distances

        # Leverage vs Studentized residuals
        if use_suffix:
            regression_datasets: Dict[str, Dict[str, np.ndarray]] = {}
            for ds in datasets:
                X, y_true = self._get_raw_data(ds)
                y_pred = self._get_predictions(ds)
                regression_datasets[ds] = {
                    "X": X,
                    "y": y_true,
                    "y_true": y_true,
                    "y_pred": y_pred,
                }

            fig_leverage = create_regression_distances_plot(
                datasets_data=regression_datasets,
                leverage_detector=self.leverage_detector,
                student_detector=self.studentized_detector,
                color_by_y=color_by_y,
                figsize=distances_figsize,
            )
            figures["distances_leverage_studentized"] = fig_leverage
        else:
            ds = datasets[0]
            X, y_true = self._get_raw_data(ds)
            y_pred = self._get_predictions(ds)

            # Create single-item dict for unified function
            regression_datasets = {
                ds: {
                    "X": X,
                    "y": y_true,
                    "y_true": y_true,
                    "y_pred": y_pred,
                }
            }

            fig_leverage = create_regression_distances_plot(
                datasets_data=regression_datasets,
                leverage_detector=self.leverage_detector,
                student_detector=self.studentized_detector,
                color_by_y=color_by_y,
                figsize=distances_figsize,
            )
            figures["distances_leverage_studentized"] = fig_leverage

        # ============================================================================
        # Regression diagnostic plots
        # ============================================================================
        # Predicted vs Actual
        if use_suffix:
            predicted_vs_actual_data: Dict[str, Dict[str, np.ndarray]] = {}
            for ds in datasets:
                _, y_true = self._get_raw_data(ds)
                y_pred = self._get_predictions(ds)
                predicted_vs_actual_data[ds] = {
                    "y_true": y_true,
                    "y_pred": y_pred,
                    "y": y_true,
                }

            figures["predicted_vs_actual"] = create_predicted_vs_actual_plot(
                datasets_data=predicted_vs_actual_data,
                color_by_y=color_by_y,
                figsize=regression_figsize,
            )
        else:
            ds = datasets[0]
            _, y_true = self._get_raw_data(ds)
            y_pred = self._get_predictions(ds)

            # Create single-item dict for unified function
            predicted_vs_actual_data = {
                ds: {
                    "y_true": y_true,
                    "y_pred": y_pred,
                    "y": y_true,
                }
            }

            figures["predicted_vs_actual"] = create_predicted_vs_actual_plot(
                datasets_data=predicted_vs_actual_data,
                color_by_y=color_by_y,
                figsize=regression_figsize,
            )

        # Residual scatter plot
        if use_suffix:
            residuals_data: Dict[str, Dict[str, np.ndarray]] = {}
            for ds in datasets:
                _, y_true = self._get_raw_data(ds)
                y_pred = self._get_predictions(ds)
                residuals_data[ds] = {
                    "y_true": y_true,
                    "y_pred": y_pred,
                    "y": y_true,
                }

            figures["residuals"] = create_y_residual_plot(
                datasets_data=residuals_data,
                color_by_y=color_by_y,
                figsize=regression_figsize,
            )
        else:
            ds = datasets[0]
            _, y_true = self._get_raw_data(ds)
            y_pred = self._get_predictions(ds)

            # Create single-item dict for unified function
            residuals_data = {
                ds: {
                    "y_true": y_true,
                    "y_pred": y_pred,
                    "y": y_true,
                }
            }

            figures["residuals"] = create_y_residual_plot(
                datasets_data=residuals_data,
                color_by_y=color_by_y,
                figsize=regression_figsize,
            )

        # Q-Q plot
        if use_suffix:
            qq_data: Dict[str, Dict[str, np.ndarray]] = {}
            for ds in datasets:
                _, y_true = self._get_raw_data(ds)
                y_pred = self._get_predictions(ds)
                qq_data[ds] = {
                    "y_true": y_true,
                    "y_pred": y_pred,
                }

            figures["qq_plot"] = create_qq_plot(
                datasets_data=qq_data,
                figsize=regression_figsize,
            )
        else:
            ds = datasets[0]
            _, y_true = self._get_raw_data(ds)
            y_pred = self._get_predictions(ds)

            # Create single-item dict for unified function
            qq_data = {
                ds: {
                    "y_true": y_true,
                    "y_pred": y_pred,
                }
            }

            figures["qq_plot"] = create_qq_plot(
                datasets_data=qq_data,
                figsize=regression_figsize,
            )

        # Residual distribution
        if use_suffix:
            residual_dist_data: Dict[str, Dict[str, np.ndarray]] = {}
            for ds in datasets:
                _, y_true = self._get_raw_data(ds)
                y_pred = self._get_predictions(ds)
                residual_dist_data[ds] = {
                    "y_true": y_true,
                    "y_pred": y_pred,
                }

            figures["residual_distribution"] = create_residual_distribution_plot(
                datasets_data=residual_dist_data,
                figsize=regression_figsize,
            )
        else:
            ds = datasets[0]
            _, y_true = self._get_raw_data(ds)
            y_pred = self._get_predictions(ds)

            # Create single-item dict for unified function
            residual_dist_data = {
                ds: {
                    "y_true": y_true,
                    "y_pred": y_pred,
                }
            }

            figures["residual_distribution"] = create_residual_distribution_plot(
                datasets_data=residual_dist_data,
                figsize=regression_figsize,
            )

        # ============================================================================
        # Spectra plots (if preprocessing exists)
        # ============================================================================
        if self.transformer is not None:
            spectra_figs = self.inspect_spectra(
                dataset=datasets if use_suffix else datasets[0],
                color_by_y=color_by_y,
                figsize=spectra_figsize,
            )
            figures.update(spectra_figs)

        return figures

    def inspect_spectra(
        self,
        dataset: Union[str, Sequence[str]] = "train",
        color_by_y: bool = True,
        xlim: Optional[Tuple[float, float]] = None,
        figsize: Tuple[float, float] = (12, 5),
    ) -> Dict[str, matplotlib.figure.Figure]:
        """Create independent plots comparing raw and preprocessed spectra.

        Only available if model is a Pipeline with preprocessing steps.

        Parameters
        ----------
        dataset : Union[str, Sequence[str]], default='train'
            Dataset(s) to visualize
        color_by_y : bool, default=True
            Whether to color by y values (single dataset only)
        xlim : tuple of float, optional
            X-axis limits for zooming
        figsize : tuple of float, default=(12, 5)
            Figure size for each plot

        Returns
        -------
        figures : dict
            Dictionary containing 'raw_spectra' and 'preprocessed_spectra'

        Raises
        ------
        ValueError
            If no preprocessing pipeline is available
        """
        if self.transformer is None:
            raise ValueError(
                "Spectra inspection requires a preprocessing pipeline. "
                "Model must be a Pipeline with preprocessing steps."
            )

        # Normalize dataset to always be a list
        datasets = normalize_datasets(dataset)
        is_multi_dataset = len(datasets) > 1

        # Determine xlabel
        xlabel = get_xlabel_for_features(self._wavenumbers is not None)

        # Get preprocessed wavenumbers
        preprocessed_wavenumbers = self._get_preprocessed_wavenumbers()

        if is_multi_dataset:
            # Multiple datasets
            raw_data = {}
            preprocessed_data = {}
            for ds in datasets:
                X_raw, _ = self._get_raw_data(ds)
                X_preprocessed = self._get_preprocessed_data(ds)
                raw_data[ds] = X_raw
                preprocessed_data[ds] = X_preprocessed

            figures = create_spectra_plots_multi_dataset(
                raw_data=raw_data,
                preprocessed_data=preprocessed_data,
                wavenumbers=self.wavenumbers,
                preprocessed_wavenumbers=preprocessed_wavenumbers,
                xlabel=xlabel,
                xlim=xlim,
                figsize=figsize,
            )
        else:
            # Single dataset
            ds = datasets[0]
            X_raw, y = self._get_raw_data(ds)
            X_preprocessed = self._get_preprocessed_data(ds)

            figures = create_spectra_plots_single_dataset(
                X_raw=X_raw,
                X_preprocessed=X_preprocessed,
                y=y,
                wavenumbers=self.wavenumbers,
                preprocessed_wavenumbers=preprocessed_wavenumbers,
                dataset_name=ds,
                color_by_y=color_by_y,
                xlabel=xlabel,
                xlim=xlim,
                figsize=figsize,
            )

        return figures
