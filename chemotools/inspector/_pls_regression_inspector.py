"""PLS Regression Inspector for model diagnostics and visualization."""

from __future__ import annotations
from typing import (
    Dict,
    Optional,
    Sequence,
    Tuple,
    Union,
    Any,
    TYPE_CHECKING,
    Literal,
    List,
)
import numpy as np
from sklearn.cross_decomposition._pls import _PLS
from sklearn.pipeline import Pipeline

if TYPE_CHECKING:
    import matplotlib.figure

from chemotools.outliers import QResiduals

from ._base import _BaseInspector, InspectorPlotConfig
from .mixins import LatentVariableMixin, RegressionMixin, SpectraMixin
from ._utils import (
    normalize_datasets,
    get_xlabel_for_features,
    get_default_scores_components,
    get_default_loadings_components,
    select_components,
)
from .helpers import _latent as _latent_plots
from .helpers._regression import (
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


class PLSRegressionInspector(
    SpectraMixin, RegressionMixin, LatentVariableMixin, _BaseInspector
):
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
    x_axis : array-like of shape (n_features,), optional
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
    x_axis : ndarray
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
    >>> inspector = PLSRegressionInspector(pipeline, X, y, x_axis=X.columns)
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

    component_label = "LV"

    def __init__(
        self,
        model: Union[_PLS, Pipeline],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        x_axis: Optional[Sequence] = None,
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
            feature_names=x_axis,
            confidence=confidence,
        )

        self._x_scores_cache: Dict[str, np.ndarray] = {}
        self._y_scores_cache: Dict[str, np.ndarray] = {}

    # ==================================================================================
    # Public Methods
    # ==================================================================================

    # ------------------------------------------------------------------
    # LatentVariableMixin hooks
    # ------------------------------------------------------------------
    def get_latent_scores(self, dataset: str) -> np.ndarray:
        """Hook for LatentVariableMixin - returns X-scores."""
        return self.get_x_scores(dataset)

    def get_latent_explained_variance(self) -> Optional[np.ndarray]:
        """Hook for LatentVariableMixin - returns explained X variance ratio."""
        return self.get_explained_x_variance_ratio()

    def get_latent_loadings(self) -> np.ndarray:
        """Hook for LatentVariableMixin - returns X-loadings."""
        return self.get_x_loadings()

    # ------------------------------------------------------------------
    # Scores methods
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Loadings and weights methods
    # ------------------------------------------------------------------
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
        return select_components(self.estimator.x_loadings_, components)

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
        return select_components(self.estimator.x_weights_, components)

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
        return select_components(self.estimator.x_rotations_, components)

    # ------------------------------------------------------------------
    # Regression coefficients
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Variance methods
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Summary method
    # ------------------------------------------------------------------
    def summary(self) -> Dict[str, SummaryValue]:
        """Get a summary dictionary of the PLS regression model.

        Returns
        -------
        summary : dict
            Dictionary containing model information
        """
        # Start with common summary fields
        summary_dict: Dict[str, SummaryValue] = self._base_summary()

        # Add PLS regression-specific metrics
        pred_summary = self.prediction_summary()
        rmse_dict = {ds: metrics["RMSE"] for ds, metrics in pred_summary.items()}
        r2_dict = {ds: metrics["R2"] for ds, metrics in pred_summary.items()}

        summary_dict["RMSE"] = rmse_dict
        summary_dict["R2"] = r2_dict

        # Add variance info if available
        x_var = self.get_explained_x_variance_ratio()
        y_var = self.get_explained_y_variance_ratio()

        if x_var is not None:
            summary_dict["explained_x_variance_ratio"] = x_var
            summary_dict["total_x_variance"] = x_var.sum() * 100

        if y_var is not None:
            summary_dict["explained_y_variance_ratio"] = y_var
            summary_dict["total_y_variance"] = y_var.sum() * 100

        return summary_dict

    # ------------------------------------------------------------------
    # Main inspection method
    # ------------------------------------------------------------------
    def inspect(
        self,
        dataset: Union[str, Sequence[str]] = "train",
        components_scores: Optional[
            Union[int, Tuple[int, int], Sequence[Union[int, Tuple[int, int]]]]
        ] = None,
        loadings_components: Optional[Union[int, Sequence[int]]] = None,
        variance_threshold: float = 0.95,
        color_by_y: bool = True,
        annotate_by: Optional[Union[str, Dict[str, np.ndarray]]] = None,
        plot_config: Optional[InspectorPlotConfig] = None,
        color_mode: Literal["continuous", "categorical"] = "continuous",
        **kwargs,
    ) -> Dict[str, matplotlib.figure.Figure]:
        """Create multiple independent PLS diagnostic plots.

        This method creates separate figure windows for:
        - One or more scores plots (X-scores, default depends on model components)
        - Multiple loadings plots (X-loadings, X-weights, X-rotations, coefficients)
        - Explained variance plots (X and Y spaces, if available)
        - Raw and preprocessed spectra plots (if preprocessing exists)
        - Regression diagnostic plots (predicted vs actual, residuals, Q-Q, distribution)
        - Distance plots (Hotelling's T² vs Q residuals, Q residuals vs Y residuals, Leverage vs Studentized residuals)

        Parameters
        ----------
        dataset : Union[str, Sequence[str]], default='train'
            Dataset(s) to inspect
        components_scores : int, tuple, sequence, or None, optional
            Component(s) for scores plots. If None (default), automatically selects based
            on number of components:
            - 1 component: 0 (LV1 vs sample index/y)
            - 2 components: (0, 1) (LV1 vs LV2)
            - 3+ components: ((0, 1), (1, 2)) (two 2D plots)
            Can also be manually specified.
        loadings_components : int, sequence of int, or None, optional
            Which components to show in loadings plots. If None (default), automatically
            selects all available components:
            - 1 component: 0
            - 2+ components: [0, 1, ..., n_components-1] (all components)
        variance_threshold : float, default=0.95
            Threshold line for explained variance plot
        color_by_y : bool, default=True
            Whether to color scores by y values (if available)
        annotate_by : str or dict, optional
            Annotations for score plot points
        plot_config : InspectorPlotConfig, optional
            Configuration object for plot sizes and styles. If None, defaults are used.
        color_mode : Literal["continuous", "categorical"], default="continuous"
            Mode for coloring points.
        **kwargs
            Optional keyword arguments to override specific fields in plot_config
            (e.g., scores_figsize=(8, 8)).

        Returns
        -------
                figures : dict
                        Dictionary containing all created figures. Keys include:
                        - 'scores_1', 'scores_2', ...: X-scores plots (combined multi-dataset when multiple datasets provided)
                        - 'x_vs_y_scores_1', 'x_vs_y_scores_2', ...: X-scores vs Y-scores plots (training set only)
                        - 'loadings_x', 'loadings_weights', 'loadings_rotations': X-related loadings plots
                        - 'regression_coefficients': Regression coefficient traces (one per target when multi-output)
                        - 'variance_x', 'variance_y': Explained variance plots (when available)
                        - 'distances_hotelling_q', 'distances_q_y_residuals', 'distances_leverage_studentized': Distance diagnostics
                        - 'predicted_vs_actual', 'residuals', 'qq_plot', 'residual_distribution': Regression diagnostics
                        - 'raw_spectra', 'preprocessed_spectra': Spectra plots (when preprocessing exists)
        """
        # Generate smart defaults based on number of components
        if components_scores is None:
            components_scores = get_default_scores_components(self.nr_components)
        if loadings_components is None:
            loadings_components = get_default_loadings_components(self.nr_components)

        # Handle configuration
        config = plot_config or InspectorPlotConfig()
        # Allow kwargs to override config for convenience
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)

        figures = {}

        datasets = normalize_datasets(dataset)
        use_suffix = len(datasets) > 1

        xlabel = get_xlabel_for_features(self.feature_names is not None)
        preprocessed_x_axis = self._get_preprocessed_x_axis()

        # ------------------------------------------------------------------
        # Variance plots (X and Y space)
        # ------------------------------------------------------------------
        x_var = self.get_explained_x_variance_ratio()
        if x_var is not None:
            variance_x_fig = self.create_latent_variance_figure(
                variance_threshold=variance_threshold,
                figsize=config.variance_figsize,
            )
            if variance_x_fig is not None:
                variance_x_fig.axes[0].set_title(
                    "Explained Variance in X-space",
                    fontsize=12,
                    fontweight="bold",
                )
                figures["variance_x"] = variance_x_fig

        # Y-space variance
        y_var = self.get_explained_y_variance_ratio()
        if y_var is not None:
            variance_y_fig = _latent_plots.create_variance_plot(
                explained_variance_ratio=y_var,
                variance_threshold=variance_threshold,
                figsize=config.variance_figsize,
            )
            variance_y_fig.axes[0].set_title(
                "Explained Variance in Y-space", fontsize=12, fontweight="bold"
            )
            figures["variance_y"] = variance_y_fig

        # ------------------------------------------------------------------
        # Loadings plots (X-loadings, X-weights, X-rotations, coefficients)
        # ------------------------------------------------------------------
        loadings_x_fig = self.create_latent_loadings_figure(
            loadings_components=loadings_components,
            xlabel=xlabel,
            figsize=config.loadings_figsize,
        )
        loadings_x_fig.axes[0].set_title("X-Loadings", fontsize=12, fontweight="bold")
        figures["loadings_x"] = loadings_x_fig

        figures["loadings_weights"] = _latent_plots.create_loadings_plot(
            loadings=self.get_x_weights(),
            feature_names=preprocessed_x_axis,
            loadings_components=loadings_components,
            xlabel=xlabel,
            figsize=config.loadings_figsize,
            component_label=self.component_label,
        )
        figures["loadings_weights"].axes[0].set_title(
            "X-Weights", fontsize=12, fontweight="bold"
        )

        figures["loadings_rotations"] = _latent_plots.create_loadings_plot(
            loadings=self.get_x_rotations(),
            feature_names=preprocessed_x_axis,
            loadings_components=loadings_components,
            xlabel=xlabel,
            figsize=config.loadings_figsize,
            component_label=self.component_label,
        )
        figures["loadings_rotations"].axes[0].set_title(
            "X-Rotations", fontsize=12, fontweight="bold"
        )

        coef = self.get_regression_coefficients()
        if coef.ndim == 1:
            coef_matrix = coef.reshape(-1, 1)
            coef_components = [0]
            component_label = "Coeff"
        else:
            coef_matrix = coef
            coef_components = list(range(coef_matrix.shape[1]))
            component_label = "Target"

        coef_fig = _latent_plots.create_loadings_plot(
            loadings=coef_matrix,
            feature_names=preprocessed_x_axis,
            loadings_components=coef_components,
            xlabel=xlabel,
            figsize=config.loadings_figsize,
            component_label=component_label,
        )
        coef_ax = coef_fig.axes[0]
        coef_ax.set_title("Regression Coefficients", fontsize=12, fontweight="bold")

        handles, _ = coef_ax.get_legend_handles_labels()
        if handles:
            if coef_matrix.shape[1] == 1:
                coef_ax.legend(handles, ["Coefficient"], loc="best")
            else:
                target_labels = [
                    f"Target {idx + 1}" for idx in range(coef_matrix.shape[1])
                ]
                coef_ax.legend(handles, target_labels, loc="best")

        figures["regression_coefficients"] = coef_fig

        # ------------------------------------------------------------------
        # Scores plots (X-scores and X vs Y scores)
        # ------------------------------------------------------------------
        scores_figures = self.create_latent_scores_figures(
            dataset=dataset,
            components=components_scores,
            color_by_y=color_by_y,
            annotate_by=annotate_by,
            figsize=config.scores_figsize,
            color_mode=color_mode,
        )
        figures.update(scores_figures)

        # X-scores vs Y-scores plots (training set only)
        x_scores = self.get_x_scores("train")
        y_scores = self.get_y_scores("train")
        _, y_train = self._get_raw_data("train")

        x_y_scores_figures = _latent_plots.create_x_vs_y_scores_plots(
            x_scores=x_scores,
            y_scores=y_scores,
            y_train=y_train,
            components=components_scores,
            color_by_y=color_by_y,
            annotate_by=annotate_by,
            figsize=config.scores_figsize,
            component_label=self.component_label,
            color_mode=color_mode,
        )
        figures.update(x_y_scores_figures)

        # ------------------------------------------------------------------
        # Distance plots (Hotelling T², Q residuals, leverage, studentized)
        # ------------------------------------------------------------------
        figures["distances_hotelling_q"] = self.create_latent_distance_figure(
            dataset=dataset,
            color_by_y=color_by_y,
            figsize=config.distances_figsize,
            annotate_by=annotate_by,
            color_mode=color_mode,
        )

        # Prepare data for regression diagnostics
        datasets_data: Dict[str, Dict[str, Any]] = {}
        for ds in datasets:
            X, y_true = self._get_raw_data(ds)
            y_pred = self._get_predictions(ds)
            datasets_data[ds] = {
                "X": X,
                "y": y_true,
                "y_true": y_true,
                "y_pred": y_pred,
            }

        # Q residuals vs Y residuals
        # Fit Q detector on training data for consistent limits
        X_train, _ = self._get_raw_data("train")
        q_detector = QResiduals(self.model, confidence=self.confidence)
        q_detector.fit(X_train)

        figures["distances_q_y_residuals"] = _latent_plots.create_q_vs_y_residuals_plot(
            datasets_data=datasets_data,
            model=self.model,
            confidence=self.confidence,
            color_by_y=color_by_y,
            figsize=config.distances_figsize,
            q_residuals_detector=q_detector,
            annotate_by=annotate_by,
            color_mode=color_mode,
        )

        # Leverage vs Studentized residuals
        figures["distances_leverage_studentized"] = create_regression_distances_plot(
            datasets_data=datasets_data,
            leverage_detector=self.leverage_detector,
            student_detector=self.studentized_detector,
            color_by_y=color_by_y,
            figsize=config.distances_figsize,
            annotate_by=annotate_by,
            color_mode=color_mode,
        )

        # ------------------------------------------------------------------
        # Regression diagnostic plots
        # ------------------------------------------------------------------
        # Predicted vs Actual
        figures["predicted_vs_actual"] = create_predicted_vs_actual_plot(
            datasets_data=datasets_data,
            color_by_y=color_by_y,
            figsize=config.regression_figsize,
            annotate_by=annotate_by,
            color_mode=color_mode,
        )

        # Residual scatter plot
        figures["residuals"] = create_y_residual_plot(
            datasets_data=datasets_data,
            color_by_y=color_by_y,
            figsize=config.regression_figsize,
            annotate_by=annotate_by,
            color_mode=color_mode,
        )

        # Q-Q plot
        figures["qq_plot"] = create_qq_plot(
            datasets_data=datasets_data,
            figsize=config.regression_figsize,
            confidence=self.confidence,
        )

        # Residual distribution
        figures["residual_distribution"] = create_residual_distribution_plot(
            datasets_data=datasets_data,
            figsize=config.regression_figsize,
        )

        # ------------------------------------------------------------------
        # Spectra plots (if preprocessing exists)
        # ------------------------------------------------------------------
        if self.transformer is not None:
            spectra_figs = self.inspect_spectra(
                dataset=datasets if use_suffix else datasets[0],
                color_by_y=color_by_y,
                figsize=config.spectra_figsize,
                color_mode=color_mode,
            )
            figures.update(spectra_figs)

        return figures
