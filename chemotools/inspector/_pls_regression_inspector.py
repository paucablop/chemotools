"""PLS Regression Inspector for model diagnostics and visualization."""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING, Union
import numpy as np
from sklearn.cross_decomposition._pls import _PLS
from sklearn.pipeline import Pipeline

if TYPE_CHECKING:
    import matplotlib.figure

from chemotools.outliers import HotellingT2, QResiduals

from ._base import _BaseInspector
from .mixins import LatentVariableMixin, RegressionMixin
from ._utils import (
    normalize_datasets,
    get_xlabel_for_features,
    get_default_scores_components,
    get_default_loadings_components,
)
from .helpers import _latent_space as _latent_plots
from .helpers._spectra import (
    create_spectra_plots_single_dataset,
    create_spectra_plots_multi_dataset,
)
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


class PLSRegressionInspector(RegressionMixin, LatentVariableMixin, _BaseInspector):
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
        self._hotelling_t2_limit: Optional[float] = None
        self._q_residuals_limit: Optional[float] = None

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

    def _get_preprocessed_wavenumbers(self) -> np.ndarray:
        """Get wavenumbers after feature selection."""
        return self._get_preprocessed_feature_names()

    # ==================================================================================
    # Public Methods
    # ==================================================================================

    # ------------------------------------------------------------------
    # LatentVariableMixin hooks
    # ------------------------------------------------------------------
    def get_latent_scores(self, dataset: str) -> np.ndarray:
        return self.get_x_scores(dataset)

    def get_latent_explained_variance(self) -> Optional[np.ndarray]:
        return self.get_explained_x_variance_ratio()

    def get_latent_loadings(self) -> np.ndarray:
        return self.get_x_loadings()

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

    def create_latent_scores_figures(
        self,
        dataset: Union[str, Sequence[str]],
        components: Union[int, Tuple[int, int], Sequence[Union[int, Tuple[int, int]]]],
        *,
        color_by_y: bool,
        annotate_by: Optional[Union[str, Dict[str, np.ndarray]]],
        figsize: Tuple[float, float],
    ) -> Dict[str, matplotlib.figure.Figure]:
        """Generate X-scores plots for requested datasets.

        For PLS, when multiple datasets are provided, only creates the combined
        multi-dataset plots (not individual per-dataset plots) to reduce clutter.

        Parameters
        ----------
        dataset : str or sequence of str
            Dataset(s) to plot ('train', 'test', 'val')
        components : tuple or sequence of tuples
            Component pairs to plot
        color_by_y : bool
            Whether to color by y values
        annotate_by : str or dict, optional
            Annotation specification
        figsize : tuple of float
            Figure size

        Returns
        -------
        dict
            Dictionary of figures with keys like 'scores_1', 'scores_2', etc.
        """
        from ._utils import normalize_datasets, normalize_components
        from .helpers import _latent_space as _latent_plots
        from chemotools.plotting._styles import DATASET_COLORS

        dataset_names = list(normalize_datasets(dataset))
        if not dataset_names:
            raise ValueError("At least one dataset is required for scores plotting")

        components_list = normalize_components(components)
        figures: Dict[str, matplotlib.figure.Figure] = {}
        multi_dataset = len(dataset_names) > 1
        explained_var = self.get_explained_x_variance_ratio()

        if multi_dataset:
            # For PLS with multiple datasets, only create combined plots
            datasets_data: Dict[str, Dict[str, Optional[np.ndarray]]] = {}
            for ds_name in dataset_names:
                x_scores = self.get_x_scores(ds_name)
                _, y = self._get_raw_data(ds_name)
                datasets_data[ds_name] = {
                    "scores": x_scores,
                    "y": y,
                }

            # Get training scores for confidence ellipse reference (even if train not requested)
            train_scores_for_ellipse = self.get_x_scores("train")

            # explained_var might be None, but the plot function expects an array
            # Use zeros if not available
            var_for_plot = (
                explained_var
                if explained_var is not None
                else np.zeros(self.nr_components)
            )

            for idx, component_spec in enumerate(components_list, start=1):
                fig = _latent_plots.create_scores_plot_multi_dataset(
                    component_spec=component_spec,
                    datasets_data=datasets_data,
                    explained_var=var_for_plot,
                    color_by_y=False,  # Color by dataset in multi-dataset mode
                    annotate_by=annotate_by,
                    figsize=figsize,
                    component_label=self.component_label,
                    train_scores_for_ellipse=train_scores_for_ellipse,
                    confidence=self.confidence,
                )
                figures[f"scores_{idx}"] = fig
        else:
            # Single dataset - create one plot per component spec
            dataset_name = dataset_names[0]
            x_scores = self.get_x_scores(dataset_name)
            _, y = self._get_raw_data(dataset_name)

            # explained_var might be None, but the plot function expects an array
            # Use zeros if not available
            var_for_plot = (
                explained_var
                if explained_var is not None
                else np.zeros(self.nr_components)
            )

            # Get training scores for ellipse reference (if not already train dataset)
            train_scores_for_ellipse = None
            if dataset_name.lower() != "train":
                try:
                    train_scores_for_ellipse = self.get_x_scores("train")
                except (ValueError, KeyError):
                    # Train dataset not available, skip ellipse
                    pass

            for idx, component_spec in enumerate(components_list, start=1):
                fig = _latent_plots.create_scores_plot_single_dataset(
                    component_spec=component_spec,
                    scores=x_scores,
                    y=y,
                    explained_var=var_for_plot,
                    dataset_name=dataset_name,
                    color_by_y=color_by_y,
                    annotate_by=annotate_by,
                    figsize=figsize,
                    component_label=self.component_label,
                    dataset_color=DATASET_COLORS.get(dataset_name, "gray"),
                    confidence=self.confidence,
                    train_scores_for_ellipse=train_scores_for_ellipse,
                )
                figures[f"scores_{idx}"] = fig

        return figures

    def _create_x_vs_y_scores_figures(
        self,
        components: Union[int, Tuple[int, int], Sequence[Union[int, Tuple[int, int]]]],
        color_by_y: bool,
        annotate_by: Optional[Union[str, Dict[str, np.ndarray]]],
        figsize: Tuple[float, float],
    ) -> Dict[str, matplotlib.figure.Figure]:
        """Create X-scores vs Y-scores plots for PLS (training set only).

        Note: Only 2D component pairs (tuples) will be plotted. Single component
        specifications (ints) will be silently skipped since X vs Y scores
        requires two components.

        Parameters
        ----------
        components : int, tuple, or sequence
            Component pairs to plot. Only tuple specifications will be used;
            int specifications are ignored.
        color_by_y : bool
            Whether to color by y values
        annotate_by : str or dict, optional
            Annotation specification
        figsize : tuple of float
            Figure size

        Returns
        -------
        dict
            Dictionary of figures with keys like 'x_vs_y_scores_1', 'x_vs_y_scores_2', etc.
            Empty dict if no 2D component pairs are provided.
        """
        from chemotools.plotting import ScoresPlot
        from ._utils import normalize_components, prepare_annotations

        components_list = normalize_components(components)
        figures: Dict[str, matplotlib.figure.Figure] = {}

        # Get training data
        x_scores = self.get_x_scores("train")
        y_scores = self.get_y_scores("train")
        _, y_train = self._get_raw_data("train")

        for idx, component_spec in enumerate(components_list, start=1):
            # Only create 2D plots (component pairs)
            if isinstance(component_spec, tuple):
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots(figsize=figsize)

                # Create combined scores array [X-score, Y-score]
                combined_scores = np.column_stack(
                    [
                        x_scores[:, component_spec[0]],
                        y_scores[:, component_spec[1]],
                    ]
                )

                # Determine color_by parameter
                color_reference = (
                    y_train if color_by_y and y_train is not None else None
                )

                # Create ScoresPlot
                plot = ScoresPlot(
                    scores=combined_scores,
                    components=(0, 1),  # We already selected the right columns
                    color_by=color_reference,
                    label="Train",
                    colormap=None,
                    confidence_ellipse=self.confidence,
                )
                plot.render(ax)

                # Add annotations if requested
                labels = prepare_annotations(annotate_by, "train", x_scores, y_train)
                if labels is not None:
                    from chemotools.plotting._utils import annotate_points

                    annotate_points(
                        ax,
                        combined_scores[:, 0],
                        combined_scores[:, 1],
                        labels,
                        fontsize=8,
                        alpha=0.7,
                        xytext=(3, 3),
                        textcoords="offset points",
                    )

                # Set custom labels
                ax.set_xlabel(
                    f"X-{self.component_label}{component_spec[0] + 1}", fontsize=10
                )
                ax.set_ylabel(
                    f"Y-{self.component_label}{component_spec[1] + 1}", fontsize=10
                )
                ax.set_title(
                    f"X-scores vs Y-scores: {self.component_label}{component_spec[0] + 1} vs {self.component_label}{component_spec[1] + 1}",
                    fontsize=12,
                    fontweight="bold",
                )
                ax.grid(alpha=0.3)

                plt.tight_layout()
                figures[f"x_vs_y_scores_{idx}"] = fig

        return figures

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
        scores_figsize: Tuple[float, float] = (6, 6),
        loadings_figsize: Tuple[float, float] = (10, 5),
        variance_figsize: Tuple[float, float] = (10, 5),
        spectra_figsize: Tuple[float, float] = (12, 5),
        distances_figsize: Tuple[float, float] = (8, 6),
        regression_figsize: Tuple[float, float] = (8, 6),
    ) -> Dict[str, matplotlib.figure.Figure]:
        """Create multiple independent PLS diagnostic plots.

        This method creates separate figure windows for:
        - One or more scores plots (X-scores, default depends on model components)
        - Multiple loadings plots (X-loadings, X-weights, X-rotations, coefficients)
        - Explained variance plots (X and Y spaces, if available)
        - Raw and preprocessed spectra plots (if preprocessing exists)
        - Regression diagnostic plots (predicted vs actual, residuals, Q-Q, distribution)
        - Distance plots (Hotelling's T² vs Q residuals, Leverage vs Studentized residuals)

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
            selects based on number of components:
            - 1 component: 0
            - 2 components: [0, 1]
            - 3+ components: [0, 1, 2]
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
                        Dictionary containing all created figures. Keys include:
                        - 'scores_1', 'scores_2', ...: X-scores plots (combined multi-dataset when multiple datasets provided)
                        - 'x_vs_y_scores_1', 'x_vs_y_scores_2', ...: X-scores vs Y-scores plots (training set only)
                        - 'loadings_x', 'loadings_weights', 'loadings_rotations': X-related loadings plots
                        - 'regression_coefficients': Regression coefficient traces (one per target when multi-output)
                        - 'variance_x', 'variance_y': Explained variance plots (when available)
                        - 'distances_hotelling_q', 'distances_leverage_studentized': Distance diagnostics
                        - 'predicted_vs_actual', 'residuals', 'qq_plot', 'residual_distribution': Regression diagnostics
                        - 'raw_spectra', 'preprocessed_spectra': Spectra plots (when preprocessing exists)
        """
        # Generate smart defaults based on number of components
        if components_scores is None:
            components_scores = get_default_scores_components(self.nr_components)
        if loadings_components is None:
            loadings_components = get_default_loadings_components(self.nr_components)

        figures = {}

        datasets = normalize_datasets(dataset)
        use_suffix = len(datasets) > 1

        xlabel = get_xlabel_for_features(self._wavenumbers is not None)
        preprocessed_wavenumbers = self._get_preprocessed_wavenumbers()

        x_var = self.get_explained_x_variance_ratio()
        if x_var is not None:
            variance_x_fig = self.create_latent_variance_figure(
                variance_threshold=variance_threshold,
                figsize=variance_figsize,
            )
            if variance_x_fig is not None:
                variance_x_fig.axes[0].set_title(
                    "Explained Variance in X-space",
                    fontsize=12,
                    fontweight="bold",
                )
                figures["variance_x"] = variance_x_fig

        y_var = self.get_explained_y_variance_ratio()
        if y_var is not None:
            variance_y_fig = _latent_plots.create_variance_plot(
                explained_variance_ratio=y_var,
                variance_threshold=variance_threshold,
                figsize=variance_figsize,
            )
            variance_y_fig.axes[0].set_title(
                "Explained Variance in Y-space", fontsize=12, fontweight="bold"
            )
            figures["variance_y"] = variance_y_fig

        loadings_x_fig = self.create_latent_loadings_figure(
            loadings_components=loadings_components,
            xlabel=xlabel,
            figsize=loadings_figsize,
        )
        loadings_x_fig.axes[0].set_title("X-Loadings", fontsize=12, fontweight="bold")
        figures["loadings_x"] = loadings_x_fig

        figures["loadings_weights"] = _latent_plots.create_loadings_plot(
            loadings=self.get_x_weights(),
            feature_names=preprocessed_wavenumbers,
            loadings_components=loadings_components,
            xlabel=xlabel,
            figsize=loadings_figsize,
            component_label=self.component_label,
        )
        figures["loadings_weights"].axes[0].set_title(
            "X-Weights", fontsize=12, fontweight="bold"
        )

        figures["loadings_rotations"] = _latent_plots.create_loadings_plot(
            loadings=self.get_x_rotations(),
            feature_names=preprocessed_wavenumbers,
            loadings_components=loadings_components,
            xlabel=xlabel,
            figsize=loadings_figsize,
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
            feature_names=preprocessed_wavenumbers,
            loadings_components=coef_components,
            xlabel=xlabel,
            figsize=loadings_figsize,
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

        scores_figures = self.create_latent_scores_figures(
            dataset=dataset,
            components=components_scores,
            color_by_y=color_by_y,
            annotate_by=annotate_by,
            figsize=scores_figsize,
        )
        figures.update(scores_figures)

        # X-scores vs Y-scores plots (training set only)
        x_y_scores_figures = self._create_x_vs_y_scores_figures(
            components=components_scores,
            color_by_y=color_by_y,
            annotate_by=annotate_by,
            figsize=scores_figsize,
        )
        figures.update(x_y_scores_figures)

        figures["distances_hotelling_q"] = self.create_latent_distance_figure(
            dataset=dataset,
            color_by_y=color_by_y,
            figsize=distances_figsize,
        )

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
                confidence=self.confidence,
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
                confidence=self.confidence,
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
