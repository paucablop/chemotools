"""PCA Inspector for model diagnostics and visualization."""

from __future__ import annotations
from typing import Dict, Optional, Sequence, Tuple, Union, TYPE_CHECKING, Literal
import numpy as np
from sklearn.decomposition._base import _BasePCA
from sklearn.pipeline import Pipeline

from chemotools.outliers import QResiduals, HotellingT2

if TYPE_CHECKING:
    from matplotlib.figure import Figure


from ._base import _BaseInspector, InspectorPlotConfig
from .mixins import LatentVariableMixin, SpectraMixin
from ._utils import (
    normalize_datasets,
    get_xlabel_for_features,
    get_default_scores_components,
    get_default_loadings_components,
    select_components,
)


class PCAInspector(SpectraMixin, LatentVariableMixin, _BaseInspector):
    """Inspector for PCA model diagnostics and visualization.

    This class provides a unified interface for inspecting PCA models by creating
    multiple independent diagnostic plots. Instead of complex dashboards with many
    subplots, each method produces several separate figure windows that are easier
    to customize, save, and interact with individually.

    The inspector provides convenience methods that create multiple independent plots:
    - inspect(): Creates all diagnostic plots (scores, loadings, explained variance)
    - inspect_spectra(): Creates raw and preprocessed spectra plots (if preprocessing exists)

    Parameters
    ----------
    model : _BasePCA or Pipeline
        Fitted PCA model or pipeline ending with PCA
    X_train : array-like of shape (n_samples, n_features)
        Training data
    y_train : array-like of shape (n_samples,), optional
        Training labels/targets (for coloring plots)
    X_test : array-like of shape (n_samples, n_features), optional
        Test data
    y_test : array-like of shape (n_samples,), optional
        Test labels/targets
    X_val : array-like of shape (n_samples, n_features), optional
        Validation data
    y_val : array-like of shape (n_samples,), optional
        Validation labels/targets
    x_axis : array-like of shape (n_features,), optional
        Feature names (e.g., wavenumbers for spectroscopy)
        If None, uses feature indices
    confidence : float, default=0.95
        Confidence level for outlier detection limits (Hotelling's T² and Q residuals).
        Must be between 0 and 1. Used to calculate critical values for diagnostic plots.

    Attributes
    ----------
    model : _BasePCA or Pipeline
        The original model passed to the inspector
    estimator : _BasePCA
        The PCA estimator
    transformer : Pipeline or None
        Preprocessing pipeline before PCA (if model was a Pipeline)
    nr_components : int
        Number of principal components
    nr_features : int
        Number of features in original data
    nr_samples : dict
        Number of samples in each dataset
    x_axis : ndarray
        Feature names/indices
    confidence : float
        Confidence level for outlier detection
    hotelling_t2_limit : float
        Critical value for Hotelling's T² statistic (computed on training data)
    q_residuals_limit : float
        Critical value for Q residuals statistic (computed on training data)

    Examples
    --------
    >>> from sklearn.decomposition import PCA
    >>> from sklearn.pipeline import make_pipeline
    >>> from sklearn.preprocessing import StandardScaler
    >>> from chemotools.datasets import load_fermentation_train
    >>> from chemotools.inspector import PCAInspector
    >>>
    >>> # Load data
    >>> X, y = load_fermentation_train()
    >>> # Create and fit pipeline
    >>> pipeline = make_pipeline(
    ...     StandardScaler(),
    ...     PCA(n_components=5)
    ... )
    >>> pipeline.fit(X)
    >>>
    >>> # Create inspector
    >>> inspector = PCAInspector(pipeline, X, y, x_axis=X.columns)
    >>>
    >>> # Print summary table
    >>> inspector.summary()
    >>>
    >>> # Create all diagnostic plots (multiple independent figures)
    >>> inspector.inspect()  # Creates scores, loadings, and variance plots
    >>>
    >>> # Compare preprocessing (creates 2 independent figures)
    >>> inspector.inspect_spectra()
    >>>
    >>> # Access underlying data for custom analysis
    >>> scores = inspector.get_scores('train')
    >>> loadings = inspector.get_loadings([0, 1, 2])
    """

    component_label = "PC"

    def __init__(
        self,
        model: Union[_BasePCA, Pipeline],
        X_train: np.ndarray,
        y_train: Optional[np.ndarray] = None,
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
            supervised=False,
            feature_names=x_axis,
            confidence=confidence,
        )

        self._scores_cache: Dict[str, np.ndarray] = {}

    # ==================================================================================
    # Public Methods
    # ==================================================================================

    # ------------------------------------------------------------------
    # LatentVariableMixin hooks
    # ------------------------------------------------------------------
    def get_latent_scores(self, dataset: str) -> np.ndarray:
        """Hook for LatentVariableMixin - returns scores."""
        return self.get_scores(dataset)

    def get_latent_explained_variance(self) -> Optional[np.ndarray]:
        """Hook for LatentVariableMixin - returns explained variance ratio."""
        return self.get_explained_variance_ratio()

    def get_latent_loadings(self) -> np.ndarray:
        """Hook for LatentVariableMixin - returns loadings."""
        return self.get_loadings()

    # ------------------------------------------------------------------
    # Scores methods
    # ------------------------------------------------------------------
    def get_scores(self, dataset: str = "train") -> np.ndarray:
        """Get PCA scores for specified dataset.

        Parameters
        ----------
        dataset : {'train', 'test', 'val'}, default='train'
            Which dataset to get scores for

        Returns
        -------
        scores : ndarray of shape (n_samples, n_components)
            PCA scores
        """
        if dataset not in self._scores_cache:
            X_preprocessed = self._get_preprocessed_data(dataset)
            scores = self.estimator.transform(X_preprocessed)
            self._scores_cache[dataset] = scores
        return self._scores_cache[dataset]

    # ------------------------------------------------------------------
    # Loadings methods
    # ------------------------------------------------------------------
    def get_loadings(
        self, components: Optional[Union[int, Sequence[int]]] = None
    ) -> np.ndarray:
        """Get PCA loadings.

        Parameters
        ----------
        components : int, list of int, or None, default=None
            Which components to return. If None, returns all components.

        Returns
        -------
        loadings : ndarray of shape (n_features, n_components_selected)
            PCA loadings (components transposed)
        """
        loadings = self.estimator.components_.T
        return select_components(loadings, components)

    # ------------------------------------------------------------------
    # Variance methods
    # ------------------------------------------------------------------
    def get_explained_variance_ratio(self) -> np.ndarray:
        """Get explained variance ratio for all components.

        Returns
        -------
        explained_variance_ratio : ndarray of shape (n_components,)
            Explained variance ratio
        """
        return self.estimator.explained_variance_ratio_

    # ------------------------------------------------------------------
    # Summary method
    # ------------------------------------------------------------------
    def summary(self) -> Dict[str, Union[str, int, float, Dict, np.ndarray]]:
        """Get a summary dictionary of the PCA model.

        Returns
        -------
        summary : dict
            Dictionary containing model information with keys:
            - 'model_type': Name of the PCA estimator class
            - 'has_preprocessing': Whether preprocessing pipeline exists
            - 'nr_features': Number of features in original data
            - 'nr_components': Number of principal components
            - 'nr_samples': Dictionary with sample counts per dataset
            - 'explained_variance_ratio': Array of variance ratios per component
            - 'cumulative_variance': Array of cumulative variance
            - 'pc_variances': Dictionary with variance explained by PC1, PC2, PC3
            - 'total_variance': Total variance explained by all components
            - 'variance_thresholds': Dictionary with components needed for 90%, 95%, 99% variance
            - 'preprocessing_steps': List of preprocessing step names (if available)

        Examples
        --------
        >>> inspector = PCAInspector(pca, X_train, y_train)
        >>> summary = inspector.summary()
        >>> print(f"Model type: {summary['model_type']}")
        >>> print(f"PC1 explains: {summary['pc_variances']['PC1']:.2f}%")
        >>> print(f"Components for 95% variance: {summary['variance_thresholds']['95%']}")
        """
        # Start with common summary fields
        summary_dict = self._base_summary()

        # Calculate cumulative variance
        explained_var = self.get_explained_variance_ratio()
        cumsum = np.cumsum(explained_var)

        # Find components for common variance thresholds
        n_90 = (
            np.argmax(cumsum >= 0.90) + 1
            if np.any(cumsum >= 0.90)
            else self.nr_components
        )
        n_95 = (
            np.argmax(cumsum >= 0.95) + 1
            if np.any(cumsum >= 0.95)
            else self.nr_components
        )
        n_99 = (
            np.argmax(cumsum >= 0.99) + 1
            if np.any(cumsum >= 0.99)
            else self.nr_components
        )

        # Build PC variances dictionary
        pc_variances = {
            "PC1": explained_var[0] * 100,
        }
        if self.nr_components > 1:
            pc_variances["PC2"] = explained_var[1] * 100
        if self.nr_components > 2:
            pc_variances["PC3"] = explained_var[2] * 100

        # Add PCA-specific fields
        summary_dict.update(
            {
                "explained_variance_ratio": explained_var,
                "cumulative_variance": cumsum,
                "pc_variances": pc_variances,
                "total_variance": cumsum[-1] * 100,
                "variance_thresholds": {
                    "90%": {
                        "n_components": n_90,
                        "actual_variance": cumsum[n_90 - 1] * 100,
                    },
                    "95%": {
                        "n_components": n_95,
                        "actual_variance": cumsum[n_95 - 1] * 100,
                    },
                    "99%": {
                        "n_components": n_99,
                        "actual_variance": cumsum[n_99 - 1] * 100,
                    },
                },
            }
        )

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
        color_by: Optional[Union[str, Dict[str, np.ndarray]]] = None,
        annotate_by: Optional[Union[str, Dict[str, np.ndarray]]] = None,
        plot_config: Optional[InspectorPlotConfig] = None,
        color_mode: Optional[Literal["continuous", "categorical"]] = None,
        **kwargs,
    ) -> Dict[str, Figure]:
        """Create all diagnostic plots for the PCA model.

        Parameters
        ----------
        dataset : str or sequence of str, default='train'
            Dataset(s) to visualize. Can be 'train', 'test', 'val', or a list.
        components_scores : int, tuple, or sequence, optional
            Components to plot for scores.
            - Int: Creates one 1D scatter plot (e.g., 0 for PC1 vs sample index)
            - Single tuple (x, y): Creates one 2D scatter plot (e.g., (0, 1) for PC1 vs PC2)
            - Sequence: Creates multiple plots (e.g., ((0, 1), (1, 2), 0) or [0, 1, (0, 1)])
        loadings_components : int, sequence of int, or None, optional
            Which components to show in loadings plot. If None (default), automatically
            selects all available components:
            - 1 component: 0
            - 2+ components: [0, 1, ..., n_components-1] (all components)
        variance_threshold : float, default=0.95
            Threshold line for explained variance plot
        color_by : str or dict, optional
            Coloring specification:
            - 'y': Color by y values (if available)
            - 'sample_index': Color by sample index
            - dict: Map dataset names to color arrays
            - None: Color by dataset (for multi-dataset plots) or 'y' (for single dataset)
        annotate_by : str or dict, optional
            Annotations for score plot points. Can be:
            - 'sample_index': Annotate with sample indices (0, 1, 2, ...)
            - 'y': Annotate with y values (only for single dataset)
            - dict: Dictionary mapping dataset names to annotation arrays
              e.g., {'train': ['A', 'B', 'C'], 'test': ['D', 'E']}
            If None (default), no annotations are added.
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
                        Dictionary containing all created figures with keys:
                        - 'scores_1', 'scores_2', ...: Combined scores plots (95% confidence ellipses)
                        - 'scores_1_train', 'scores_1_test', ...: Dataset-specific copies of each scores plot
                            (only when multiple datasets are provided); each plot uses a dedicated dataset colour
                        - 'loadings': Loadings plot
                        - 'variance': Explained variance plot
                        - 'distances': Diagnostic distances plot (Hotelling's T² vs Q residuals)
                        - 'raw_spectra', 'preprocessed_spectra': Spectra plots (if preprocessing exists)

                        Combined scores plots render all requested datasets on shared axes, coloured by
                        dataset. The number of 'scores_N*' entries depends on the ``components_scores``
                        parameter.

        Examples
        --------
        >>> inspector = PCAInspector(pca, X_train, y_train)
        >>> # Default: 2 scores plots + loadings + variance + spectra (if preprocessing exists)
        >>> figs = inspector.inspect()
        >>> # Multiple datasets for comparison
        >>> inspector.X_test = X_test
        >>> inspector.y_test = y_test
        >>> figs = inspector.inspect(dataset=["train", "test"])
        >>> # Access individual figures
        >>> figs["scores_1_train"].savefig("scores_1_train.png")
        >>> figs["scores_1_test"].savefig("scores_1_test.png")
        >>> # Single 2D scores plot (PC1 vs PC2)
        >>> figs = inspector.inspect(components_scores=(0, 1))
        >>> # Single 1D scores plot (PC1 vs sample index or y)
        >>> figs = inspector.inspect(components_scores=0)
        >>> # Three plots: 2D, 2D, and 1D
        >>> figs = inspector.inspect(components_scores=((0, 1), (1, 2), 2))
        >>> # Mix of 1D and 2D plots
        >>> figs = inspector.inspect(components_scores=[0, 1, (0, 1)])
        >>> # Save individual plots
        >>> figs['scores_1'].savefig('scores_pc1_pc2.png')
        >>> figs['loadings'].savefig('loadings.png')
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

        # Auto-resolve color_by if None
        # If single dataset, default to coloring by 'y' (if available)
        # If multiple datasets, default to coloring by dataset (color_by=None)
        if color_by is None and not use_suffix:
            color_by = "y"

        # ------------------------------------------------------------------
        # Variance plot
        # ------------------------------------------------------------------
        variance_fig = self.create_latent_variance_figure(
            variance_threshold=variance_threshold,
            figsize=config.variance_figsize,
        )
        if variance_fig is not None:
            figures["variance"] = variance_fig

        # ------------------------------------------------------------------
        # Loadings plot
        # ------------------------------------------------------------------
        xlabel = get_xlabel_for_features(self.feature_names is not None)
        figures["loadings"] = self.create_latent_loadings_figure(
            loadings_components=loadings_components,
            xlabel=xlabel,
            figsize=config.loadings_figsize,
        )

        # ------------------------------------------------------------------
        # Scores plots
        # ------------------------------------------------------------------
        scores_figures = self.create_latent_scores_figures(
            dataset=dataset,
            components=components_scores,
            color_by=color_by,
            annotate_by=annotate_by,
            figsize=config.scores_figsize,
            color_mode=color_mode,
        )
        figures.update(scores_figures)

        # ------------------------------------------------------------------
        # Distance plot (Hotelling T² vs Q residuals)
        # ------------------------------------------------------------------
        # Fit detectors once on training data for consistent limits and efficiency
        X_train, _ = self._get_raw_data("train")

        hotelling_detector = HotellingT2(self.model, confidence=self.confidence)
        hotelling_detector.fit(X_train)

        q_detector = QResiduals(self.model, confidence=self.confidence)
        q_detector.fit(X_train)

        figures["distances"] = self.create_latent_distance_figure(
            dataset=dataset,
            color_by=color_by,
            figsize=config.distances_figsize,
            annotate_by=annotate_by,
            color_mode=color_mode,
            hotelling_detector=hotelling_detector,
            q_residuals_detector=q_detector,
        )

        # ------------------------------------------------------------------
        # Spectra plots (if preprocessing exists)
        # ------------------------------------------------------------------
        if self.transformer is not None:
            spectra_figs = self.inspect_spectra(
                dataset=datasets if use_suffix else datasets[0],
                color_by=color_by,
                figsize=config.spectra_figsize,
                color_mode=color_mode,
            )
            figures.update(spectra_figs)

        return figures
        """Create multiple independent PCA diagnostic plots.

        This method creates separate figure windows for:
        - One or more scores plots (default depends on model components)
        - One loadings plot (overlaid or single component)
        - One explained variance plot
        - Raw and preprocessed spectra plots (if include_spectra=True and preprocessing exists)

        Each plot is a separate figure that can be independently manipulated,
        saved, or displayed.

        Parameters
        ----------
        dataset : Union[str, Sequence[str]], default='train'
            Dataset(s) to inspect. Can be a single dataset name ("train", "test", or "val")
            or a sequence of dataset names (e.g., ["train", "test"]).
        components_scores : int, tuple of two ints, sequence, or None, optional
            Component(s) for scores plots. If None (default), automatically selects based
            on number of components:
            - 1 component: 0 (PC1 vs sample index/y)
            - 2 components: (0, 1) (PC1 vs PC2)
            - 3+ components: ((0, 1), (1, 2)) (two 2D plots)
            Can also be manually specified as:
            - Single int: Creates one 1D plot (e.g., 0 for PC1 vs sample index/y-value)
            - Single tuple (x, y): Creates one 2D scatter plot (e.g., (0, 1) for PC1 vs PC2)
            - Sequence: Creates multiple plots (e.g., ((0, 1), (1, 2), 0) or [0, 1, (0, 1)])
        loadings_components : int, sequence of int, or None, optional
            Which components to show in loadings plot. If None (default), automatically
            selects all available components:
            - 1 component: 0
            - 2+ components: [0, 1, ..., n_components-1] (all components)
        variance_threshold : float, default=0.95
            Threshold line for explained variance plot
        color_by : str or dict, default='y'
            Coloring specification:
            - 'y': Color by y values (if available)
            - 'sample_index': Color by sample index
            - dict: Map dataset names to color arrays
        annotate_by : str or dict, optional
            Annotations for score plot points. Can be:
            - 'sample_index': Annotate with sample indices (0, 1, 2, ...)
            - 'y': Annotate with y values (only for single dataset)
            - dict: Dictionary mapping dataset names to annotation arrays
              e.g., {'train': ['A', 'B', 'C'], 'test': ['D', 'E']}
            If None (default), no annotations are added.
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
                        Dictionary containing all created figures with keys:
                        - 'scores_1', 'scores_2', ...: Combined scores plots (95% confidence ellipses)
                        - 'scores_1_train', 'scores_1_test', ...: Dataset-specific copies of each scores plot
                            (only when multiple datasets are provided); each plot uses a dedicated dataset colour
                        - 'loadings': Loadings plot
                        - 'variance': Explained variance plot
                        - 'distances': Diagnostic distances plot (Hotelling's T² vs Q residuals)
                        - 'raw_spectra', 'preprocessed_spectra': Spectra plots (if preprocessing exists)

                        Combined scores plots render all requested datasets on shared axes, coloured by
                        dataset. The number of 'scores_N*' entries depends on the ``components_scores``
                        parameter.

        Examples
        --------
        >>> inspector = PCAInspector(pca, X_train, y_train)
        >>> # Default: 2 scores plots + loadings + variance + spectra (if preprocessing exists)
        >>> figs = inspector.inspect()
        >>> # Multiple datasets for comparison
        >>> inspector.X_test = X_test
        >>> inspector.y_test = y_test
        >>> figs = inspector.inspect(dataset=["train", "test"])
        >>> # Access individual figures
        >>> figs["scores_1_train"].savefig("scores_1_train.png")
        >>> figs["scores_1_test"].savefig("scores_1_test.png")
        >>> # Single 2D scores plot (PC1 vs PC2)
        >>> figs = inspector.inspect(components_scores=(0, 1))
        >>> # Single 1D scores plot (PC1 vs sample index or y)
        >>> figs = inspector.inspect(components_scores=0)
        >>> # Three plots: 2D, 2D, and 1D
        >>> figs = inspector.inspect(components_scores=((0, 1), (1, 2), 2))
        >>> # Mix of 1D and 2D plots
        >>> figs = inspector.inspect(components_scores=[0, 1, (0, 1)])
        >>> # Save individual plots
        >>> figs['scores_1'].savefig('scores_pc1_pc2.png')
        >>> figs['loadings'].savefig('loadings.png')
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

        # ------------------------------------------------------------------
        # Variance plot
        # ------------------------------------------------------------------
        variance_fig = self.create_latent_variance_figure(
            variance_threshold=variance_threshold,
            figsize=config.variance_figsize,
        )
        if variance_fig is not None:
            figures["variance"] = variance_fig

        # ------------------------------------------------------------------
        # Loadings plot
        # ------------------------------------------------------------------
        xlabel = get_xlabel_for_features(self.feature_names is not None)
        figures["loadings"] = self.create_latent_loadings_figure(
            loadings_components=loadings_components,
            xlabel=xlabel,
            figsize=config.loadings_figsize,
        )

        # ------------------------------------------------------------------
        # Scores plots
        # ------------------------------------------------------------------
        scores_figures = self.create_latent_scores_figures(
            dataset=dataset,
            components=components_scores,
            color_by=color_by,
            annotate_by=annotate_by,
            figsize=config.scores_figsize,
            color_mode=color_mode,
        )
        figures.update(scores_figures)

        # ------------------------------------------------------------------
        # Distance plot (Hotelling T² vs Q residuals)
        # ------------------------------------------------------------------
        # Fit detectors once on training data for consistent limits and efficiency
        X_train, _ = self._get_raw_data("train")

        hotelling_detector = HotellingT2(self.model, confidence=self.confidence)
        hotelling_detector.fit(X_train)

        q_detector = QResiduals(self.model, confidence=self.confidence)
        q_detector.fit(X_train)

        figures["distances"] = self.create_latent_distance_figure(
            dataset=dataset,
            color_by=color_by,
            figsize=config.distances_figsize,
            annotate_by=annotate_by,
            color_mode=color_mode,
            hotelling_detector=hotelling_detector,
            q_residuals_detector=q_detector,
        )

        # ------------------------------------------------------------------
        # Spectra plots (if preprocessing exists)
        # ------------------------------------------------------------------
        if self.transformer is not None:
            spectra_figs = self.inspect_spectra(
                dataset=datasets if use_suffix else datasets[0],
                color_by=color_by,
                figsize=config.spectra_figsize,
                color_mode=color_mode,
            )
            figures.update(spectra_figs)

        return figures
