"""PCA Inspector for model diagnostics and visualization."""

from __future__ import annotations
from typing import Dict, Optional, Sequence, Tuple, Union, TYPE_CHECKING
import numpy as np
from sklearn.decomposition._base import _BasePCA
from sklearn.pipeline import Pipeline

if TYPE_CHECKING:
    import matplotlib.figure

from chemotools.outliers import HotellingT2, QResiduals

from ._base import _BaseInspector
from .mixins import LatentVariableMixin
from ._utils import (
    normalize_datasets,
    get_xlabel_for_features,
)
from ._plot_utils_spectra import (
    create_spectra_plots_single_dataset,
    create_spectra_plots_multi_dataset,
)


class PCAInspector(LatentVariableMixin, _BaseInspector):
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
    wavenumbers : array-like of shape (n_features,), optional
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
    wavenumbers : ndarray
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
    >>> inspector = PCAInspector(pipeline, X, y, wavenumbers=X.columns)
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
            supervised=False,
            feature_names=wavenumbers,
        )

        if not 0 < confidence < 1:
            raise ValueError(f"confidence must be between 0 and 1, got {confidence}")
        self._confidence = confidence

        if self.feature_names is not None:
            self._wavenumbers = np.array(self.feature_names, copy=True)
        else:
            self._wavenumbers = np.arange(self.n_features_in_)

        self._scores_cache: Dict[str, np.ndarray] = {}
        self._hotelling_t2_limit: Optional[float] = None
        self._q_residuals_limit: Optional[float] = None

    # ==================================================================================
    # Properties
    # ==================================================================================

    @property
    def model(self) -> Union[_BasePCA, Pipeline]:
        """Return the original model (PCA or Pipeline)."""
        return self._model

    @property
    def estimator(self) -> _BasePCA:
        """Return the PCA estimator."""
        return self.estimator_

    @property
    def transformer(self) -> Optional[Pipeline]:
        """Return the preprocessing pipeline (if available)."""
        return super().transformer

    @property
    def nr_components(self) -> int:
        """Return the number of principal components."""
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
        """Return the Hotelling's T² critical value at the specified confidence level.

        Calculated using the training data. The limit is cached after first calculation.

        Returns
        -------
        limit : float
            Critical value for Hotelling's T² statistic

        Examples
        --------
        >>> inspector = PCAInspector(pca, X_train, confidence=0.95)
        >>> t2_limit = inspector.hotelling_t2_limit
        >>> # Check if samples are outliers
        >>> t2_values = inspector.get_scores('test')  # Use custom method if needed
        """
        if self._hotelling_t2_limit is None:
            hotelling = HotellingT2(self.model, confidence=self._confidence)
            X_train, _ = self._get_raw_data("train")
            hotelling.fit(X_train)
            self._hotelling_t2_limit = hotelling.critical_value_
        return self._hotelling_t2_limit

    @property
    def q_residuals_limit(self) -> float:
        """Return the Q residuals critical value at the specified confidence level.

        Calculated using the training data. The limit is cached after first calculation.

        Returns
        -------
        limit : float
            Critical value for Q residuals statistic

        Examples
        --------
        >>> inspector = PCAInspector(pca, X_train, confidence=0.99)
        >>> q_limit = inspector.q_residuals_limit
        >>> # Check if samples are outliers
        >>> q_values = inspector.get_scores('test')  # Use custom method if needed
        """
        if self._q_residuals_limit is None:
            q_detector = QResiduals(self.model, confidence=self._confidence)
            X_train, _ = self._get_raw_data("train")
            q_detector.fit(X_train)
            self._q_residuals_limit = q_detector.critical_value_
        return self._q_residuals_limit

    # ==================================================================================
    # Private Methods
    # ==================================================================================

    def _get_preprocessed_wavenumbers(self) -> np.ndarray:
        """Get wavenumbers after feature selection.

        Returns
        -------
        wavenumbers : np.ndarray
            Wavenumbers/feature indices after feature selection. If no feature
            selector is present, returns the original wavenumbers.
        """
        return self._get_preprocessed_feature_names()

    # ==================================================================================
    # Public Methods
    # ==================================================================================

    # ------------------------------------------------------------------
    # LatentVariableMixin hooks
    # ------------------------------------------------------------------
    def get_latent_scores(self, dataset: str) -> np.ndarray:
        return self.get_scores(dataset)

    def get_latent_explained_variance(self) -> Optional[np.ndarray]:
        return self.get_explained_variance_ratio()

    def get_latent_loadings(self) -> np.ndarray:
        return self.get_loadings()

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

        if components is not None:
            if isinstance(components, int):
                components = [components]
            loadings = loadings[:, components]

        return loadings

    def get_explained_variance_ratio(self) -> np.ndarray:
        """Get explained variance ratio for all components.

        Returns
        -------
        explained_variance_ratio : ndarray of shape (n_components,)
            Explained variance ratio
        """
        return self.estimator.explained_variance_ratio_

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

        # Build summary dictionary
        summary_dict = {
            "model_type": type(self.estimator).__name__,
            "has_preprocessing": self.transformer is not None,
            "nr_features": self.nr_features,
            "nr_components": self.nr_components,
            "nr_samples": self.nr_samples.copy(),
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
    ) -> Dict[str, matplotlib.figure.Figure]:
        """Create multiple independent PCA diagnostic plots.

        This method creates separate figure windows for:
        - One or more scores plots (default: PC1 vs PC2 and PC2 vs PC3)
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
        components_scores : int, tuple of two ints, or sequence, default=((0, 1), (1, 2))
            Component(s) for scores plots. Can be:
            - Single int: Creates one 1D plot (e.g., 0 for PC1 vs sample index/y-value)
            - Single tuple (x, y): Creates one 2D scatter plot (e.g., (0, 1) for PC1 vs PC2)
            - Sequence: Creates multiple plots (e.g., ((0, 1), (1, 2), 0) or [0, 1, (0, 1)])
        loadings_components : int or sequence of int, default=[0, 1, 2]
            Which components to show in loadings plot
        variance_threshold : float, default=0.95
            Threshold line for explained variance plot
        color_by_y : bool, default=True
            Whether to color scores by y values (if available)
        annotate_by : str or dict, optional
            Annotations for score plot points. Can be:
            - 'sample_index': Annotate with sample indices (0, 1, 2, ...)
            - 'y': Annotate with y values (only for single dataset)
            - dict: Dictionary mapping dataset names to annotation arrays
              e.g., {'train': ['A', 'B', 'C'], 'test': ['D', 'E']}
            If None (default), no annotations are added.
        scores_figsize : tuple of float, default=(6, 6)
            Figure size for each scores plot (width, height) in inches
        loadings_figsize : tuple of float, default=(10, 5)
            Figure size for loadings plot (width, height) in inches
        variance_figsize : tuple of float, default=(10, 5)
            Figure size for variance plot (width, height) in inches
        spectra_figsize : tuple of float, default=(12, 5)
            Figure size for spectra plots (width, height) in inches
        distances_figsize : tuple of float, default=(8, 6)
            Figure size for distances plot (width, height) in inches

        Returns
        -------
        figures : dict
            Dictionary containing all created figures with keys:
            - 'scores_1', 'scores_2', ...: One or more scores plots (with 95% confidence ellipses)
            - 'loadings': Loadings plot
            - 'variance': Explained variance plot
            - 'distances': Diagnostic distances plot (Hotelling's T² vs Q residuals)
            - 'raw_spectra', 'preprocessed_spectra': Spectra plots (if preprocessing exists)

            For multiple datasets, scores, distances, and spectra plots show all datasets together
            on the same figure, colored by dataset. Number of 'scores_N' entries depends on
            components_scores parameter

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
        figures = {}

        datasets = normalize_datasets(dataset)
        use_suffix = len(datasets) > 1

        variance_fig = self.create_latent_variance_figure(
            variance_threshold=variance_threshold,
            figsize=variance_figsize,
        )
        if variance_fig is not None:
            figures["variance"] = variance_fig

        xlabel = get_xlabel_for_features(self.feature_names is not None)
        figures["loadings"] = self.create_latent_loadings_figure(
            loadings_components=loadings_components,
            xlabel=xlabel,
            figsize=loadings_figsize,
        )

        scores_figures = self.create_latent_scores_figures(
            dataset=dataset,
            components=components_scores,
            color_by_y=color_by_y,
            annotate_by=annotate_by,
            figsize=scores_figsize,
        )
        figures.update(scores_figures)

        figures["distances"] = self.create_latent_distance_figure(
            dataset=dataset,
            color_by_y=color_by_y,
            figsize=distances_figsize,
        )

        # Add spectra plots if preprocessing exists
        # Note: We call inspect_spectra once with all datasets to plot them together
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
        Creates two separate figure windows: one for raw spectra and one
        for preprocessed spectra. When multiple datasets are provided,
        all spectra are plotted on the same figure with colors indicating
        the dataset.

        Parameters
        ----------
        dataset : Union[str, Sequence[str]], default='train'
            Dataset(s) to visualize. Can be a single dataset name or a sequence
            of dataset names (e.g., ["train", "test"]).
        color_by_y : bool, default=True
            Whether to color by y values (if available). Only used for single dataset.
            Ignored when multiple datasets are provided (colors by dataset instead).
        xlim : tuple of float, optional
            X-axis limits for zooming into spectral regions
        figsize : tuple of float, default=(12, 5)
            Figure size for each plot (width, height) in inches

        Returns
        -------
        figures : dict
            Dictionary containing both figures with keys:
            'raw_spectra', 'preprocessed_spectra'

        Raises
        ------
        ValueError
            If no preprocessing pipeline is available

        Examples
        --------
        >>> inspector = PCAInspector(pipeline, X_train, y_train)
        >>> # Single dataset
        >>> figs = inspector.inspect_spectra()  # Creates 2 separate plots
        >>> figs = inspector.inspect_spectra(xlim=(1000, 1800))  # Zoom in
        >>> # Multiple datasets comparison
        >>> inspector.X_test = X_test
        >>> figs = inspector.inspect_spectra(dataset=["train", "test"])
        >>> figs['raw_spectra'].savefig('raw_spectra_comparison.png')
        """
        if self.transformer is None:
            raise ValueError(
                "Spectra inspection requires a preprocessing pipeline. "
                "Model must be a Pipeline with preprocessing steps."
            )

        figures = {}

        # Normalize dataset to always be a list
        datasets = normalize_datasets(dataset)
        is_multi_dataset = len(datasets) > 1

        # Determine xlabel based on wavenumbers
        xlabel = get_xlabel_for_features(self.feature_names is not None)

        # Get preprocessed wavenumbers (may be subset if feature selection)
        preprocessed_wavenumbers = self._get_preprocessed_wavenumbers()

        if is_multi_dataset:
            # Multiple datasets: plot all on same figure, color by dataset
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
