"""Preprocessing Inspector for pipeline step visualization."""

from __future__ import annotations

import warnings
from typing import (
    TYPE_CHECKING,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
from sklearn.base import is_classifier, is_regressor
from sklearn.cross_decomposition._pls import _PLS
from sklearn.decomposition._base import _BasePCA
from sklearn.pipeline import Pipeline
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

from .core.base import InspectorDataset, _DataHoldingBase
from .core.spectra import SpectraMixin
from .core.summaries import PreprocessingSummary
from .core.utils import (
    get_xlabel_for_features,
    normalize_datasets,
    prepare_color_values,
)
from .helpers._preprocessing import create_preprocessing_step_plot

if TYPE_CHECKING:
    from matplotlib.figure import Figure


def _is_model_step(step: object) -> bool:
    """Return ``True`` if *step* is a model/estimator that should be excluded.

    Steps excluded from preprocessing visualization:
    - PCA and other decomposition models (``_BasePCA``)
    - PLS and related cross-decomposition models (``_PLS``)
    - All estimators from ``sklearn.decomposition`` (NMF, FastICA, etc.)
    - All estimators from ``sklearn.cross_decomposition``
    - Classifiers (``is_classifier``)
    - Regressors (``is_regressor``)
    - The string ``"passthrough"`` (no-op placeholder)
    """
    if isinstance(step, str):
        # Handles "passthrough" and any future string sentinels
        return True

    if isinstance(step, (_BasePCA, _PLS)):
        return True

    if is_classifier(step) or is_regressor(step):
        return True

    # Catch decomposition models not derived from _BasePCA (NMF, FastICA, …)
    module = type(step).__module__ or ""
    if module.startswith("sklearn.decomposition") or module.startswith(
        "sklearn.cross_decomposition"
    ):
        return True

    return False


class PreprocessingInspector(SpectraMixin, _DataHoldingBase):
    """Inspector for visualizing the effects of each preprocessing step in a pipeline.

    The ``PreprocessingInspector`` takes a **fitted** scikit-learn
    :class:`~sklearn.pipeline.Pipeline` together with the datasets that were
    used for training (and, optionally, testing/validation).  It walks through
    the pipeline steps, applies each preprocessing transformer cumulatively, and
    generates one plot per step so that users can visually inspect how each
    transformation modifies their data.

    Steps that are *model* estimators — such as PCA, PLS, classifiers, or
    regressors — are automatically detected and **excluded** from the
    visualization, because they do not represent a preprocessing
    transformation.

    The class also inherits :class:`SpectraMixin`, which provides the
    :meth:`inspect_spectra` method for a quick *raw vs. fully preprocessed*
    comparison.

    Parameters
    ----------
    pipeline : Pipeline
        A **fitted** scikit-learn ``Pipeline``.  All steps must already be
        fitted (i.e. ``pipeline.fit(X)`` has been called).
    X_train : array-like of shape (n_samples, n_features)
        Training feature matrix (required).
    y_train : array-like of shape (n_samples,), optional
        Training target values.  Used for colouring plots when
        ``color_by='y'``.
    X_test : array-like of shape (n_samples, n_features), optional
        Test feature matrix.
    y_test : array-like of shape (n_samples,), optional
        Test target values.
    X_val : array-like of shape (n_samples, n_features), optional
        Validation feature matrix.
    y_val : array-like of shape (n_samples,), optional
        Validation target values.
    x_axis : array-like of shape (n_features,), optional
        Feature names or axis values (e.g. wavenumbers).  If ``None``,
        integer indices are used.

    Attributes
    ----------
    pipeline : Pipeline
        The original fitted pipeline.
    model : Pipeline
        Alias for ``pipeline`` (consistent with ``PCAInspector`` /
        ``PLSRegressionInspector``).
    preprocessing_steps : list of tuple
        ``(name, transformer)`` pairs for every step that will be visualised
        (model steps are excluded).
    datasets_ : dict of str to InspectorDataset
        Dictionary of loaded datasets keyed by ``'train'``, ``'test'``,
        ``'val'``.
    n_features_in_ : int
        Number of input features.

    Raises
    ------
    TypeError
        If *pipeline* is not a :class:`~sklearn.pipeline.Pipeline`.
    RuntimeError
        If the pipeline has not been fitted.
    ValueError
        If ``X_train`` has inconsistent shape with other datasets.

    Examples
    --------
    >>> from sklearn.pipeline import make_pipeline
    >>> from sklearn.preprocessing import StandardScaler, MinMaxScaler
    >>> from sklearn.decomposition import PCA
    >>> from chemotools.inspector import PreprocessingInspector
    >>>
    >>> pipe = make_pipeline(StandardScaler(), MinMaxScaler(), PCA(n_components=3))
    >>> pipe.fit(X_train)
    >>>
    >>> inspector = PreprocessingInspector(pipe, X_train, y_train)
    >>> figures = inspector.inspect()          # one plot per preprocessing step
    >>> figures = inspector.inspect_spectra()  # raw vs. fully preprocessed
    """

    def __init__(
        self,
        pipeline: Pipeline,
        X_train: np.ndarray,
        y_train: Optional[np.ndarray] = None,
        X_test: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        x_axis: Optional[np.ndarray] = None,
    ) -> None:
        self._validate_pipeline(pipeline)
        self._pipeline = pipeline

        # --- Identify preprocessing vs. model steps ----------------------------
        self._preprocessing_steps: List[Tuple[str, object]] = [
            (name, step) for name, step in pipeline.steps if not _is_model_step(step)
        ]

        if not self._preprocessing_steps:
            raise ValueError(
                "The pipeline does not contain any preprocessing steps to "
                "visualise.  All steps were identified as model/estimator steps."
            )

        # Warn about nested pipelines / ColumnTransformer — not fully supported
        from sklearn.compose import ColumnTransformer

        for name, step in self._preprocessing_steps:
            if isinstance(step, (Pipeline, ColumnTransformer)):
                warnings.warn(
                    f"Step '{name}' is a {type(step).__name__}, which is treated "
                    f"as a single opaque step. Individual sub-steps inside it "
                    f"will not be visualised separately.",
                    UserWarning,
                    stacklevel=2,
                )

        # --- Validate and build datasets ----------------------------------------
        X_train = check_array(
            X_train,
            dtype="numeric",
            ensure_2d=True,
            ensure_all_finite=True,
            input_name="X_train",
        )

        datasets: Dict[str, InspectorDataset] = {
            "train": InspectorDataset(
                X=X_train,
                y=self._validate_y(y_train, X_train.shape[0], "y_train"),
            ),
        }
        self._add_optional_dataset(datasets, "test", X_test, y_test, X_train.shape[1])
        self._add_optional_dataset(datasets, "val", X_val, y_val, X_train.shape[1])

        # --- Initialise data-holding base ----------------------------------------
        super().__init__(
            datasets=datasets,
            n_features_in=X_train.shape[1],
            feature_names=x_axis,
        )

    # ------------------------------------------------------------------
    # Static / private helpers for __init__
    # ------------------------------------------------------------------
    @staticmethod
    def _validate_pipeline(pipeline: object) -> None:
        """Validate that *pipeline* is a fitted sklearn Pipeline."""
        if not isinstance(pipeline, Pipeline):
            raise TypeError(
                f"Expected a sklearn Pipeline, got {type(pipeline).__name__}."
            )
        try:
            check_is_fitted(pipeline)
        except Exception as exc:
            raise RuntimeError(
                "The pipeline must be fitted before passing it to "
                "PreprocessingInspector."
            ) from exc

    @staticmethod
    def _add_optional_dataset(
        datasets: Dict[str, InspectorDataset],
        name: str,
        X: Optional[np.ndarray],
        y: Optional[np.ndarray],
        expected_features: int,
    ) -> None:
        """Validate and store an optional (test / val) dataset."""
        if X is None:
            return
        X = check_array(
            X,
            dtype="numeric",
            ensure_2d=True,
            ensure_all_finite=True,
            input_name=f"X_{name}",
        )
        if X.shape[1] != expected_features:
            raise ValueError(
                f"X_{name} must have the same number of features as X_train. "
                f"Got {X.shape[1]} vs {expected_features}."
            )
        datasets[name] = InspectorDataset(
            X=X,
            y=PreprocessingInspector._validate_y(y, X.shape[0], f"y_{name}"),
        )

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _validate_y(
        y: Optional[np.ndarray], expected_n: int, name: str
    ) -> Optional[np.ndarray]:
        """Validate and normalise a target array."""
        if y is None:
            return None
        arr = check_array(
            y,
            dtype=None,
            ensure_2d=False,
            ensure_all_finite=True,
            input_name=name,
        )
        if arr.ndim == 2 and arr.shape[1] == 1:
            arr = arr.ravel()
        if arr.shape[0] != expected_n:
            raise ValueError(
                f"{name} must have {expected_n} samples, got {arr.shape[0]}."
            )
        return arr

    # ------------------------------------------------------------------
    # SpectraMixin protocol implementation
    # ------------------------------------------------------------------
    @property
    def transformer(self) -> Optional[Pipeline]:
        """Return a pipeline containing only the preprocessing steps.

        This is used by :class:`SpectraMixin` to generate *raw vs.
        preprocessed* comparison plots via :meth:`inspect_spectra`.
        """
        if not self._preprocessing_steps:
            return None
        return Pipeline(list(self._preprocessing_steps))

    def _get_preprocessed_data(self, dataset: str) -> np.ndarray:
        """Return fully preprocessed X (through all preprocessing steps)."""
        if dataset in self._preprocessed_cache:
            return self._preprocessed_cache[dataset]

        X = self._get_dataset(dataset).X
        transformer = self.transformer
        if transformer is None:
            result = X
        else:
            result = transformer.transform(X)

        self._preprocessed_cache[dataset] = result
        return result

    def _get_preprocessed_x_axis(self) -> np.ndarray:
        """Return x-axis values after full preprocessing.

        If preprocessing changes the number of features (e.g. feature
        selection), the returned array will reflect the new dimensionality.
        """
        X_prep = self._get_preprocessed_data("train")
        return self._resolve_x_axis_after_transform(
            self._preprocessing_steps, X_prep.shape[1]
        )

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------
    @property
    def pipeline(self) -> Pipeline:
        """Return the original pipeline."""
        return self._pipeline

    @property
    def model(self) -> Pipeline:
        """Return the original pipeline.

        Alias for :attr:`pipeline`, provided for consistency with
        ``PCAInspector`` and ``PLSRegressionInspector``.
        """
        return self._pipeline

    @property
    def preprocessing_steps(self) -> List[Tuple[str, object]]:
        """Return the list of ``(name, transformer)`` preprocessing steps."""
        return list(self._preprocessing_steps)

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------
    def __repr__(self) -> str:  # noqa: D105
        datasets = ", ".join(
            f"{name}({ds.n_samples})" for name, ds in self.datasets_.items()
        )
        return (
            f"PreprocessingInspector("
            f"steps={len(self._preprocessing_steps)}, "
            f"features={self.n_features_in_}, "
            f"datasets=[{datasets}])"
        )

    # ------------------------------------------------------------------
    # inspect() — step-by-step preprocessing visualization
    # ------------------------------------------------------------------
    def inspect(
        self,
        dataset: Union[str, Sequence[str]] = "train",
        color_by: Optional[Union[str, Dict[str, np.ndarray]]] = "y",
        xlim: Optional[Tuple[float, float]] = None,
        figsize: Tuple[float, float] = (12, 5),
        color_mode: Optional[Literal["continuous", "categorical"]] = None,
    ) -> Dict[str, "Figure"]:
        """Generate one plot per preprocessing step showing cumulative effects.

        For a pipeline with steps ``[A, B, C, PCA]`` (where PCA is excluded),
        this method produces:

        1. **Raw** – the original input data
        2. **After A** – ``A.transform(X)``
        3. **After A + B** – ``B.transform(A.transform(X))``
        4. **After A + B + C** – ``C.transform(B.transform(A.transform(X)))``

        Parameters
        ----------
        dataset : str or sequence of str, default='train'
            Dataset(s) to visualise.  When a sequence is given, all datasets
            are overlaid on the same axes, coloured by dataset name.
        color_by : str or dict, default='y'
            Colouring specification (single-dataset mode only):

            - ``'y'``: colour by target values (if available)
            - ``'sample_index'``: colour by sample index
            - dict mapping dataset names to colour arrays

            Ignored when multiple datasets are provided (colours by dataset
            instead).
        xlim : tuple of float, optional
            X-axis limits for zooming into a spectral region.
        figsize : tuple of float, default=(12, 5)
            Figure size ``(width, height)`` in inches for each subplot.
        color_mode : {``'continuous'``, ``'categorical'``}, optional
            Override automatic colour-mode detection.

        Returns
        -------
        figures : dict of str to Figure
            Dictionary mapping step names to matplotlib ``Figure`` objects.
            Keys follow the pattern ``'raw'``, ``'step_1_<name>'``,
            ``'step_2_<name>'``, etc.

        Examples
        --------
        >>> inspector = PreprocessingInspector(pipeline, X_train, y_train)
        >>> figures = inspector.inspect()
        >>> figures['raw'].savefig('raw_spectra.png')
        >>> figures['step_1_standardscaler'].savefig('after_scaling.png')
        """
        self.close_figures()

        datasets = normalize_datasets(dataset)
        is_multi = len(datasets) > 1
        xlabel = get_xlabel_for_features(self.feature_names is not None)

        figures: Dict[str, "Figure"] = {}

        # --- Raw data plot -----------------------------------------------------
        if is_multi:
            figures["raw"] = self._plot_multi_dataset_step(
                datasets,
                step_data=None,
                title="Raw Spectra",
                xlabel=xlabel,
                xlim=xlim,
                figsize=figsize,
                color_mode=color_mode,
            )
        else:
            ds_name = datasets[0]
            ds = self._get_dataset(ds_name)
            color_values = prepare_color_values(color_by, ds_name, ds.y, ds.X.shape[0])
            figures["raw"] = create_preprocessing_step_plot(
                X=ds.X,
                x_axis=self._x_axis,
                title=f"Raw Spectra ({ds_name.capitalize()})",
                xlabel=xlabel,
                color_values=color_values,
                xlim=xlim,
                figsize=figsize,
                color_mode=color_mode,
            )

        # --- One plot per preprocessing step (iterative: O(N) transforms) -----
        # Keep track of the cumulative transformed X for each dataset so that
        # each step only applies its own transform on the previous output,
        # avoiding the O(N²) cost of rebuilding sub-pipelines.
        cumulative: Dict[str, np.ndarray] = {
            ds_name: self._get_dataset(ds_name).X for ds_name in datasets
        }

        for step_idx, (step_name, _step_transformer) in enumerate(
            self._preprocessing_steps, start=1
        ):
            # Apply only this step's transform to the previous cumulative output
            for ds_name in datasets:
                cumulative[ds_name] = _step_transformer.transform(cumulative[ds_name])  # type: ignore[ty:unresolved-attribute]

            # Cumulative step label for the title/key
            latest_step_type = type(_step_transformer).__name__
            fig_key = f"step_{step_idx}_{step_name}"
            title = f"Step {step_idx}: after {latest_step_type}"

            steps_so_far = self._preprocessing_steps[:step_idx]

            if is_multi:
                step_x_axis = self._resolve_x_axis_after_transform(
                    steps_so_far, cumulative[datasets[0]].shape[1]
                )

                figures[fig_key] = self._plot_multi_dataset_step(
                    datasets,
                    step_data=dict(cumulative),
                    title=title,
                    xlabel=xlabel,
                    xlim=xlim,
                    figsize=figsize,
                    color_mode=color_mode,
                    step_x_axis=step_x_axis,
                )
            else:
                ds_name = datasets[0]
                ds = self._get_dataset(ds_name)
                step_x_axis = self._resolve_x_axis_after_transform(
                    steps_so_far, cumulative[ds_name].shape[1]
                )

                color_values = prepare_color_values(
                    color_by, ds_name, ds.y, ds.X.shape[0]
                )

                figures[fig_key] = create_preprocessing_step_plot(
                    X=cumulative[ds_name],
                    x_axis=step_x_axis,
                    title=f"{title} ({ds_name.capitalize()})",
                    xlabel=xlabel,
                    color_values=color_values,
                    xlim=xlim,
                    figsize=figsize,
                    color_mode=color_mode,
                )

        return self._track_figures(figures)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    def summary(self) -> PreprocessingSummary:
        """Return a summary of the pipeline and preprocessing steps.

        Returns
        -------
        summary : PreprocessingSummary
            Typed summary dataclass.  Printing the returned object produces a
            human-readable table (via ``__repr__``).
        """
        steps_info = [
            {
                "step": i,
                "name": name,
                "type": type(transformer).__name__,
            }
            for i, (name, transformer) in enumerate(self._preprocessing_steps, start=1)
        ]

        excluded = [
            {
                "name": name,
                "type": type(transformer).__name__,
            }
            for name, transformer in self._pipeline.steps
            if _is_model_step(transformer)
        ]

        return PreprocessingSummary(
            pipeline_type=type(self._pipeline).__name__,
            total_steps=len(self._pipeline.steps),
            n_preprocessing_steps=len(self._preprocessing_steps),
            n_excluded_steps=len(excluded),
            n_features=self.n_features_in_,
            n_samples=self.n_samples,
            steps=steps_info,
            excluded=excluded,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _plot_multi_dataset_step(
        self,
        datasets: List[str],
        step_data: Optional[Dict[str, np.ndarray]],
        title: str,
        xlabel: str,
        xlim: Optional[Tuple[float, float]],
        figsize: Tuple[float, float],
        color_mode: Optional[Literal["continuous", "categorical"]],
        step_x_axis: Optional[np.ndarray] = None,
    ) -> "Figure":
        """Create a single figure with multiple datasets overlaid."""
        import matplotlib.pyplot as plt

        from chemotools.plotting import SpectraPlot
        from chemotools.plotting._styles import DATASET_COLORS

        fig, ax = plt.subplots(figsize=figsize)

        for ds_name in datasets:
            if step_data is not None:
                X = step_data[ds_name]
                x_ax = step_x_axis if step_x_axis is not None else np.arange(X.shape[1])
            else:
                X = self._get_dataset(ds_name).X
                x_ax = self._x_axis

            color = DATASET_COLORS.get(ds_name, "black")
            labels: List[Optional[str]] = [ds_name.capitalize()] + [None] * (
                X.shape[0] - 1
            )

            plot = SpectraPlot(x=x_ax, y=X, labels=labels, color_mode=color_mode)
            plot.render(ax=ax, color=color, alpha=0.6, linewidth=1)

        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel("Intensity", fontsize=12)
        ax.grid(alpha=0.3)
        if xlim:
            ax.set_xlim(xlim)
        ax.legend()

        return fig
