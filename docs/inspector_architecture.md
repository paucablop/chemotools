# Inspector Module — Critical Architecture Description

> **Scope**: `chemotools/inspector/` — ~8,500 LOC across 15 files.
> **Status**: Experimental (emits `FutureWarning` on import).

---

## 1. Module Map

```
inspector/
├── __init__.py                        # Public API surface (3 classes)
├── _pca_inspector.py                  # PCAInspector              (555 LOC)
├── _pls_regression_inspector.py       # PLSRegressionInspector    (946 LOC)
├── _preprocessing_inspector.py        # PreprocessingInspector    (611 LOC)
├── core/                              # Shared abstractions
│   ├── base.py                        # _DataHoldingBase, _BaseInspector, dataclasses   (567 LOC)
│   ├── spectra.py                     # SpectraMixin              (187 LOC)
│   ├── latent.py                      # LatentVariableMixin       (422 LOC)
│   ├── regression.py                  # RegressionMixin           (424 LOC)
│   ├── summaries.py                   # Summary dataclasses       (151 LOC)
│   ├── utils.py                       # Shared utility functions  (382 LOC)
│   └── validation.py                  # Input validation          (81 LOC)
└── helpers/                           # Pure plotting functions
    ├── _latent.py                     # Scores/loadings/variance/distance plots   (1222 LOC)
    ├── _regression.py                 # Predicted-vs-actual, residuals, Q-Q       (521 LOC)
    ├── _spectra.py                    # Raw/preprocessed spectra comparison       (239 LOC)
    └── _preprocessing.py             # Per-step preprocessing plot                (77 LOC)
```

---

## 2. Class Hierarchy

```
                        object
                          │
                ┌─────────┴──────────┐
                │                    │
         _DataHoldingBase          ABC            (from core/base.py)
          │          │               │
          │          └───────┬───────┘
          │                  │
          │           _BaseInspector              (from core/base.py)
          │                  │
          │          ┌───────┴────────┐
          │          │               │
     PreprocessingInspector    PCAInspector    PLSRegressionInspector
      (+ SpectraMixin)     (+ SpectraMixin    (+ SpectraMixin
                            + LatentMixin)     + LatentMixin
                                               + RegressionMixin)
```

### Full MRO chains

| Class | MRO |
|---|---|
| `PCAInspector` | `→ SpectraMixin → LatentVariableMixin → _BaseInspector → _DataHoldingBase → ABC → object` |
| `PLSRegressionInspector` | `→ SpectraMixin → RegressionMixin → LatentVariableMixin → _BaseInspector → _DataHoldingBase → ABC → object` |
| `PreprocessingInspector` | `→ SpectraMixin → _DataHoldingBase → object` |

---

## 3. Responsibility Map

### 3.1 `_DataHoldingBase` — Data Storage & Figure Lifecycle

The foundational layer. Owns:

| Concern | Members |
|---|---|
| **Dataset storage** | `datasets_: Dict[str, InspectorDataset]`, `n_features_in_`, `feature_names` |
| **Dataset access** | `_get_dataset()`, `_get_raw_data()`, `_iter_datasets()` |
| **X-axis / features** | `_x_axis`, `x_axis` property, `n_features`, `n_samples` |
| **Preprocessing cache** | `_preprocessed_cache: Dict[str, ndarray]` |
| **Figure lifecycle** | `_tracked_figures`, `close_figures()`, `_track_figures()`, `_cleanup_previous_figures()` |
| **Static utils** | `_normalize_target_array()`, `_prepare_labels()` |

**Design note**: This class mixes two concerns — data management and figure lifecycle management. These are orthogonal responsibilities. Extracting figure tracking into a separate `_FigureTracker` mixin would improve single-responsibility adherence, but the practical benefit is marginal given the small surface area (3 methods + 1 list).

### 3.2 `_BaseInspector(_DataHoldingBase, ABC)` — Estimator-Backed Inspectors

Extends `_DataHoldingBase` with everything needed for model-based inspectors (PCA, PLS). Owns:

| Concern | Members |
|---|---|
| **Model extraction** | `_model`, `estimator_`, `transformer_`, `model` / `estimator` / `transformer` properties |
| **Component resolution** | `n_components_`, `_resolve_n_components()` |
| **Confidence** | `_confidence`, `confidence` property |
| **Preprocessed data** | `_get_preprocessed_data()`, `_get_feature_mask()`, `_get_preprocessed_feature_names()`, `_get_preprocessed_x_axis()`, `_transform_data()` |
| **Config preparation** | `_prepare_inspection_config()` — normalises dataset/color_by/annotate_by inputs |
| **Summary** | `_base_summary()`, `_get_preprocessing_steps()` |
| **Input validation** | Full `__init__` validation pipeline: `check_array`, feature consistency, supervised mode |

**Critical coupling**: `_BaseInspector` is a *heavy consumer* of `_DataHoldingBase` — it calls `_normalize_target_array()`, `_prepare_labels()`, `_get_dataset()`, reads `feature_names`, `n_features_in_`, `datasets_`, and delegates to `super().__init__()`. Removing the inheritance would require ~10 forwarding methods with no safety gain.

### 3.3 Mixins — Protocol-Driven Composition

All mixins use the same structural pattern:

1. Define a `Protocol` (under `TYPE_CHECKING`) listing the attributes/methods they expect on `self`.
2. Provide a `_*_inspector()` method that returns `self` cast to that protocol.
3. Implement plotting/analysis methods that call through the protocol.

| Mixin | Protocol expects | Provides |
|---|---|---|
| `SpectraMixin` | `transformer`, `x_axis`, `feature_names`, `_get_raw_data()`, `_get_preprocessed_data()`, `_get_preprocessed_x_axis()` | `inspect_spectra()` |
| `LatentVariableMixin` | `model`, `confidence`, `_get_raw_data()`, `_get_preprocessed_feature_names()` + abstract hooks: `get_latent_scores()`, `get_latent_loadings()`, `get_latent_explained_variance()` | `n_components`, `hotelling_t2_limit`, `q_residuals_limit`, `create_latent_scores_figures()`, `create_latent_loadings_figure()`, `create_latent_variance_figure()`, `create_latent_distance_figure()`, `latent_summary()` |
| `RegressionMixin` | `model`, `confidence`, `datasets_`, `_get_raw_data()`, `estimator`, `_get_preprocessed_data()` | `RMSE_*`, `R2_*`, `regression_rmse()`, `regression_r2()`, `regression_bias()`, `regression_summary()`, `create_predicted_vs_actual_plot()`, `create_residuals_plot()`, `create_qq_plot()`, `create_residual_distribution_plot()` |

**Architectural consequence**: The protocols effectively define a "flat attribute contract" — any host class must place these attributes directly on `self`. This is what makes composition (wrapping `_DataHoldingBase` in a field) impractical: it would require forwarding every protocol-expected member.

### 3.4 Concrete Inspectors

| Inspector | Inherits | Unique Responsibilities |
|---|---|---|
| `PCAInspector` | `SpectraMixin + LatentVariableMixin + _BaseInspector` | PCA scores/loadings/variance access, `inspect()` orchestration (scores + loadings + variance + distances + spectra), `summary() → PCASummary` |
| `PLSRegressionInspector` | `SpectraMixin + RegressionMixin + LatentVariableMixin + _BaseInspector` | PLS scores/loadings/x-y-weights, explained X/Y variance, `inspect()` orchestration (scores + loadings + variance + distances + regression diagnostics + spectra), `summary() → PLSRegressionSummary` |
| `PreprocessingInspector` | `SpectraMixin + _DataHoldingBase` | Pipeline step-by-step visualization, `_is_model_step()` filtering, nested pipeline warnings, `inspect()` with O(N) iterative transforms, `summary() → PreprocessingSummary` |

### 3.5 Helpers — Pure Plotting Functions

The `helpers/` subpackage contains stateless functions that receive data and return `matplotlib.figure.Figure` objects. They have **no knowledge** of inspectors, datasets, or caching.

| Module | Functions | LOC |
|---|---|---|
| `_latent.py` | `create_scores_plot_single_dataset`, `create_scores_plot_multi_dataset`, `create_loadings_plot`, `create_variance_plot`, `create_model_distances_plot` | 1222 |
| `_regression.py` | `create_predicted_vs_actual_plot`, `create_y_residual_plot`, `create_qq_plot`, `create_residual_distribution_plot`, `create_regression_distances_plot` | 521 |
| `_spectra.py` | `create_spectra_plots_single_dataset`, `create_spectra_plots_multi_dataset` | 239 |
| `_preprocessing.py` | `create_preprocessing_step_plot` | 77 |

**This is the cleanest layer.** Pure functions, no side effects, easy to test independently.

---

## 4. Data Flow

```
User constructs Inspector(model, X_train, ...)
         │
         ▼
__init__ validates inputs, stores InspectorDataset instances
         │
         ▼
User calls .inspect()
         │
         ├──▶ _prepare_inspection_config()   ← normalises dataset/color_by/annotate_by
         │
         ├──▶ _get_preprocessed_data(ds)     ← runs transformer.transform(X), caches result
         │
         ├──▶ mixin.create_*_figure()        ← mixin methods build plot params
         │         │
         │         └──▶ helpers.create_*()   ← pure functions produce Figure objects
         │
         ├──▶ _track_figures(figures)         ← stores refs for close_figures()
         │
         └──▶ return Dict[str, Figure]

User calls .inspect_spectra()
         │
         └──▶ SpectraMixin.inspect_spectra()
                   │
                   ├──▶ _get_raw_data(ds)
                   ├──▶ _get_preprocessed_data(ds)
                   └──▶ helpers._spectra.create_spectra_plots_*()
```

---

## 5. Critical Assessment

### 5.1 Strengths

**Clean layering between data and plotting.** The `helpers/` functions are pure and stateless — the hardest thing to get wrong in a plotting library. The inspectors handle orchestration; the helpers handle rendering.

**Protocol-based mixins are the right abstraction.** They allow `SpectraMixin` to work with both `_BaseInspector` (which has a real `transformer`) and `PreprocessingInspector` (which synthesises one from its steps) without either knowing about the other. The protocols serve as a lightweight interface contract without requiring formal ABCs.

**`InspectorDataset` is frozen.** Immutable data containers prevent mutation bugs and make reasoning about state straightforward.

### 5.2 Tensions & Trade-offs

#### 5.2.1 `_DataHoldingBase` has two concerns

It manages both **data storage** and **figure lifecycle**. These are independent responsibilities that could be separated. However, the practical cost of merging them is low: the figure-tracking surface is 3 methods and 1 list. Splitting would add a class (and an MRO slot) for minimal benefit.

**Verdict**: Acceptable trade-off. Monitor if figure lifecycle grows in complexity.

#### 5.2.2 `_BaseInspector.__init__` is doing too much

The `__init__` method (~90 lines) validates inputs, normalises arrays, checks dataset consistency, extracts model components, builds the dataset dictionary, calls `super().__init__`, and resolves components. This is an initialisation-heavy class.

**Risk**: If a new inspector type needs different validation (e.g. an unsupervised model without `n_components`), modifying this `__init__` will be fragile.

**Mitigation path**: Extract validation into a builder/factory or a `@classmethod` alternate constructor.

#### 5.2.3 The mixin protocol contract is implicit

The mixin protocols are defined under `TYPE_CHECKING` — they exist for type-checkers but have **zero runtime weight**. If a host class fails to implement a protocol member, the error occurs only at the point of use (e.g. `AttributeError` when `inspect_spectra()` calls `self.transformer`), not at class definition time.

**Risk**: Adding a new inspector that forgets to implement `_get_preprocessed_data()` will pass instantiation and fail only when the user calls `inspect_spectra()`.

**Mitigation path**: Add a `__init_subclass__` check or use `runtime_checkable` protocols. The trade-off is startup cost vs. safety.

#### 5.2.4 `PreprocessingInspector` is a parallel hierarchy

`PreprocessingInspector` inherits `_DataHoldingBase` directly (not `_BaseInspector`). It implements its own versions of `transformer`, `_get_preprocessed_data()`, `_get_preprocessed_x_axis()`, and `model`. These shadow the same-named members in `_BaseInspector`, but with different semantics:

| Member | `_BaseInspector` | `PreprocessingInspector` |
|---|---|---|
| `transformer` | Stored `Pipeline` extracted from model | Synthesised `Pipeline` from preprocessing steps |
| `_get_preprocessed_data()` | Uses `transformer_.transform()` | Uses `self.transformer.transform()` (re-built each call) |
| `model` | Original model (PCA/Pipeline) | Original pipeline |

This works because `SpectraMixin` only depends on the protocol contract, not on which class provides it. But it means there is **no shared base** guaranteeing these methods exist — the contract is upheld by convention and the protocols.

**Verdict**: This is the most architecturally significant decision in the module. It's correct for the current requirements (PreprocessingInspector doesn't need `n_components`, `confidence`, or model extraction), but it creates a maintenance risk if the two branches need to converge.

#### 5.2.5 LOC distribution is skewed

| Layer | LOC | % |
|---|---|---|
| PLSRegressionInspector | 946 | 11% |
| helpers/_latent.py | 1222 | 14% |
| helpers/_regression.py | 521 | 6% |
| Everything else | 5866 | 69% |

`_latent.py` alone is 1222 LOC — the largest single file. It handles five distinct plot types (scores single/multi, loadings, variance, distances) with extensive styling logic. This file would benefit from being split, e.g. `_scores.py`, `_loadings.py`, `_variance.py`, `_distances.py`.

`PLSRegressionInspector` at 946 LOC is large for a class that mostly orchestrates mixin calls. The bulk comes from its `inspect()` method (~300 lines including docstring) and PLS-specific variance/weight methods. This is functional but dense.

### 5.3 Missing Pieces

| Gap | Impact | Effort |
|---|---|---|
| **No runtime protocol enforcement** | Silent failures when new inspectors miss protocol members | Low — `__init_subclass__` or `runtime_checkable` |
| **No plugin/registry pattern** | Adding a new inspector type requires editing `__init__.py` and understanding the full MRO | Medium |
| **`inspect()` methods are non-composable** | Each concrete inspector has its own `inspect()` that hardcodes which plots to create. Users can't easily mix-and-match plots from different mixins. | Medium — would require a plot-registry or config-driven approach |
| **No lazy import of matplotlib** | `matplotlib` is imported eagerly at module level (guarded by `_optional.py`). For library users who only want `PreprocessingInspector`, they still pay the import cost of the full plotting stack. | Low |
| **Figure lifecycle relies on user discipline** | `close_figures()` must be called manually or via `inspect()` auto-cleanup. No context manager or `__del__` fallback. | Low |

---

## 6. Dependency Graph

```
inspector/__init__.py
  ├── _pca_inspector.py
  │     ├── core/base.py        (_BaseInspector, InspectorPlotConfig)
  │     ├── core/spectra.py     (SpectraMixin)
  │     ├── core/latent.py      (LatentVariableMixin)
  │     ├── core/summaries.py   (PCASummary)
  │     ├── core/utils.py       (normalize helpers)
  │     └── outliers module     (HotellingT2, QResiduals)
  │
  ├── _pls_regression_inspector.py
  │     ├── core/base.py        (_BaseInspector, InspectorPlotConfig)
  │     ├── core/spectra.py     (SpectraMixin)
  │     ├── core/latent.py      (LatentVariableMixin)
  │     ├── core/regression.py  (RegressionMixin)
  │     ├── core/summaries.py   (PLSRegressionSummary)
  │     ├── core/utils.py
  │     ├── helpers/_latent.py
  │     ├── helpers/_regression.py
  │     └── outliers module     (HotellingT2, Leverage, QResiduals, StudentizedResiduals)
  │
  └── _preprocessing_inspector.py
        ├── core/base.py        (_DataHoldingBase, InspectorDataset)
        ├── core/spectra.py     (SpectraMixin)
        ├── core/summaries.py   (PreprocessingSummary)
        ├── core/utils.py
        └── helpers/_preprocessing.py

core/base.py
  ├── core/summaries.py    (InspectorSummary)
  ├── core/utils.py        (normalize_datasets)
  ├── core/validation.py   (_validate_and_extract_model, _validate_datasets_consistency)
  └── chemotools/_types.py (ModelInput)

core/spectra.py  → helpers/_spectra.py
core/latent.py   → helpers/_latent.py, outliers module, core/summaries.py
core/regression.py → helpers/_regression.py, core/summaries.py
```

---

## 7. Summary

The inspector module uses a **mixin-composition** architecture where a shared data-holding base provides storage and access patterns, protocol-driven mixins contribute domain-specific plotting abilities, and concrete inspectors orchestrate everything into user-facing `inspect()` / `summary()` methods. Plotting logic is fully separated into stateless helper functions.

The design is well-suited to the current three-inspector scope. The main architectural risks are (1) the implicit protocol contracts that aren't enforced at runtime, (2) the parallel `PreprocessingInspector` hierarchy that duplicates protocol-satisfying methods with different semantics, and (3) non-composable `inspect()` orchestrators that will accumulate complexity if more inspector types are added.
