# Inspector Module — Improvement Recommendations

> Generated: March 2026  
> Scope: `chemotools.inspector` — `PCAInspector`, `PLSRegressionInspector`, `PreprocessingInspector`, shared core, and helpers.

---

## 1. Architecture & Code Duplication

### 1.1 Extract a shared data-holding base from `_BaseInspector`

`_BaseInspector` currently bundles two distinct responsibilities:

- **Data management** — dataset storage, validation, feature-axis setup, preprocessed-data caching, figure tracking.
- **Estimator management** — model/estimator extraction, `n_components` resolution, `transformer_` storage.

`PreprocessingInspector` needs the first but not the second, so it duplicates all data-management code. Splitting `_BaseInspector` into a lightweight `_DataHoldingBase` (or mixin) and an `_EstimatorInspectorBase` would eliminate this duplication.

- [x] Design a `_DataHoldingBase` class that owns: `datasets_`, `n_features_in_`, `_x_axis`, `feature_names`, `_preprocessed_cache`, `_tracked_figures`, and all associated methods (`_get_dataset`, `_get_raw_data`, `_get_preprocessed_data`, `_get_preprocessed_x_axis`, `close_figures`, `_track_figures`, `_cleanup_previous_figures`, `_validate_y` / `_normalize_target_array`).
- [x] Create `_EstimatorInspectorBase(_DataHoldingBase)` that adds: estimator/transformer extraction, `n_components_` resolution, `confidence`, outlier detectors.
- [x] Refactor `PCAInspector` and `PLSRegressionInspector` to inherit from `_EstimatorInspectorBase`.
- [x] Refactor `PreprocessingInspector` to inherit from `_DataHoldingBase`.
- [x] Remove the private `_PreprocessingDataset` class; use the shared `InspectorDataset` frozen dataclass.
- [x] Verify all tests still pass after refactoring.

### 1.2 Unify dataset container types

- [x] Replace `_PreprocessingDataset` (mutable, `__slots__`) with `InspectorDataset` (frozen dataclass) from `core/base.py`.
- [x] Ensure `PreprocessingInspector` uses the same `.n_samples` property accessor pattern.

---

## 2. API Consistency

### 2.1 Summary return types

`PCAInspector.summary()` and `PLSRegressionInspector.summary()` return typed dataclasses (`PCASummary`, `PLSRegressionSummary`). `PreprocessingInspector.summary()` returns a bare `dict` **and** prints to stdout as a side effect.

- [x] Create a `PreprocessingSummary` dataclass in `core/summaries.py`.
- [x] Have `PreprocessingInspector.summary()` return a `PreprocessingSummary` instance.
- [x] Move the print-to-stdout logic to `PreprocessingSummary.__repr__()` or a separate `print_summary()` helper.
- [x] Ensure the dataclass has a `.to_dict()` method consistent with `InspectorSummary`.

### 2.2 Property naming alignment

- [x] Add a `model` property to `PreprocessingInspector` as an alias for `pipeline` (or rename `pipeline` → `model`) to match the sibling inspector API.
- [ ] Consider deprecating the `pipeline` property name with a `FutureWarning`.

### 2.3 Add `__repr__` to `PreprocessingInspector`

- [x] Implement `__repr__` showing class name, number of steps, datasets available, and feature count — similar to what `_BaseInspector` provides.

---

## 3. Correctness & Robustness

### 3.1 Harden `_is_model_step()`

The current implementation checks `isinstance(step, (_BasePCA, _PLS))` plus `is_classifier()` / `is_regressor()`. This misses unsupervised models that aren't PCA (e.g., `NMF`, `FastICA`, `KernelPCA`).

- [ ] Consider inverting the logic: a step is a **preprocessing** step if it implements `TransformerMixin` and does **not** reduce dimensionality (i.e., output features == input features), or explicitly check for known non-preprocessing base classes more broadly.
- [x] Alternatively, expand the isinstance check to include `sklearn.base.BaseEstimator` subclasses that have a `transform` but also a `fit_transform` that changes dimensionality (e.g. all of `sklearn.decomposition`).
- [x] Add unit tests for `NMF`, `FastICA`, `KernelPCA` steps to verify they are excluded.

### 3.2 Handle `"passthrough"` pipeline steps

- [x] Add a check in `_preprocessing_steps` filtering for steps that are the string `"passthrough"` — skip them.
- [x] Add a test with a pipeline containing a `"passthrough"` step.

### 3.3 Nested pipelines / ColumnTransformer

- [ ] Document (at minimum) that nested `Pipeline` or `ColumnTransformer` steps are not supported.
- [ ] Optionally raise a clear error if one is detected.
- [ ] Add a test that verifies the behavior or error message.

---

## 4. Performance

### 4.1 Cache intermediate step transforms in `inspect()`

`PreprocessingInspector.inspect()` rebuilds a sub-pipeline from scratch for each step, causing $O(N^2)$ transforms. Each step should only transform the output of the previous step.

- [x] Refactor `inspect()` to use an iterative approach: `X_current = step_i.transform(X_previous)` instead of `Pipeline(steps[:i]).transform(X_raw)`.
- [x] Verify figures remain identical after refactoring (compare pixel output or numerical values).
- [x] Add a benchmark or regression test for pipelines with many steps.

---

## 5. Test Coverage Gaps

### 5.1 `PreprocessingInspector` missing tests

- [x] Feature-selection step in pipeline (exercises `_resolve_step_x_axis` with changed dimensionality).
- [ ] `"passthrough"` step in pipeline.
- [ ] Unsupervised non-PCA models (`NMF`, `FastICA`) — verify exclusion.
- [x] `inspect_spectra(dataset=["train", "test"])` — multi-dataset spectra comparison.
- [x] Pipeline with chemotools-native transformers (e.g., `SavitzkyGolay`, `SNV`).
- [x] Large pipeline (5+ steps) — verify correct number of figures and performance.

### 5.2 Cross-inspector coverage

- [x] Test that `PCAInspector`, `PLSRegressionInspector`, and `PreprocessingInspector` can all be instantiated from the same pipeline and produce consistent results for the shared spectra-comparison functionality.
- [x] Test `inspect_spectra()` when no preprocessing exists (currently raises `ValueError` — see bug fix below).

---

## 6. Bug Fixes

### 6.1 `inspect_spectra()` should always show raw spectra

Currently, `SpectraMixin.inspect_spectra()` raises `ValueError("Spectra inspection requires a preprocessing pipeline")` when `transformer is None`. This prevents users from viewing raw spectra for standalone models.

- [x] Modify `inspect_spectra()` to **always** return a `"raw_spectra"` figure.
- [x] Only add the `"preprocessed_spectra"` figure when `transformer is not None`.
- [x] Update tests for all three inspectors.
- [x] Update docstring to reflect the new behavior.

---

## 7. Documentation

- [ ] Add a "How Inspectors Work" narrative page to the docs explaining the mixin composition (`SpectraMixin`, `LatentVariableMixin`, `RegressionMixin`) and when to use each inspector.
- [ ] Add a gallery example showing `PreprocessingInspector` usage with a real chemometrics pipeline.
- [ ] Document the relationship between `plotting` module primitives (e.g., `SpectraPlot`) and inspector helpers.

---

## Priority Matrix

| Priority | Items |
|----------|-------|
| **P0 — Bug** | 6.1 (`inspect_spectra` always show raw) |
| **P1 — API consistency** | 2.1 (summary dataclass), 2.2 (property naming), 1.2 (dataset container) |
| **P1 — Correctness** | 3.1 (harden `_is_model_step`), 3.2 (`"passthrough"`) |
| **P2 — Architecture** | 1.1 (split `_BaseInspector`) |
| **P2 — Performance** | 4.1 (cache intermediate transforms) |
| **P3 — Tests** | 5.1, 5.2 |
| **P3 — Docs** | 7.x |
