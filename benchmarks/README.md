# Benchmark Artifacts

This folder stores benchmark and profiling artifacts by transformer.

## Layout

- `benchmarks/snv/baseline/`: baseline timing summaries (JSON)
- `benchmarks/snv/candidate/`: post-change timing summaries (JSON)
- `benchmarks/snv/profiles/`: cProfile dumps (`.perf`) for Snakeviz
- `benchmarks/snv/notes/`: hypothesis and investigation notes (Markdown)

## Unified runner (new)

Use the registry-driven runner to avoid one script per estimator.

Registry files:
- Single source of truth: `benchmarks/registry/` directory
- Shared config: `benchmarks/registry/shared.yaml`
- Domain estimators: one YAML per transformer family, including
	`benchmarks/registry/adaptation.yaml`,
	`benchmarks/registry/augmentation.yaml`,
	`benchmarks/registry/baseline.yaml`,
	`benchmarks/registry/derivative.yaml`,
	`benchmarks/registry/feature_selection.yaml`,
	`benchmarks/registry/scale.yaml`,
	`benchmarks/registry/scatter.yaml`, and
	`benchmarks/registry/smooth.yaml`

- List available entries:
	- `python -m benchmarks.benchmark list`
- Run one estimator with default scenario/profile:
	- `python -m benchmarks.benchmark run --estimator adaptation.direct_standardization`
- Run with explicit scenario/profile:
	- `python -m benchmarks.benchmark run --estimator adaptation.x_axis_interpolator --scenario standard --profile fast`
- Run all registered estimators for the standard scenario:
	- `python -m benchmarks.benchmark run-all --scenario standard --continue-on-error`
	- defaults: `--profile fast`
	- print all variant params/stats: `--variant-output all`
	- note: `adaptation.x_axis_interpolator` can take several minutes on `standard`, especially with `--profile regular`
	- memory study tip: add `--variant-output all --output-dir benchmarks/results` to capture per-variant post-fit RAM in JSON
- Override constructor parameters:
	- `python -m benchmarks.benchmark run --estimator adaptation.x_axis_interpolator --set n_jobs=-1 --set method=\"cubic\"`
- Compare baseline vs candidate using registry policy thresholds:
	- `python -m benchmarks.benchmark compare --baseline benchmarks/results/adaptation_direct_standardization/adaptation_direct_standardization_fast_tiny.json --candidate benchmarks/results/adaptation_direct_standardization/adaptation_direct_standardization_fast_tiny.json --estimator adaptation.direct_standardization`

`run` prints benchmark metrics directly in the terminal. Saving JSON is optional:
- Save explicitly with `--output benchmarks/results/<path>.json`
- Without `--output`, nothing is written to disk

Results are written to `benchmarks/results/` as JSON.
Each variant result includes `post_fit_estimator_bytes` (estimated RAM footprint right after `fit`).

## Default workflow with scripts

1. Baseline timing
- `python profile_1.py`
- writes `benchmarks/snv/baseline/snv_baseline.json`

2. Hotspot profile
- `python profile_2.py`
- writes `benchmarks/snv/profiles/snv_profile.perf`
- visualize with `snakeviz benchmarks/snv/profiles/snv_profile.perf`

3. Hypothesis note
- `python profile_3.py`
- writes `benchmarks/snv/notes/snv_hypothesis.md`

4. Candidate timing and comparison
- `python profile_1.py --output benchmarks/snv/candidate/snv_candidate.json`
- `python profile_4.py`

## Scaling to other transformers

Create the same 4 subfolders under `benchmarks/<transformer_name>/` and pass
custom paths with `--output` and `--profile`.
