"""Benchmark execution engine for registry-defined estimators."""

from __future__ import annotations

import gc
import importlib
import json
import sys
import time
import tracemalloc
import types
from pathlib import Path
from typing import Any

import numpy as np

from .spec import (
    EstimatorSpec,
    RegistrySpec,
    expand_constructor_variants,
    make_run_id,
    should_skip,
)


def _resolve_class(class_path: str) -> type[Any]:
    module_name, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def _coerce_dtype(dtype_name: str) -> np.dtype[Any]:
    try:
        return np.dtype(dtype_name)
    except TypeError as exc:
        raise ValueError(f"Unsupported dtype: {dtype_name}") from exc


def _resolve_mapping(
    mapping: dict[str, Any], context: dict[str, Any]
) -> dict[str, Any]:
    resolved: dict[str, Any] = {}
    for key, value in mapping.items():
        if isinstance(value, str) and value in context:
            resolved[key] = context[value]
        else:
            resolved[key] = value
    return resolved


def _estimate_object_size_bytes(obj: Any, seen: set[int] | None = None) -> int:
    """Estimate total in-memory size of an object graph in bytes."""
    if seen is None:
        seen = set()

    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)

    # Ignore code-like objects that can massively skew recursive scans.
    if isinstance(
        obj,
        (
            types.ModuleType,
            types.FunctionType,
            types.BuiltinFunctionType,
            types.MethodType,
            type,
        ),
    ):
        return 0

    size = sys.getsizeof(obj)

    if isinstance(obj, np.ndarray):
        return size + int(obj.nbytes)

    if isinstance(obj, dict):
        return size + sum(
            _estimate_object_size_bytes(key, seen)
            + _estimate_object_size_bytes(value, seen)
            for key, value in obj.items()
        )

    if isinstance(obj, (list, tuple, set, frozenset)):
        return size + sum(_estimate_object_size_bytes(item, seen) for item in obj)

    if hasattr(obj, "__dict__"):
        return size + _estimate_object_size_bytes(vars(obj), seen)

    if hasattr(obj, "__slots__"):
        slots = getattr(obj, "__slots__")
        if isinstance(slots, str):
            slots = (slots,)
        slot_size = 0
        for slot in slots:
            if hasattr(obj, slot):
                slot_size += _estimate_object_size_bytes(getattr(obj, slot), seen)
        return size + slot_size

    return size


class BenchmarkRunner:
    """Execute benchmark scenarios declared in the registry."""

    def __init__(self, registry: RegistrySpec) -> None:
        self.registry = registry

    def list_estimators(self) -> list[str]:
        return sorted(self.registry.estimators)

    def list_scenarios(self) -> list[str]:
        return sorted(self.registry.scenarios)

    def list_profiles(self) -> list[str]:
        return sorted(self.registry.profiles)

    def run(
        self,
        *,
        estimator_key: str,
        scenario_name: str | None,
        profile_name: str,
        constructor_overrides: dict[str, Any] | None = None,
        output_path: Path | None = None,
    ) -> dict[str, Any]:
        if estimator_key not in self.registry.estimators:
            raise KeyError(f"Unknown estimator key: {estimator_key}")

        if profile_name not in self.registry.profiles:
            raise KeyError(f"Unknown profile name: {profile_name}")

        spec = self.registry.estimators[estimator_key]
        scenario_key = scenario_name or spec.scenario_default
        if scenario_key not in self.registry.scenarios:
            raise KeyError(f"Unknown scenario name: {scenario_key}")

        scenario = self.registry.scenarios[scenario_key]
        profile = self.registry.profiles[profile_name]

        variants = expand_constructor_variants(spec)
        if constructor_overrides:
            variants = [{**params, **constructor_overrides} for params in variants]

        all_results: list[dict[str, Any]] = []
        for constructor_params in variants:
            if should_skip(constructor_params, scenario_key, spec.skip_if):
                continue

            run_id = make_run_id(
                estimator_key, scenario_key, profile_name, constructor_params
            )
            result = self._run_single(
                spec=spec,
                scenario_name=scenario_key,
                scenario=scenario,
                profile_name=profile_name,
                profile=profile,
                constructor_params=constructor_params,
                run_id=run_id,
            )
            all_results.append(result)

        if not all_results:
            raise RuntimeError(
                "No benchmark variants to run after applying skip rules."
            )

        payload = {
            "registry_version": self.registry.version,
            "estimator": estimator_key,
            "scenario": scenario_key,
            "profile": profile_name,
            "results": all_results,
        }

        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            payload["output_path"] = str(output_path)

        return payload

    def _default_output_path(
        self,
        *,
        estimator_key: str,
        profile_name: str,
        scenario_name: str,
    ) -> Path:
        safe_key = estimator_key.replace(".", "_")
        filename = f"{safe_key}_{profile_name}_{scenario_name}.json"
        return Path("benchmarks/results") / safe_key / filename

    def _build_data_context(
        self,
        *,
        spec: EstimatorSpec,
        scenario: Any,
    ) -> dict[str, Any]:
        rng = np.random.default_rng(scenario.seed)
        dtype = _coerce_dtype(scenario.dtype)

        context: dict[str, Any] = {
            "n_fit_samples": scenario.n_fit_samples,
            "n_transform_samples": scenario.n_transform_samples,
            "n_features": scenario.n_features,
            "dtype": str(dtype),
        }

        fit_adapter = spec.fit_strategy.get("adapter", "plain_fit")

        if fit_adapter == "source_target_pair":
            source_scale = float(spec.fit_strategy.get("source_scale", 1.5))
            source_noise = float(spec.fit_strategy.get("source_noise", 0.1))

            x_target = rng.normal(
                size=(scenario.n_fit_samples, scenario.n_features)
            ).astype(
                dtype,
                copy=False,
            )
            x_source = x_target * source_scale + source_noise * rng.normal(
                size=x_target.shape
            ).astype(dtype, copy=False)
            x_transform = rng.normal(
                size=(scenario.n_transform_samples, scenario.n_features)
            ).astype(dtype, copy=False)

            context.update(
                {
                    "X_target": x_target,
                    "X_source": x_source,
                    "X_fit": x_target,
                    "y_fit": rng.normal(size=scenario.n_fit_samples).astype(
                        dtype, copy=False
                    ),
                    "X_external": rng.normal(size=x_target.shape).astype(
                        dtype, copy=False
                    ),
                    "X_transform": x_transform,
                }
            )
            return context

        if fit_adapter == "interpolation_fit":
            common_points = int(
                spec.fit_strategy.get("common_points", scenario.n_features)
            )
            x_fit = rng.normal(
                size=(scenario.n_fit_samples, scenario.n_features)
            ).astype(
                dtype,
                copy=False,
            )
            x_transform = rng.normal(
                size=(scenario.n_transform_samples, scenario.n_features)
            ).astype(dtype, copy=False)
            x_axis_shared = np.linspace(
                1100.0, 2500.0, scenario.n_features, dtype=np.float64
            )
            common_x_axis = np.linspace(
                x_axis_shared[0], x_axis_shared[-1], common_points, dtype=np.float64
            )

            x_axis_mode = spec.transform_strategy.get("x_axis_mode", "shared")
            if x_axis_mode == "shared":
                x_axis_transform: np.ndarray | list[np.ndarray] = x_axis_shared
            elif x_axis_mode == "per-row":
                offsets = np.linspace(
                    -0.15,
                    0.15,
                    scenario.n_transform_samples,
                    dtype=np.float64,
                )
                x_axis_transform = x_axis_shared[None, :] + offsets[:, None]
            else:
                raise ValueError(f"Unknown x_axis_mode: {x_axis_mode}")

            context.update(
                {
                    "X_fit": x_fit,
                    "y_fit": rng.normal(size=scenario.n_fit_samples).astype(
                        dtype, copy=False
                    ),
                    "X_external": rng.normal(size=x_fit.shape).astype(
                        dtype, copy=False
                    ),
                    "X_transform": x_transform,
                    "common_x_axis": common_x_axis,
                    "x_axis_transform": x_axis_transform,
                }
            )
            return context

        x_fit = rng.normal(size=(scenario.n_fit_samples, scenario.n_features)).astype(
            dtype,
            copy=False,
        )
        x_transform = rng.normal(
            size=(scenario.n_transform_samples, scenario.n_features)
        ).astype(
            dtype,
            copy=False,
        )
        y_fit = rng.normal(size=scenario.n_fit_samples).astype(dtype, copy=False)
        x_external = rng.normal(size=x_fit.shape).astype(dtype, copy=False)
        context.update(
            {
                "X_fit": x_fit,
                "y_fit": y_fit,
                "X_external": x_external,
                "X_transform": x_transform,
            }
        )
        return context

    def _run_single(
        self,
        *,
        spec: EstimatorSpec,
        scenario_name: str,
        scenario: Any,
        profile_name: str,
        profile: Any,
        constructor_params: dict[str, Any],
        run_id: str,
    ) -> dict[str, Any]:
        context = self._build_data_context(spec=spec, scenario=scenario)

        cls = _resolve_class(spec.class_path)
        init_params = dict(constructor_params)

        # Strategy-provided constructor values override defaults when needed.
        if "common_x_axis" in context and "common_x_axis" not in init_params:
            init_params["common_x_axis"] = context["common_x_axis"]

        estimator = cls(**init_params)

        fit_input_name = spec.fit_strategy.get("fit_input", "X_fit")
        x_fit = context[fit_input_name]
        fit_kwargs = _resolve_mapping(spec.fit_strategy.get("fit_kwargs", {}), context)
        estimator.fit(x_fit, **fit_kwargs)
        post_fit_estimator_bytes = _estimate_object_size_bytes(estimator)

        transform_input_name = spec.transform_strategy.get(
            "transform_input", "X_transform"
        )
        x_transform = context[transform_input_name]
        transform_kwargs = _resolve_mapping(
            spec.transform_strategy.get("transform_kwargs", {}),
            context,
        )

        runs = int(spec.profile_defaults.get("calls", profile.runs))
        warmups = int(profile.warmups)

        for _ in range(warmups):
            estimator.transform(x_transform, **transform_kwargs)

        gc_was_enabled = gc.isenabled()
        gc.disable()
        times_ms: list[float] = []

        try:
            for _ in range(runs):
                t0 = time.perf_counter_ns()
                estimator.transform(x_transform, **transform_kwargs)
                t1 = time.perf_counter_ns()
                times_ms.append((t1 - t0) / 1_000_000.0)
        finally:
            if gc_was_enabled:
                gc.enable()

        arr = np.asarray(times_ms, dtype=np.float64)
        mean_ms = float(arr.mean())
        std_ms = float(arr.std(ddof=1)) if runs > 1 else 0.0

        peak_bytes = None
        if profile.measure_peak_memory:
            tracemalloc.start()
            estimator.transform(x_transform, **transform_kwargs)
            _, peak_bytes = tracemalloc.get_traced_memory()
            tracemalloc.stop()

        return {
            "run_id": run_id,
            "estimator_key": spec.key,
            "scenario": scenario_name,
            "profile": profile_name,
            "constructor_params": constructor_params,
            "fit_strategy": spec.fit_strategy,
            "transform_strategy": spec.transform_strategy,
            "runs": runs,
            "warmups": warmups,
            "n_fit_samples": scenario.n_fit_samples,
            "n_transform_samples": scenario.n_transform_samples,
            "n_features": scenario.n_features,
            "mean_ms": mean_ms,
            "median_ms": float(np.median(arr)),
            "std_ms": std_ms,
            "min_ms": float(arr.min()),
            "max_ms": float(arr.max()),
            "p05_ms": float(np.percentile(arr, 5)),
            "p95_ms": float(np.percentile(arr, 95)),
            "p99_ms": float(np.percentile(arr, 99)),
            "iqr_ms": float(np.percentile(arr, 75) - np.percentile(arr, 25)),
            "cv_percent": float((std_ms / mean_ms) * 100.0) if mean_ms > 0 else 0.0,
            "post_fit_estimator_bytes": post_fit_estimator_bytes,
            "peak_memory_bytes": peak_bytes,
        }
