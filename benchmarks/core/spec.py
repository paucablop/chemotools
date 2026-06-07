"""Typed registry loading and parameter expansion for benchmarks."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ScenarioSpec:
    name: str
    n_fit_samples: int
    n_transform_samples: int
    n_features: int
    dtype: str
    seed: int


@dataclass(frozen=True)
class ProfileSpec:
    name: str
    runs: int
    warmups: int
    measure_peak_memory: bool


@dataclass(frozen=True)
class EstimatorSpec:
    key: str
    group: str
    class_path: str
    constructor_defaults: dict[str, Any]
    constructor_grid: dict[str, list[Any]]
    fit_strategy: dict[str, Any]
    transform_strategy: dict[str, Any]
    scenario_default: str
    profile_defaults: dict[str, Any]
    compare_policy: dict[str, Any]
    skip_if: list[dict[str, Any]]


@dataclass(frozen=True)
class RegistrySpec:
    version: int
    groups: dict[str, dict[str, Any]]
    scenarios: dict[str, ScenarioSpec]
    profiles: dict[str, ProfileSpec]
    estimators: dict[str, EstimatorSpec]


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        msg = (
            "Loading YAML registry requires PyYAML. Install it with: pip install pyyaml"
        )
        raise RuntimeError(msg) from exc

    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid registry format in {path}")
    return data


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid registry format in {path}")
    return data


def _load_registry_document(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix in {".yml", ".yaml"}:
        return _load_yaml(path)
    if suffix == ".json":
        return _load_json(path)
    raise ValueError(f"Unsupported registry format: {path}")


def _merge_section(
    merged: dict[str, Any],
    incoming: dict[str, Any],
    section: str,
    source: Path,
) -> None:
    section_data = incoming.get(section, {})
    if not isinstance(section_data, dict):
        raise ValueError(f"Section '{section}' must be a mapping in {source}")

    for key, value in section_data.items():
        if key in merged[section]:
            raise ValueError(
                f"Duplicate key '{key}' in section '{section}' from {source}"
            )
        merged[section][key] = value


def _merge_registries(registries: list[tuple[Path, dict[str, Any]]]) -> dict[str, Any]:
    if not registries:
        raise ValueError("No registry documents found")

    merged: dict[str, Any] = {
        "groups": {},
        "scenarios": {},
        "profiles": {},
        "estimators": [],
    }

    version: int | None = None
    estimator_keys: set[str] = set()

    for source, raw in registries:
        incoming_version = raw.get("version")
        if incoming_version is not None:
            current = int(incoming_version)
            if version is None:
                version = current
            elif current != version:
                raise ValueError(
                    f"Mismatched registry version in {source}: "
                    f"expected {version}, got {current}"
                )

        _merge_section(merged, raw, "groups", source)
        _merge_section(merged, raw, "scenarios", source)
        _merge_section(merged, raw, "profiles", source)

        estimators = raw.get("estimators", [])
        if not isinstance(estimators, list):
            raise ValueError(f"Section 'estimators' must be a list in {source}")

        for estimator in estimators:
            if not isinstance(estimator, dict):
                raise ValueError(f"Estimator entries must be mappings in {source}")
            key = estimator.get("key")
            if not isinstance(key, str):
                raise ValueError(f"Estimator key must be a string in {source}")
            if key in estimator_keys:
                raise ValueError(f"Duplicate estimator key '{key}' in {source}")
            estimator_keys.add(key)
            merged["estimators"].append(estimator)

    merged["version"] = 1 if version is None else version
    return merged


def load_registry(path: str | Path) -> RegistrySpec:
    """Load the benchmark registry from YAML/JSON file or a directory."""
    registry_path = Path(path)

    if registry_path.is_dir():
        files = sorted(
            [
                child
                for child in registry_path.iterdir()
                if child.is_file()
                and child.suffix.lower() in {".yml", ".yaml", ".json"}
            ]
        )
        raw = _merge_registries(
            [(file, _load_registry_document(file)) for file in files]
        )
    else:
        raw = _load_registry_document(registry_path)

    scenario_specs: dict[str, ScenarioSpec] = {}
    for scenario_name, scenario in raw.get("scenarios", {}).items():
        scenario_specs[scenario_name] = ScenarioSpec(name=scenario_name, **scenario)

    profile_specs: dict[str, ProfileSpec] = {}
    for profile_name, profile in raw.get("profiles", {}).items():
        profile_specs[profile_name] = ProfileSpec(name=profile_name, **profile)

    estimator_specs: dict[str, EstimatorSpec] = {}
    for estimator in raw.get("estimators", []):
        spec = EstimatorSpec(
            key=estimator["key"],
            group=estimator["group"],
            class_path=estimator["class_path"],
            constructor_defaults=estimator.get("constructor_defaults", {}),
            constructor_grid=estimator.get("constructor_grid", {}),
            fit_strategy=estimator.get("fit_strategy", {}),
            transform_strategy=estimator.get("transform_strategy", {}),
            scenario_default=estimator["scenario_default"],
            profile_defaults=estimator.get("profile_defaults", {}),
            compare_policy=estimator.get("compare_policy", {}),
            skip_if=estimator.get("skip_if", []),
        )
        estimator_specs[spec.key] = spec

    return RegistrySpec(
        version=int(raw.get("version", 1)),
        groups=raw.get("groups", {}),
        scenarios=scenario_specs,
        profiles=profile_specs,
        estimators=estimator_specs,
    )


def expand_constructor_variants(spec: EstimatorSpec) -> list[dict[str, Any]]:
    """Expand constructor variants from defaults + constructor_grid."""
    if not spec.constructor_grid:
        return [dict(spec.constructor_defaults)]

    grid_keys = sorted(spec.constructor_grid)
    grid_values = [spec.constructor_grid[key] for key in grid_keys]

    variants: list[dict[str, Any]] = []
    for values in product(*grid_values):
        params = dict(spec.constructor_defaults)
        params.update(dict(zip(grid_keys, values, strict=True)))
        variants.append(params)

    return variants


def should_skip(
    constructor_params: dict[str, Any],
    scenario_name: str,
    skip_rules: list[dict[str, Any]],
) -> bool:
    """Return True when any skip rule matches."""
    for rule in skip_rules:
        scenario = rule.get("scenario")
        if scenario is not None and scenario != scenario_name:
            continue

        required = rule.get("when", {})
        if all(constructor_params.get(key) == value for key, value in required.items()):
            return True

    return False


def make_run_id(
    estimator_key: str,
    scenario_name: str,
    profile_name: str,
    constructor_params: dict[str, Any],
) -> str:
    """Create a deterministic identifier for a benchmark run."""
    payload = {
        "estimator": estimator_key,
        "scenario": scenario_name,
        "profile": profile_name,
        "params": constructor_params,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:10]
    return f"{estimator_key.replace('.', '_')}_{digest}"
