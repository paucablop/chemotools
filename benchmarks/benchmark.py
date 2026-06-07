"""Unified benchmark runner based on registry entries."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from benchmarks.core import BenchmarkRunner, load_registry
from benchmarks.core.spec import expand_constructor_variants, should_skip


def _parse_override(entry: str) -> tuple[str, Any]:
    if "=" not in entry:
        raise ValueError(f"Invalid --set value: {entry}. Expected key=value.")
    key, value = entry.split("=", 1)

    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        lowered = value.lower()
        if lowered == "true":
            parsed = True
        elif lowered == "false":
            parsed = False
        elif lowered == "null":
            parsed = None
        else:
            parsed = value

    return key, parsed


def _format_bytes(num_bytes: float | int | None) -> str:
    if num_bytes is None:
        return "n/a"

    value = float(num_bytes)
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    unit_idx = 0
    while value >= 1024.0 and unit_idx < len(units) - 1:
        value /= 1024.0
        unit_idx += 1
    return f"{value:.2f} {units[unit_idx]}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Registry-driven benchmark runner")
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("benchmarks/registry"),
        help="Path to benchmark registry file or directory",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List available registry entries")
    list_parser.add_argument(
        "--kind",
        choices=("estimators", "scenarios", "profiles", "all"),
        default="all",
        help="What to list",
    )

    run_parser = subparsers.add_parser("run", help="Run a benchmark")
    run_parser.add_argument("--estimator", required=True, help="Estimator key")
    run_parser.add_argument(
        "--scenario", help="Scenario name (defaults to estimator default)"
    )
    run_parser.add_argument("--profile", default="regular", help="Profile name")
    run_parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="Override constructor params, e.g. --set n_jobs=4",
    )
    run_parser.add_argument(
        "--output",
        type=Path,
        help="Optional output JSON path",
    )

    run_all_parser = subparsers.add_parser(
        "run-all",
        help="Run all benchmark estimators for a scenario",
    )
    run_all_parser.add_argument(
        "--scenario",
        default="standard",
        help="Scenario name to use for all estimators",
    )
    run_all_parser.add_argument(
        "--profile",
        default="fast",
        help="Profile name",
    )
    run_all_parser.add_argument(
        "--output-dir",
        type=Path,
        help="Optional directory to save one JSON result per estimator",
    )
    run_all_parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue running remaining estimators when one fails",
    )
    run_all_parser.add_argument(
        "--variant-output",
        choices=("best", "all"),
        default="best",
        help=(
            "Console output level for variants: 'best' prints summary only, "
            "'all' prints params and stats for every variant"
        ),
    )

    compare_parser = subparsers.add_parser(
        "compare", help="Compare baseline and candidate benchmark outputs"
    )
    compare_parser.add_argument("--baseline", type=Path, required=True)
    compare_parser.add_argument("--candidate", type=Path, required=True)
    compare_parser.add_argument(
        "--estimator",
        help="Estimator key (optional, inferred from files when omitted)",
    )
    compare_parser.add_argument(
        "--output",
        type=Path,
        help="Optional output path for comparison report JSON",
    )

    return parser


def cmd_list(runner: BenchmarkRunner, kind: str) -> None:
    if kind in {"estimators", "all"}:
        print("Estimators:")
        for entry in runner.list_estimators():
            print(f"  - {entry}")

    if kind in {"scenarios", "all"}:
        print("Scenarios:")
        for entry in runner.list_scenarios():
            print(f"  - {entry}")

    if kind in {"profiles", "all"}:
        print("Profiles:")
        for entry in runner.list_profiles():
            print(f"  - {entry}")


def cmd_run(args: argparse.Namespace, runner: BenchmarkRunner) -> None:
    overrides: dict[str, Any] = {}
    for override in args.overrides:
        key, value = _parse_override(override)
        overrides[key] = value

    payload = runner.run(
        estimator_key=args.estimator,
        scenario_name=args.scenario,
        profile_name=args.profile,
        constructor_overrides=overrides,
        output_path=args.output,
    )

    print("Benchmark completed")
    print(f"  Estimator: {payload['estimator']}")
    print(f"  Scenario: {payload['scenario']}")
    print(f"  Profile: {payload['profile']}")
    print(f"  Variants: {len(payload['results'])}")

    for index, result in enumerate(payload["results"], start=1):
        print(f"\nVariant {index}")
        print(f"  params: {result['constructor_params']}")
        print(
            "  median (ms): "
            f"{result['median_ms']:.6f} "
            f"(mean={result['mean_ms']:.6f}, std={result['std_ms']:.6f})"
        )
        print(f"  p95/p99 (ms): {result['p95_ms']:.6f} / {result['p99_ms']:.6f}")
        print(f"  min/max (ms): {result['min_ms']:.6f} / {result['max_ms']:.6f}")
        print(f"  cv (%): {result['cv_percent']:.2f}")
        print(
            "  post-fit RAM: "
            f"{_format_bytes(result.get('post_fit_estimator_bytes'))} "
            f"({result.get('post_fit_estimator_bytes', 0)} bytes)"
        )

    output_path = payload.get("output_path")
    if output_path:
        print(f"\nSaved: {output_path}")
    else:
        print("\nSaved: no (pass --output to write JSON)")


def _default_output_path_for_estimator(
    *,
    output_dir: Path,
    estimator_key: str,
    profile_name: str,
    scenario_name: str,
) -> Path:
    safe_key = estimator_key.replace(".", "_")
    filename = f"{safe_key}_{profile_name}_{scenario_name}.json"
    return output_dir / safe_key / filename


def cmd_run_all(args: argparse.Namespace, runner: BenchmarkRunner) -> int:
    if args.scenario not in runner.registry.scenarios:
        raise KeyError(f"Unknown scenario name: {args.scenario}")
    if args.profile not in runner.registry.profiles:
        raise KeyError(f"Unknown profile name: {args.profile}")

    estimators = runner.list_estimators()
    failures: list[tuple[str, str]] = []
    started_at = time.perf_counter()

    print("Run-all benchmark")
    print(f"  Estimators: {len(estimators)}")
    print(f"  Scenario: {args.scenario}")
    print(f"  Profile: {args.profile}")
    print(f"  Variant output: {args.variant_output}")

    for index, estimator in enumerate(estimators, start=1):
        spec = runner.registry.estimators[estimator]
        variants = [
            params
            for params in expand_constructor_variants(spec)
            if not should_skip(params, args.scenario, spec.skip_if)
        ]
        variant_count = len(variants)
        print(f"\nRunning [{index}/{len(estimators)}]: {estimator}")
        print(f"  Planned variants: {variant_count}")
        output_path = None
        if args.output_dir is not None:
            output_path = _default_output_path_for_estimator(
                output_dir=args.output_dir,
                estimator_key=estimator,
                profile_name=args.profile,
                scenario_name=args.scenario,
            )

        estimator_started_at = time.perf_counter()
        try:
            payload = runner.run(
                estimator_key=estimator,
                scenario_name=args.scenario,
                profile_name=args.profile,
                constructor_overrides=None,
                output_path=output_path,
            )
        except Exception as exc:  # pragma: no cover - CLI guardrail
            failures.append((estimator, str(exc)))
            elapsed_s = time.perf_counter() - estimator_started_at
            print(f"  Status: FAIL ({exc})")
            print(f"  Elapsed: {elapsed_s:.2f} s")
            if not args.continue_on_error:
                break
            continue

        executed_variants = len(payload["results"])
        median_ms = min(float(item["median_ms"]) for item in payload["results"])
        elapsed_s = time.perf_counter() - estimator_started_at
        print(
            f"  Status: PASS ({executed_variants} variant(s), best median={median_ms:.6f} ms)"
        )

        if args.variant_output == "all":
            for variant_index, result in enumerate(payload["results"], start=1):
                print(f"    Variant {variant_index}")
                print(f"      params: {result['constructor_params']}")
                print(
                    "      median (ms): "
                    f"{result['median_ms']:.6f} "
                    f"(mean={result['mean_ms']:.6f}, std={result['std_ms']:.6f})"
                )
                print(
                    "      p95/p99 (ms): "
                    f"{result['p95_ms']:.6f} / {result['p99_ms']:.6f}"
                )
                print(
                    "      min/max (ms): "
                    f"{result['min_ms']:.6f} / {result['max_ms']:.6f}"
                )
                print(f"      cv (%): {result['cv_percent']:.2f}")
                print(
                    "      post-fit RAM: "
                    f"{_format_bytes(result.get('post_fit_estimator_bytes'))} "
                    f"({result.get('post_fit_estimator_bytes', 0)} bytes)"
                )

        print(f"  Elapsed: {elapsed_s:.2f} s")

    passed = len(failures) == 0
    total_elapsed_s = time.perf_counter() - started_at
    print("\nRun-all summary")
    print(f"  Passed: {len(estimators) - len(failures)}")
    print(f"  Failed: {len(failures)}")
    print(f"  Total elapsed: {total_elapsed_s:.2f} s")
    if failures:
        print("  Failed estimators:")
        for estimator, message in failures:
            print(f"    - {estimator}: {message}")

    return 0 if passed else 1


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid benchmark payload in {path}")
    return payload


def _variant_key(result: dict[str, Any]) -> str:
    params = result.get("constructor_params", {})
    return json.dumps(params, sort_keys=True, separators=(",", ":"))


def _percent_change(baseline: float, candidate: float) -> float:
    if baseline == 0.0:
        return 0.0 if candidate == 0.0 else float("inf")
    return ((candidate - baseline) / baseline) * 100.0


def cmd_compare(args: argparse.Namespace, runner: BenchmarkRunner) -> int:
    baseline = _read_json(args.baseline)
    candidate = _read_json(args.candidate)

    estimator = (
        args.estimator or baseline.get("estimator") or candidate.get("estimator")
    )
    if not estimator:
        raise ValueError("Could not infer estimator; pass --estimator explicitly.")

    if estimator not in runner.registry.estimators:
        raise KeyError(f"Unknown estimator key in registry: {estimator}")

    policy = runner.registry.estimators[estimator].compare_policy
    max_p95_regression = float(policy.get("max_p95_regression_percent", float("inf")))
    require_median_improvement = bool(policy.get("require_median_improvement", False))
    stability_guard = policy.get("stability_guard", {})
    max_cv_percent = stability_guard.get("max_cv_percent")

    baseline_results = {
        _variant_key(result): result for result in baseline.get("results", [])
    }
    candidate_results = {
        _variant_key(result): result for result in candidate.get("results", [])
    }

    shared_keys = sorted(set(baseline_results) & set(candidate_results))
    missing_in_candidate = sorted(set(baseline_results) - set(candidate_results))
    missing_in_baseline = sorted(set(candidate_results) - set(baseline_results))

    rows: list[dict[str, Any]] = []
    all_passed = True

    for key in shared_keys:
        base = baseline_results[key]
        cand = candidate_results[key]

        median_change = _percent_change(
            float(base["median_ms"]), float(cand["median_ms"])
        )
        p95_change = _percent_change(float(base["p95_ms"]), float(cand["p95_ms"]))

        passed_median = True
        if require_median_improvement:
            passed_median = float(cand["median_ms"]) < float(base["median_ms"])

        passed_p95 = p95_change <= max_p95_regression

        candidate_cv = float(cand["cv_percent"])
        passed_stability = True
        if max_cv_percent is not None:
            passed_stability = candidate_cv <= float(max_cv_percent)

        passed = passed_median and passed_p95 and passed_stability
        all_passed = all_passed and passed

        rows.append(
            {
                "constructor_params": cand.get("constructor_params", {}),
                "baseline_median_ms": float(base["median_ms"]),
                "candidate_median_ms": float(cand["median_ms"]),
                "median_change_percent": median_change,
                "baseline_p95_ms": float(base["p95_ms"]),
                "candidate_p95_ms": float(cand["p95_ms"]),
                "p95_change_percent": p95_change,
                "candidate_cv_percent": candidate_cv,
                "passed_median": passed_median,
                "passed_p95": passed_p95,
                "passed_stability": passed_stability,
                "passed": passed,
            }
        )

    if missing_in_candidate or missing_in_baseline:
        all_passed = False

    report = {
        "estimator": estimator,
        "policy": {
            "max_p95_regression_percent": max_p95_regression,
            "require_median_improvement": require_median_improvement,
            "max_cv_percent": max_cv_percent,
        },
        "baseline": str(args.baseline),
        "candidate": str(args.candidate),
        "variants_compared": len(rows),
        "missing_in_candidate": [json.loads(item) for item in missing_in_candidate],
        "missing_in_baseline": [json.loads(item) for item in missing_in_baseline],
        "rows": rows,
        "passed": all_passed,
    }

    print("Comparison report")
    print(f"  Estimator: {estimator}")
    print(f"  Variants compared: {len(rows)}")
    print(f"  Missing in candidate: {len(missing_in_candidate)}")
    print(f"  Missing in baseline: {len(missing_in_baseline)}")
    print(f"  Status: {'PASS' if all_passed else 'FAIL'}")

    for index, row in enumerate(rows, start=1):
        print(f"\nVariant {index}")
        print(f"  params: {row['constructor_params']}")
        print(
            "  median (ms): "
            f"{row['baseline_median_ms']:.6f} -> {row['candidate_median_ms']:.6f} "
            f"({row['median_change_percent']:+.2f}%)"
        )
        print(
            "  p95 (ms): "
            f"{row['baseline_p95_ms']:.6f} -> {row['candidate_p95_ms']:.6f} "
            f"({row['p95_change_percent']:+.2f}%)"
        )
        print(f"  cv (%): {row['candidate_cv_percent']:.2f}")
        print(
            "  checks: "
            f"median={row['passed_median']} "
            f"p95={row['passed_p95']} "
            f"stability={row['passed_stability']}"
        )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nSaved report to {args.output}")

    return 0 if all_passed else 1


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    registry = load_registry(args.registry)
    runner = BenchmarkRunner(registry)

    if args.command == "list":
        cmd_list(runner, args.kind)
        return

    if args.command == "run":
        cmd_run(args, runner)
        return

    if args.command == "run-all":
        raise SystemExit(cmd_run_all(args, runner))
        return

    if args.command == "compare":
        raise SystemExit(cmd_compare(args, runner))
        return

    raise RuntimeError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
