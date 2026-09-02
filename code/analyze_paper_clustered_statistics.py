"""Recompute paper-level inference with Call100 calls as the independent units.

The released trial tables contain repeated generations for each call (six presets
and fifteen trials).  Those rows are useful descriptively, but treating all of
them as independent makes uncertainty too small.  This script first averages
within each call, then bootstraps the 100 call means.  A second sensitivity
analysis averages calls within their originating POP909 song (artificial stress
calls remain separate clusters) and bootstraps those source-cluster means.

The scheduler replay uses a commit-anchored 80 ms first-generated-event
readiness target. It does not contain MIDI-send, audio-onset, or per-event
readiness timestamps and therefore reports only a lower bound and cannot
estimate end-to-end latency or mid-phrase buffer starvation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANDIDATE = ROOT / "results" / "call100_harmonized_evaluation" / "harmonized_trial_metrics.csv"
DEFAULT_ABLATION = ROOT / "results" / "call100_ablation_latency" / "ablation_trial_metrics.csv"
DEFAULT_LATENCY = ROOT / "results" / "call100_ablation_latency" / "latency_log_all_trials.csv"
DEFAULT_MANIFEST = ROOT / "results" / "call100_dataset" / "call100_manifest_public.csv"
DEFAULT_OUTPUT = ROOT / "results" / "paper_clustered_statistics"


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def provenance_path(path: Path) -> str:
    """Return a portable input label without exposing a local absolute path."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.name


def fnum(value: object) -> float:
    return float(value)


def percentile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("Cannot compute a percentile of an empty sequence")
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def bootstrap_mean_ci(values: Sequence[float], iterations: int, seed: int) -> Tuple[float, float, float]:
    if not values:
        raise ValueError("No values supplied for bootstrap")
    rng = random.Random(seed)
    sample_count = len(values)
    draws = [
        sum(values[rng.randrange(sample_count)] for _ in range(sample_count)) / sample_count
        for _ in range(iterations)
    ]
    return mean(values), percentile(draws, 0.025), percentile(draws, 0.975)


def mean_by_key(
    rows: Iterable[Mapping[str, str]],
    key_fields: Sequence[str],
    value_field: str,
) -> Dict[Tuple[str, ...], float]:
    groups: Dict[Tuple[str, ...], List[float]] = defaultdict(list)
    for row in rows:
        key = tuple(str(row[field]) for field in key_fields)
        groups[key].append(fnum(row[value_field]))
    return {key: mean(values) for key, values in groups.items()}


def source_clusters(manifest_rows: Sequence[Mapping[str, str]]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for row in manifest_rows:
        call_id = str(row["call_id"])
        origin = str(row.get("origin", ""))
        source_group = str(row.get("source_group", ""))
        source_file = Path(str(row.get("source_file", ""))).name
        if origin == "artificial_stress":
            cluster = f"artificial_{call_id}"
        else:
            match = re.fullmatch(r"song_(\d+)", source_group)
            if match is None:
                match = re.match(r"R\d+_(\d+)", source_file)
            cluster = f"pop909_{int(match.group(1)):03d}" if match else f"other_{call_id}"
        mapping[call_id] = cluster
    return mapping


def aggregate_by_source(call_values: Mapping[str, float], clusters: Mapping[str, str]) -> Dict[str, float]:
    grouped: Dict[str, List[float]] = defaultdict(list)
    for call_id, value in call_values.items():
        grouped[clusters[call_id]].append(value)
    return {cluster: mean(values) for cluster, values in grouped.items()}


def leave_one_cluster_out(call_values: Mapping[str, float], clusters: Mapping[str, str]) -> Tuple[float, float]:
    unique_clusters = sorted(set(clusters[call_id] for call_id in call_values))
    estimates = []
    for omitted in unique_clusters:
        retained = [
            value
            for call_id, value in call_values.items()
            if clusters[call_id] != omitted
        ]
        estimates.append(mean(retained))
    return min(estimates), max(estimates)


def comparison_row(
    label: str,
    call_values: Mapping[str, float],
    clusters: Mapping[str, str],
    descriptive_pairs: int,
    iterations: int,
    seed: int,
) -> Dict[str, object]:
    ordered_calls = [call_values[call_id] for call_id in sorted(call_values)]
    call_mean, call_low, call_high = bootstrap_mean_ci(ordered_calls, iterations, seed)
    source_values = aggregate_by_source(call_values, clusters)
    source_mean, source_low, source_high = bootstrap_mean_ci(
        [source_values[key] for key in sorted(source_values)],
        iterations,
        seed + 1000,
    )
    loo_low, loo_high = leave_one_cluster_out(call_values, clusters)
    return {
        "comparison": label,
        "descriptive_paired_rows": descriptive_pairs,
        "independent_calls": len(ordered_calls),
        "source_clusters": len(source_values),
        "mean_difference": f"{call_mean:.6f}",
        "call_cluster_ci95_low": f"{call_low:.6f}",
        "call_cluster_ci95_high": f"{call_high:.6f}",
        "positive_call_fraction": f"{sum(value > 0 for value in ordered_calls) / len(ordered_calls):.6f}",
        "source_cluster_mean_difference": f"{source_mean:.6f}",
        "source_cluster_ci95_low": f"{source_low:.6f}",
        "source_cluster_ci95_high": f"{source_high:.6f}",
        "leave_one_source_out_min": f"{loo_low:.6f}",
        "leave_one_source_out_max": f"{loo_high:.6f}",
    }


def candidate_statistics(
    rows: Sequence[Mapping[str, str]],
    clusters: Mapping[str, str],
    iterations: int,
    seed: int,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    call_candidate = mean_by_key(rows, ["call_id", "candidate"], "objective_score")
    candidates = sorted({key[1] for key in call_candidate})
    summaries: List[Dict[str, object]] = []
    for index, candidate in enumerate(candidates):
        values = {
            call_id: value
            for (call_id, candidate_id), value in call_candidate.items()
            if candidate_id == candidate
        }
        center, low, high = bootstrap_mean_ci(
            [values[key] for key in sorted(values)], iterations, seed + index
        )
        summaries.append(
            {
                "candidate": candidate,
                "descriptive_rows": sum(row["candidate"] == candidate for row in rows),
                "independent_calls": len(values),
                "mean_objective_score": f"{center:.6f}",
                "call_cluster_ci95_low": f"{low:.6f}",
                "call_cluster_ci95_high": f"{high:.6f}",
            }
        )

    comparisons = [
        ("controlled_minus_raw", "amt_small_controlled", "amt_small_raw"),
        ("controlled_minus_motif", "amt_small_controlled", "motif_transform_baseline"),
        ("motif_minus_raw", "motif_transform_baseline", "amt_small_raw"),
    ]
    output: List[Dict[str, object]] = []
    for index, (label, left, right) in enumerate(comparisons):
        diffs = {
            call_id: call_candidate[(call_id, left)] - call_candidate[(call_id, right)]
            for call_id in sorted(clusters)
        }
        descriptive_pairs = sum(row["candidate"] == left for row in rows)
        output.append(
            comparison_row(
                label,
                diffs,
                clusters,
                descriptive_pairs,
                iterations,
                seed + 100 + index,
            )
        )
    return summaries, output


def ablation_statistics(
    rows: Sequence[Mapping[str, str]],
    clusters: Mapping[str, str],
    iterations: int,
    seed: int,
) -> List[Dict[str, object]]:
    variants = ["A0", "A1", "A2", "A3", "A4", "A5", "A6"]
    metrics = [
        ("objective_score", "SCC"),
        ("style_compliance_score", "constraint_target"),
        ("non_style_structural_score", "remaining_components"),
    ]
    output: List[Dict[str, object]] = []
    for metric_index, (field, metric_label) in enumerate(metrics):
        call_variant = mean_by_key(rows, ["call_id", "variant_short"], field)
        for step_index, (left, right) in enumerate(zip(variants, variants[1:])):
            diffs = {
                call_id: call_variant[(call_id, right)] - call_variant[(call_id, left)]
                for call_id in sorted(clusters)
            }
            row = comparison_row(
                f"{right}_minus_{left}",
                diffs,
                clusters,
                descriptive_pairs=9000,
                iterations=iterations,
                seed=seed + 200 + metric_index * 20 + step_index,
            )
            row["metric"] = metric_label
            output.append(row)
    return output


def latency_statistics(
    rows: Sequence[Mapping[str, str]],
    clusters: Mapping[str, str],
    iterations: int,
    seed: int,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    latency_field = (
        "commit_to_first_event_readiness_lower_bound_ms"
        if rows and "commit_to_first_event_readiness_lower_bound_ms" in rows[0]
        else "endpoint_to_first_midi_ms"
    )
    latency_by_call = mean_by_key(rows, ["call_id", "condition"], latency_field)
    miss_field = (
        "startup_deadline_miss"
        if rows and "startup_deadline_miss" in rows[0]
        else "buffer_underrun_count"
    )
    miss_by_call = mean_by_key(rows, ["call_id", "condition"], miss_field)
    conditions = ["L0_preload_off", "L1_preload_on"]
    summaries: List[Dict[str, object]] = []
    for index, condition in enumerate(conditions):
        latency_values = [
            latency_by_call[(call_id, condition)] for call_id in sorted(clusters)
        ]
        miss_values = [miss_by_call[(call_id, condition)] for call_id in sorted(clusters)]
        lat_center, lat_low, lat_high = bootstrap_mean_ci(
            latency_values, iterations, seed + 300 + index
        )
        miss_center, miss_low, miss_high = bootstrap_mean_ci(
            miss_values, iterations, seed + 320 + index
        )
        summaries.append(
            {
                "condition": condition,
                "descriptive_rows": sum(row["condition"] == condition for row in rows),
                "independent_calls": len(latency_values),
                "mean_first_event_readiness_lower_bound_ms": f"{lat_center:.6f}",
                # Legacy alias retained for older report consumers.
                "mean_endpoint_to_first_midi_ms": f"{lat_center:.6f}",
                "call_cluster_ci95_low_ms": f"{lat_low:.6f}",
                "call_cluster_ci95_high_ms": f"{lat_high:.6f}",
                "startup_deadline_miss_rate": f"{miss_center:.6f}",
                "startup_miss_call_cluster_ci95_low": f"{miss_low:.6f}",
                "startup_miss_call_cluster_ci95_high": f"{miss_high:.6f}",
            }
        )

    latency_diffs = {
        call_id: latency_by_call[(call_id, "L0_preload_off")]
        - latency_by_call[(call_id, "L1_preload_on")]
        for call_id in sorted(clusters)
    }
    miss_diffs = {
        call_id: miss_by_call[(call_id, "L0_preload_off")]
        - miss_by_call[(call_id, "L1_preload_on")]
        for call_id in sorted(clusters)
    }
    comparisons = [
        comparison_row(
            "preload_readiness_lower_bound_reduction_ms",
            latency_diffs,
            clusters,
            descriptive_pairs=9000,
            iterations=iterations,
            seed=seed + 340,
        ),
        comparison_row(
            "startup_deadline_miss_rate_reduction",
            miss_diffs,
            clusters,
            descriptive_pairs=9000,
            iterations=iterations,
            seed=seed + 341,
        ),
    ]
    return summaries, comparisons


def markdown_table(rows: Sequence[Mapping[str, object]], fields: Sequence[str]) -> List[str]:
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join("---" for _ in fields) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |")
    return lines


def build_report(
    candidate_summary: Sequence[Mapping[str, object]],
    candidate_comparisons: Sequence[Mapping[str, object]],
    ablation: Sequence[Mapping[str, object]],
    latency_summary: Sequence[Mapping[str, object]],
    latency_comparisons: Sequence[Mapping[str, object]],
) -> str:
    lines = [
        "# Cluster-aware paper statistics",
        "",
        "Trial rows are descriptive. Primary uncertainty is computed after averaging within each of the 100 calls. Source-cluster sensitivity uses 87 independent clusters (77 POP909 songs and 10 artificial stress calls).",
        "",
        "## Candidate means",
        "",
        *markdown_table(candidate_summary, ["candidate", "mean_objective_score", "call_cluster_ci95_low", "call_cluster_ci95_high"]),
        "",
        "## Candidate comparisons",
        "",
        *markdown_table(candidate_comparisons, ["comparison", "mean_difference", "call_cluster_ci95_low", "call_cluster_ci95_high", "positive_call_fraction", "source_cluster_ci95_low", "source_cluster_ci95_high"]),
        "",
        "## Stepwise ablation (SCC only)",
        "",
        *markdown_table([row for row in ablation if row["metric"] == "SCC"], ["comparison", "mean_difference", "call_cluster_ci95_low", "call_cluster_ci95_high", "positive_call_fraction"]),
        "",
        "## Scheduler replay",
        "",
        *markdown_table(latency_summary, ["condition", "mean_first_event_readiness_lower_bound_ms", "call_cluster_ci95_low_ms", "call_cluster_ci95_high_ms", "startup_deadline_miss_rate"]),
        "",
        *markdown_table(latency_comparisons, ["comparison", "mean_difference", "call_cluster_ci95_low", "call_cluster_ci95_high", "positive_call_fraction"]),
        "",
        "The 80 ms target is anchored to endpoint commit. Full streaming underrun requires per-event readiness timestamps and is not estimated by these data.",
        "",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--ablation", type=Path, default=DEFAULT_ABLATION)
    parser.add_argument("--latency", type=Path, default=DEFAULT_LATENCY)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-iterations", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=20260815)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.bootstrap_iterations < 1000:
        raise SystemExit("--bootstrap-iterations must be at least 1000")
    manifest_rows = read_csv(args.manifest)
    clusters = source_clusters(manifest_rows)
    if len(clusters) != 100:
        raise RuntimeError(f"Expected 100 calls, found {len(clusters)}")
    if len(set(clusters.values())) != 87:
        raise RuntimeError(f"Expected 87 source clusters, found {len(set(clusters.values()))}")

    candidate_rows = read_csv(args.candidate)
    ablation_rows = read_csv(args.ablation)
    latency_rows = read_csv(args.latency)
    candidate_summary, candidate_comparisons = candidate_statistics(
        candidate_rows, clusters, args.bootstrap_iterations, args.seed
    )
    ablation = ablation_statistics(
        ablation_rows, clusters, args.bootstrap_iterations, args.seed
    )
    latency_summary, latency_comparisons = latency_statistics(
        latency_rows, clusters, args.bootstrap_iterations, args.seed
    )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "candidate_call_cluster_summary.csv", candidate_summary)
    write_csv(output_dir / "candidate_clustered_comparisons.csv", candidate_comparisons)
    write_csv(output_dir / "ablation_clustered_stepwise.csv", ablation)
    write_csv(output_dir / "latency_call_cluster_summary.csv", latency_summary)
    write_csv(output_dir / "latency_clustered_comparisons.csv", latency_comparisons)
    (output_dir / "report.md").write_text(
        build_report(
            candidate_summary,
            candidate_comparisons,
            ablation,
            latency_summary,
            latency_comparisons,
        ),
        encoding="utf-8",
        newline="\n",
    )
    provenance = {
        "bootstrap_iterations": args.bootstrap_iterations,
        "seed": args.seed,
        "independent_calls": len(clusters),
        "source_clusters": len(set(clusters.values())),
        "source_cluster_definition": "77 POP909 song IDs plus 10 artificial calls",
        "inputs": {
            "candidate": {"path": provenance_path(args.candidate), "sha256": sha256(args.candidate)},
            "ablation": {"path": provenance_path(args.ablation), "sha256": sha256(args.ablation)},
            "latency": {"path": provenance_path(args.latency), "sha256": sha256(args.latency)},
            "manifest": {"path": provenance_path(args.manifest), "sha256": sha256(args.manifest)},
        },
    }
    (output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"[done] output={output_dir}")


if __name__ == "__main__":
    main()
