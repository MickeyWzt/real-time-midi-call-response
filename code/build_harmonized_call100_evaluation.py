"""Build a comparable Call100 candidate evaluation from one shared A0/A6 batch.

The legacy candidate run labeled a style-projected response as ``amt_small_raw``.
This script replaces that condition with the true A0 output, uses A6 as the
controlled condition, retains the rule-based motif batch, and re-scores every
MIDI with the current, versioned metric implementation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import zlib
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, Iterable, List, Sequence, Tuple

import evaluate_melody_metrics as metrics


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ABLATION = ROOT / "ab_tests" / "objective_ablation_call100_trials15_latency" / "ablation_all_results.csv"
DEFAULT_LEGACY = ROOT / "ab_tests" / "objective_search_call100_trials15" / "all_objective_results.csv"
DEFAULT_OUTPUT = ROOT / "ab_tests" / "call100_harmonized_evaluation"

METRIC_FIELDS = [
    "objective_score",
    "style_compliance_score",
    "non_style_structural_score",
    "tonality_score",
    "rhythm_score",
    "interval_score",
    "repetition_score",
    "pitch_diversity_score",
    "compression_score",
    "pche",
    "upc",
    "psr",
    "tone_span_ratio",
    "cpr",
    "longest_repeat_run",
    "strong_beat_stable_rate",
    "qualified_note_rate",
    "qualified_rhythm_rate",
    "groove_similarity",
    "cadence_score",
    "max_abs_interval",
    "note_count",
    "duration_seconds",
]


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def provenance_label(path: Path) -> str:
    """Return a portable provenance label without exposing host directories."""

    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return resolved.name


def fnum(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def source_rows(ablation_path: Path, legacy_path: Path) -> List[Dict[str, str]]:
    ablation = read_csv(ablation_path)
    legacy = read_csv(legacy_path)
    selected: List[Dict[str, str]] = []
    for row in ablation:
        variant = row.get("variant_short")
        if variant not in {"A0", "A6"}:
            continue
        item = dict(row)
        item["candidate"] = "amt_small_raw" if variant == "A0" else "amt_small_controlled"
        item["source_experiment"] = "A0_A6_shared_ablation_batch"
        item["source_variant"] = variant
        selected.append(item)
    for row in legacy:
        if row.get("candidate") != "motif_transform_baseline":
            continue
        item = dict(row)
        item["source_experiment"] = "legacy_motif_batch_rescored"
        item["source_variant"] = "motif"
        selected.append(item)
    return selected


def metric_args() -> argparse.Namespace:
    return metrics.build_parser().parse_args([])


def rescore(rows: Sequence[Dict[str, str]], args: argparse.Namespace, max_rows: int | None) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    selected = list(rows[:max_rows]) if max_rows is not None else list(rows)
    output: List[Dict[str, object]] = []
    max_abs_drift = 0.0
    drift_count = 0
    for index, source in enumerate(selected, start=1):
        midi_path = Path(source["response_midi_path"])
        if not midi_path.exists():
            raise FileNotFoundError(midi_path)
        notes, ticks_per_beat = metrics.read_notes(midi_path)
        scored = metrics.evaluate_notes(notes, ticks_per_beat, args)
        old_score = fnum(source.get("objective_score"))
        drift = abs(fnum(scored.get("objective_score")) - old_score)
        max_abs_drift = max(max_abs_drift, drift)
        drift_count += int(drift > 5e-7)
        item: Dict[str, object] = {
            "call_id": source.get("call_id", ""),
            "origin": source.get("origin", ""),
            "source_dataset": source.get("source_dataset", ""),
            "category": source.get("category", ""),
            "sub_category": source.get("sub_category", ""),
            "preset": source.get("preset", ""),
            "candidate": source.get("candidate", ""),
            "trial": int(fnum(source.get("trial"))),
            "seed": int(fnum(source.get("seed"))),
            "source_experiment": source.get("source_experiment", ""),
            "source_variant": source.get("source_variant", ""),
            "score_version": metrics.SCORE_VERSION,
            "response_midi_path": str(midi_path),
            "response_midi_sha256": sha256(midi_path),
            "previous_objective_score": f"{old_score:.6f}",
            "rescore_abs_drift": f"{drift:.9f}",
        }
        for field in METRIC_FIELDS:
            item[field] = f"{fnum(scored.get(field)):.6f}"
        output.append(item)
        if index % 1000 == 0 or index == len(selected):
            print(f"[rescore] {index}/{len(selected)}")
    audit = {
        "row_count": len(output),
        "max_abs_objective_score_drift": max_abs_drift,
        "rows_with_drift_gt_5e-7": drift_count,
    }
    return output, audit


def summarize(rows: Sequence[Dict[str, object]], keys: Sequence[str]) -> List[Dict[str, object]]:
    groups: Dict[Tuple[object, ...], List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(key, "") for key in keys)].append(row)
    output: List[Dict[str, object]] = []
    for group_key, items in groups.items():
        result: Dict[str, object] = {key: value for key, value in zip(keys, group_key)}
        result["sample_count"] = len(items)
        result["inference_scope"] = "descriptive repeated rows; use results/paper_clustered_statistics for uncertainty"
        for field in ["objective_score", "style_compliance_score", "non_style_structural_score"]:
            values = [fnum(item.get(field)) for item in items]
            center = mean(values)
            sd = stdev(values) if len(values) > 1 else 0.0
            result[f"mean_{field}"] = f"{center:.6f}"
            result[f"sd_{field}"] = f"{sd:.6f}"
        output.append(result)
    output.sort(key=lambda item: (str(item.get("candidate", "")), str(item.get("preset", ""))))
    return output


def paired_row_descriptives(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    import numpy as np

    buckets: Dict[Tuple[str, str, int], Dict[str, float]] = defaultdict(dict)
    for row in rows:
        key = (str(row.get("preset")), str(row.get("call_id")), int(fnum(row.get("trial"))))
        buckets[key][str(row.get("candidate"))] = fnum(row.get("objective_score"))
    comparisons = [
        ("controlled_minus_raw", "amt_small_controlled", "amt_small_raw"),
        ("controlled_minus_motif", "amt_small_controlled", "motif_transform_baseline"),
        ("motif_minus_raw", "motif_transform_baseline", "amt_small_raw"),
    ]
    output: List[Dict[str, object]] = []
    for label, left, right in comparisons:
        diffs = np.array(
            [values[left] - values[right] for values in buckets.values() if left in values and right in values],
            dtype=float,
        )
        center = float(diffs.mean())
        sd = float(diffs.std(ddof=1))
        output.append(
            {
                "comparison": label,
                "candidate_a": left,
                "candidate_b": right,
                "paired_row_count": len(diffs),
                "mean_difference": f"{center:.6f}",
                "sd_difference": f"{sd:.6f}",
                "positive_rows": int(np.sum(diffs > 0)),
                "negative_rows": int(np.sum(diffs < 0)),
                "tied_rows": int(np.sum(diffs == 0)),
                "inference_scope": "descriptive repeated rows; use results/paper_clustered_statistics/candidate_clustered_comparisons.csv",
            }
        )
    return output


def write_report(output_dir: Path, summary: Sequence[Dict[str, object]], tests: Sequence[Dict[str, object]], audit: Dict[str, object]) -> None:
    lines = [
        "# Harmonized Call100 Candidate Evaluation",
        "",
        "The raw and controlled AMT conditions reuse the exact A0 and A6 batches from the module ablation. The motif batch is retained from the rule-based run, and all 27,000 MIDI files are re-scored with the same versioned evaluator.",
        "",
        f"- Score version: `{metrics.SCORE_VERSION}`",
        f"- Rows: `{audit['row_count']}`",
        f"- Maximum score drift after re-scoring: `{audit['max_abs_objective_score_drift']:.9f}`",
        "",
        "## Candidate Summary",
        "",
        "| candidate | n | composite | style compliance | non-style structural |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in summary:
        lines.append(
            f"| {row['candidate']} | {row['sample_count']} | {row['mean_objective_score']} | "
            f"{row['mean_style_compliance_score']} | {row['mean_non_style_structural_score']} |"
        )
    lines.extend(
        [
            "",
            "## Descriptive Paired Rows",
            "",
            "The 9,000 repeated rows per candidate are not independent musical inputs. This section reports descriptive row summaries only. Canonical uncertainty is in `results/paper_clustered_statistics/` after averaging within 100 Calls, with an 87-source sensitivity analysis.",
            "",
            "| comparison | rows | mean difference | SD across rows |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for row in tests:
        lines.append(
            f"| {row['comparison']} | {row['paired_row_count']} | {row['mean_difference']} | "
            f"{row['sd_difference']} |"
        )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ablation-results", type=Path, default=DEFAULT_ABLATION)
    parser.add_argument("--legacy-results", type=Path, default=DEFAULT_LEGACY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-iterations", type=int, default=5000, help=argparse.SUPPRESS)
    parser.add_argument("--max-rows", type=int, default=None)
    args = parser.parse_args()

    metrics.require_runtime()
    rows = source_rows(args.ablation_results, args.legacy_results)
    expected = 27000 if args.max_rows is None else min(args.max_rows, 27000)
    if len(rows) != 27000:
        raise RuntimeError(f"Expected 27,000 source rows, found {len(rows)}")
    rescored, audit = rescore(rows, metric_args(), args.max_rows)
    if len(rescored) != expected:
        raise RuntimeError(f"Expected {expected} rescored rows, found {len(rescored)}")
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "all_harmonized_results.csv", rescored)
    summary = summarize(rescored, ["candidate"])
    preset_summary = summarize(rescored, ["preset", "candidate"])
    tests = paired_row_descriptives(rescored) if args.max_rows is None else []
    write_csv(output_dir / "summary_by_candidate.csv", summary)
    write_csv(output_dir / "summary_by_preset_candidate.csv", preset_summary)
    write_csv(output_dir / "paired_comparisons.csv", tests)
    spec = metrics.score_spec(metric_args())
    (output_dir / "score_spec.json").write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
    provenance = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "runtime": {
            "python_version": platform.python_version(),
            "zlib_compile_version": zlib.ZLIB_VERSION,
            "zlib_runtime_version": zlib.ZLIB_RUNTIME_VERSION,
        },
        "ablation_results": provenance_label(args.ablation_results),
        "ablation_results_sha256": sha256(args.ablation_results),
        "legacy_results": provenance_label(args.legacy_results),
        "legacy_results_sha256": sha256(args.legacy_results),
        "evaluator": provenance_label(Path(metrics.__file__)),
        "evaluator_sha256": sha256(Path(metrics.__file__).resolve()),
        "candidate_sources": {
            "amt_small_raw": "A0_raw_amt",
            "amt_small_controlled": "A6_full_controlled",
            "motif_transform_baseline": "legacy motif batch, re-scored",
        },
        "audit": audit,
    }
    (output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    write_report(output_dir, summary, tests, audit)
    print(f"[done] output={output_dir}")


if __name__ == "__main__":
    main()
