"""
Formal Call100 ablation and realtime-latency logging runner.

Default scale:
  100 calls x 6 presets x 7 ablation variants x 15 trials = 63,000 samples.

Outputs are written to a new directory and do not modify Call50 or the completed
Call100 objective-search run.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import run_call100_objective_search as base


ROOT = Path(__file__).resolve().parents[1]
AB_DIR = ROOT / "ab_tests"
DEFAULT_MANIFEST = AB_DIR / "calls_100_public_final" / "call100_manifest.csv"
DEFAULT_OUTPUT_DIR = AB_DIR / "objective_ablation_call100_trials15_latency"

VARIANTS: List[Dict[str, object]] = [
    {
        "variant": "A0_raw_amt",
        "short": "A0",
        "module_added": "raw AMT",
        "base_candidate": "amt_small_raw",
        "description": "No external control.",
    },
    {
        "variant": "A1_prompt_cleaning",
        "short": "A1",
        "module_added": "prompt cleaning",
        "base_candidate": "amt_small_controlled",
        "description": "Only tail-repeat compression / prompt cleaning is enabled.",
    },
    {
        "variant": "A2_repetition_suppression",
        "short": "A2",
        "module_added": "repetition suppression",
        "base_candidate": "amt_small_controlled",
        "description": "Adds generation-time repetition suppression / resampling.",
    },
    {
        "variant": "A3_duration_matching",
        "short": "A3",
        "module_added": "duration matching",
        "base_candidate": "amt_small_controlled",
        "description": "Adds response duration and note-count matching.",
    },
    {
        "variant": "A4_fallback",
        "short": "A4",
        "module_added": "motif fallback",
        "base_candidate": "amt_small_controlled",
        "description": "Adds motif fallback for empty or invalid neural output.",
    },
    {
        "variant": "A5_style_constraint",
        "short": "A5",
        "module_added": "style constraint",
        "base_candidate": "amt_small_controlled",
        "description": "Adds pentatonic / two-bar / 4-4 response style constraints.",
    },
    {
        "variant": "A6_full_controlled",
        "short": "A6",
        "module_added": "full controlled",
        "base_candidate": "amt_small_controlled",
        "description": "All controlled-AMT modules are enabled.",
    },
]
VARIANT_BY_NAME = {str(row["variant"]): row for row in VARIANTS}
DEFAULT_VARIANTS = [str(row["variant"]) for row in VARIANTS]
DEFAULT_PRESETS = list(base.PRESETS)

ABLATION_FIELDS = [
    "call_id",
    "origin",
    "source_dataset",
    "category",
    "sub_category",
    "preset",
    "ablation_variant",
    "variant_short",
    "module_added",
    "candidate",
    "base_candidate",
    "trial",
    "seed",
    "objective_score",
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
    "raw_response_seconds",
    "actual_response_seconds",
    "duration_match_ratio",
    "duration_error_sec",
    "duration_stretch_factor",
    "fallback_used",
    "fallback_count",
    "empty_output_before_fallback",
    "prompt_cleaning_applied",
    "repetition_resample_count",
    "rejected_dominant_count",
    "duration_matching_applied",
    "style_constraint_applied",
    "full_theory_applied",
    "first_token_latency_sec",
    "response_midi_path",
]

SUMMARY_METRICS = [
    "objective_score",
    "tonality_score",
    "rhythm_score",
    "interval_score",
    "repetition_score",
    "pitch_diversity_score",
    "compression_score",
    "cpr",
    "longest_repeat_run",
    "psr",
    "strong_beat_stable_rate",
    "cadence_score",
    "qualified_rhythm_rate",
    "duration_match_ratio",
    "duration_error_sec",
    "qualified_note_rate",
]

LATENCY_FIELDS = [
    "condition",
    "preset",
    "call_id",
    "trial",
    "seed",
    "length_bin",
    "response_note_count",
    "response_duration_sec",
    "t_endpoint_commit_ms",
    "t_infer_start_ms",
    "t_first_token_ready_ms",
    "t_first_midi_queued_ms",
    "t_first_midi_out_ms",
    "t_response_end_ms",
    "endpoint_to_first_midi_ms",
    "first_token_latency_ms",
    "queue_delay_ms",
    "total_generation_ms",
    "buffer_underrun_count",
    "buffer_underrun_rate",
    "preload_window_ms",
    "micro_buffer_ms",
    "source_latency_event",
]


def write_csv(path: Path, rows: Sequence[Dict[str, object]], fields: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        keys = set()
        for row in rows:
            keys.update(row)
        fields = sorted(keys)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def fnum(value: object, default: float = 0.0) -> float:
    try:
        if value in {"", None}:
            return default
        return float(value)
    except Exception:
        return default


def mean(values: Iterable[float]) -> float:
    vals = [v for v in values if math.isfinite(v)]
    return sum(vals) / len(vals) if vals else 0.0


def stdev(values: Sequence[float]) -> float:
    vals = [v for v in values if math.isfinite(v)]
    if len(vals) < 2:
        return 0.0
    m = mean(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))


def percentile(values: Sequence[float], q: float) -> float:
    vals = sorted(v for v in values if math.isfinite(v))
    if not vals:
        return 0.0
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    return vals[lo] * (hi - pos) + vals[hi] * (pos - lo)


def duration_ratio(actual: float, target: float) -> Tuple[float, float, float]:
    if actual <= 0 or target <= 0:
        return 0.0, 0.0, abs(actual - target)
    return min(actual, target) / max(actual, target), actual / target, abs(actual - target)


def split_run_name(run_name: str) -> Tuple[str, str]:
    if "__" not in run_name:
        raise ValueError(f"Bad ablation run name: {run_name}")
    return tuple(run_name.split("__", 1))  # type: ignore[return-value]


def selected(raw: str, allowed: Sequence[str], label: str) -> List[str]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = [value for value in values if value not in allowed]
    if unknown:
        raise SystemExit(f"Unknown {label}: {unknown}. Available: {allowed}")
    return values


def run_name(preset: str, variant: str) -> str:
    return f"{preset}__{variant}"


def run_logged(command: List[str], cwd: Path, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\n[command] " + " ".join(command) + "\n")
        log.flush()
        proc = subprocess.run(command, cwd=str(cwd), stdout=log, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, command)


def run_complete(run_dir: Path, expected_rows: int) -> bool:
    return base.preset_complete(run_dir, expected_rows)


def run_generation(
    args: argparse.Namespace,
    presets: Sequence[str],
    variants: Sequence[str],
    input_dir: Path,
    call_count: int,
) -> None:
    expected_per_run = call_count * args.trials
    latency_log = Path(args.output_dir) / "latency_events.jsonl"
    for preset_index, preset in enumerate(presets, start=1):
        preset_seed = args.seed + preset_index * 10_000_000
        for variant in variants:
            variant_meta = VARIANT_BY_NAME[variant]
            name = run_name(preset, variant)
            run_dir = Path(args.output_dir) / name
            if args.aggregate_only:
                continue
            if run_dir.exists() and args.force:
                shutil.rmtree(run_dir)
            if run_complete(run_dir, expected_per_run):
                print(f"[skip] {name}: already complete ({expected_per_run} responses)")
                continue
            if run_dir.exists():
                shutil.rmtree(run_dir)
            command = [
                sys.executable,
                str(base.CODE_DIR / "offline_ab_test.py"),
                "--input-dir",
                str(input_dir),
                "--trials",
                str(args.trials),
                "--seed",
                str(preset_seed),
                "--output-dir",
                str(run_dir),
                "--candidates",
                str(variant_meta["base_candidate"]),
                "--ablation-variant",
                variant,
                "--offline",
                "--bpm",
                str(args.bpm),
                "--pentatonic-root",
                str(args.pentatonic_root),
                "--pentatonic-mode",
                args.pentatonic_mode,
                "--latency-log",
                str(latency_log),
            ]
            command = base.apply_preset(command, base.PRESETS[preset])
            log_path = Path(args.output_dir) / "logs" / f"{name}.log"
            print(f"[run] {name}: generating {expected_per_run} responses seed={preset_seed}")
            run_logged(command, ROOT, log_path)
            eval_command = [
                sys.executable,
                str(base.CODE_DIR / "evaluate_melody_metrics.py"),
                "--run-dir",
                str(run_dir),
                "--scale-root",
                str(args.pentatonic_root),
                "--scale-mode",
                args.pentatonic_mode,
                "--beats-per-bar",
                "4",
            ]
            print(f"[eval] {name}")
            run_logged(eval_command, ROOT, log_path)
            if not run_complete(run_dir, expected_per_run):
                raise RuntimeError(f"Ablation run did not complete cleanly: {name}")


def aggregate_results(
    args: argparse.Namespace,
    presets: Sequence[str],
    variants: Sequence[str],
    manifest: Sequence[Dict[str, str]],
) -> List[Dict[str, object]]:
    manifest_by_id = {row["call_id"]: row for row in manifest}
    rows: List[Dict[str, object]] = []
    for preset in presets:
        for variant in variants:
            name = run_name(preset, variant)
            run_dir = Path(args.output_dir) / name
            answers = {row["sample_id"]: row for row in read_csv(run_dir / "answer_key.csv")}
            for metric in read_csv(run_dir / "objective_metrics.csv"):
                answer = answers.get(metric["sample_id"], {})
                call_id = answer.get("call_id") or metric.get("call_id")
                if call_id not in manifest_by_id:
                    raise RuntimeError(f"Unknown call_id in {name}: {call_id}")
                source = manifest_by_id[str(call_id)]
                raw_seconds = fnum(answer.get("response_seconds"))
                actual_seconds = fnum(metric.get("duration_seconds"))
                match_ratio, stretch_factor, duration_error = duration_ratio(actual_seconds, raw_seconds)
                fallback_count = int(fnum(answer.get("fallback_count"), 0))
                variant_meta = VARIANT_BY_NAME[variant]
                rows.append(
                    {
                        "call_id": call_id,
                        "origin": source.get("origin", ""),
                        "source_dataset": source.get("source_dataset", ""),
                        "category": source.get("category", ""),
                        "sub_category": source.get("sub_category", ""),
                        "preset": preset,
                        "ablation_variant": variant,
                        "variant_short": variant_meta["short"],
                        "module_added": variant_meta["module_added"],
                        "candidate": variant,
                        "base_candidate": answer.get("candidate", variant_meta["base_candidate"]),
                        "trial": int(fnum(answer.get("trial") or metric.get("trial"), 0)),
                        "seed": int(fnum(answer.get("seed"), 0)),
                        "objective_score": fnum(metric.get("objective_score")),
                        "tonality_score": fnum(metric.get("tonality_score")),
                        "rhythm_score": fnum(metric.get("rhythm_score")),
                        "interval_score": fnum(metric.get("interval_score")),
                        "repetition_score": fnum(metric.get("repetition_score")),
                        "pitch_diversity_score": fnum(metric.get("pitch_diversity_score")),
                        "compression_score": fnum(metric.get("compression_score")),
                        "pche": fnum(metric.get("pche")),
                        "upc": fnum(metric.get("upc")),
                        "psr": fnum(metric.get("psr")),
                        "tone_span_ratio": fnum(metric.get("tone_span_ratio")),
                        "cpr": fnum(metric.get("cpr")),
                        "longest_repeat_run": fnum(metric.get("longest_repeat_run")),
                        "strong_beat_stable_rate": fnum(metric.get("strong_beat_stable_rate")),
                        "qualified_note_rate": fnum(metric.get("qualified_note_rate")),
                        "qualified_rhythm_rate": fnum(metric.get("qualified_rhythm_rate")),
                        "groove_similarity": fnum(metric.get("groove_similarity")),
                        "cadence_score": fnum(metric.get("cadence_score")),
                        "max_abs_interval": fnum(metric.get("max_abs_interval")),
                        "note_count": int(fnum(metric.get("note_count"), 0)),
                        "raw_response_seconds": raw_seconds,
                        "actual_response_seconds": actual_seconds,
                        "duration_match_ratio": match_ratio,
                        "duration_error_sec": duration_error,
                        "duration_stretch_factor": fnum(answer.get("duration_stretch_factor"), stretch_factor),
                        "fallback_used": int(fallback_count > 0 or bool(answer.get("generation_error"))),
                        "fallback_count": fallback_count,
                        "empty_output_before_fallback": int(fnum(answer.get("empty_output_before_fallback"), 0)),
                        "prompt_cleaning_applied": int(fnum(answer.get("prompt_cleaning_applied"), 0)),
                        "repetition_resample_count": int(fnum(answer.get("rejected_repeat_count"), 0)),
                        "rejected_dominant_count": int(fnum(answer.get("rejected_dominant_count"), 0)),
                        "duration_matching_applied": int(fnum(answer.get("duration_matching_applied"), 0)),
                        "style_constraint_applied": int(fnum(answer.get("style_constraint_applied"), 0)),
                        "full_theory_applied": int(fnum(answer.get("full_theory_applied"), 0)),
                        "first_token_latency_sec": fnum(answer.get("first_token_latency_sec")),
                        "response_midi_path": answer.get("response_only_midi") or metric.get("midi_path", ""),
                    }
                )
    return rows


def summarize_group(rows: Sequence[Dict[str, object]], keys: Sequence[str]) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[object, ...], List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(key, "") for key in keys)].append(row)
    result: List[Dict[str, object]] = []
    for group_key, items in sorted(grouped.items()):
        out = {key: value for key, value in zip(keys, group_key)}
        out["sample_count"] = len(items)
        for metric in SUMMARY_METRICS:
            vals = [fnum(row.get(metric)) for row in items]
            m = mean(vals)
            sd = stdev(vals)
            se = sd / math.sqrt(len(vals)) if vals else 0.0
            out[f"mean_{metric}"] = f"{m:.6f}"
            out[f"ci95_low_{metric}"] = f"{m - 1.96 * se:.6f}"
            out[f"ci95_high_{metric}"] = f"{m + 1.96 * se:.6f}"
        result.append(out)
    return result


def variant_order(variant: str) -> int:
    return DEFAULT_VARIANTS.index(variant)


def module_contributions(rows: Sequence[Dict[str, object]], presets: Sequence[str], variants: Sequence[str], iterations: int) -> List[Dict[str, object]]:
    result: List[Dict[str, object]] = []
    metrics = [
        "objective_score",
        "cpr",
        "longest_repeat_run",
        "repetition_score",
        "duration_match_ratio",
        "duration_error_sec",
        "psr",
        "strong_beat_stable_rate",
        "cadence_score",
        "qualified_rhythm_rate",
    ]
    row_map: Dict[Tuple[str, str, str, int], Dict[str, object]] = {}
    for row in rows:
        row_map[(str(row["preset"]), str(row["ablation_variant"]), str(row["call_id"]), int(row["trial"]))] = row

    groups = [("overall", None)] + [(preset, preset) for preset in presets]
    for group_name, preset_filter in groups:
        for idx in range(1, len(variants)):
            prev_variant = variants[idx - 1]
            variant = variants[idx]
            for metric in metrics:
                diffs: List[float] = []
                for key, row in row_map.items():
                    preset, row_variant, call_id, trial = key
                    if preset_filter is not None and preset != preset_filter:
                        continue
                    if row_variant != variant:
                        continue
                    prev = row_map.get((preset, prev_variant, call_id, trial))
                    if prev is None:
                        continue
                    diffs.append(fnum(row.get(metric)) - fnum(prev.get(metric)))
                observed, low, high, p_value = base.bootstrap_ci(
                    diffs,
                    iterations,
                    seed=20260622 + len(result) * 13,
                )
                result.append(
                    {
                        "group": group_name,
                        "module_step": f"{VARIANT_BY_NAME[variant]['short']} minus {VARIANT_BY_NAME[prev_variant]['short']}",
                        "module_added": VARIANT_BY_NAME[variant]["module_added"],
                        "metric": metric,
                        "paired_sample_count": len(diffs),
                        "mean_delta": f"{observed:.6f}",
                        "ci95_low": f"{low:.6f}",
                        "ci95_high": f"{high:.6f}",
                        "p_two_sided": f"{p_value:.6f}",
                    }
                )
    return result


def validate_rows(rows: Sequence[Dict[str, object]], manifest: Sequence[Dict[str, str]], presets: Sequence[str], variants: Sequence[str], trials: int) -> Dict[str, object]:
    errors: List[str] = []
    manifest_ids = {row["call_id"] for row in manifest}
    expected_rows = len(manifest_ids) * len(presets) * len(variants) * trials
    combos = Counter((row.get("preset"), row.get("ablation_variant"), row.get("call_id")) for row in rows)
    bad_combos = [key for key, count in combos.items() if count != trials]
    empty_scores = sum(1 for row in rows if row.get("objective_score") in {"", None})
    missing_midis = sum(1 for row in rows if not Path(str(row.get("response_midi_path", ""))).exists())
    unknown_ids = sorted({str(row.get("call_id")) for row in rows if row.get("call_id") not in manifest_ids})
    if len(rows) != expected_rows:
        errors.append(f"ablation_all_results.csv row count expected {expected_rows}, got {len(rows)}")
    expected_combo_count = len(manifest_ids) * len(presets) * len(variants)
    if len(combos) != expected_combo_count:
        errors.append(f"preset/variant/call_id combo count expected {expected_combo_count}, got {len(combos)}")
    if bad_combos:
        errors.append(f"{len(bad_combos)} preset/variant/call_id combos do not have {trials} rows")
    if empty_scores:
        errors.append(f"objective_score has {empty_scores} empty values")
    if missing_midis:
        errors.append(f"{missing_midis} response_midi_path values do not exist")
    if unknown_ids:
        errors.append(f"unknown call_id values: {unknown_ids[:10]}")
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "expected_rows": expected_rows,
        "actual_rows": len(rows),
        "expected_combo_count": expected_combo_count,
        "actual_combo_count": len(combos),
        "trials_per_combo": trials,
        "bad_combo_count": len(bad_combos),
        "empty_objective_scores": empty_scores,
        "missing_response_midis": missing_midis,
        "unknown_call_ids": unknown_ids,
        "errors": errors,
        "status": "passed" if not errors else "failed",
    }


def svg_escape(text: object) -> str:
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def write_svg(path: Path, parts: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def chart_ablation_bar(summary: Sequence[Dict[str, object]], chart_dir: Path) -> None:
    rows = sorted(summary, key=lambda row: variant_order(str(row["ablation_variant"])))
    values = [fnum(row.get("mean_objective_score")) for row in rows]
    max_value = max(values) if values else 1.0
    width, height = 900, 430
    left, top, chart_w, chart_h = 90, 60, 760, 270
    bar_w = chart_w / max(1, len(rows))
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="30" y="36" font-family="Arial" font-size="22" font-weight="700">A0-A6 Mean Objective Score</text>',
        f'<line x1="{left}" y1="{top + chart_h}" x2="{left + chart_w}" y2="{top + chart_h}" stroke="#111827"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + chart_h}" stroke="#111827"/>',
    ]
    for i, row in enumerate(rows):
        value = values[i]
        h = 0 if max_value <= 0 else (value / max_value) * chart_h
        x = left + i * bar_w + 12
        y = top + chart_h - h
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w - 24:.1f}" height="{h:.1f}" fill="#2563eb"/>')
        parts.append(f'<text x="{x + (bar_w - 24) / 2:.1f}" y="{top + chart_h + 24}" text-anchor="middle" font-family="Arial" font-size="12">{svg_escape(row["variant_short"])}</text>')
        parts.append(f'<text x="{x + (bar_w - 24) / 2:.1f}" y="{y - 6:.1f}" text-anchor="middle" font-family="Arial" font-size="11">{value:.3f}</text>')
    parts.append("</svg>")
    write_svg(chart_dir / "ablation_mean_objective_score.svg", parts)


def chart_module_heatmap(contrib: Sequence[Dict[str, object]], chart_dir: Path) -> None:
    focus = [row for row in contrib if row.get("group") == "overall"]
    steps = []
    metrics = ["objective_score", "cpr", "longest_repeat_run", "repetition_score", "duration_match_ratio", "duration_error_sec", "psr", "cadence_score"]
    for row in focus:
        step = str(row["module_step"])
        if step not in steps:
            steps.append(step)
    lookup = {(str(row["module_step"]), str(row["metric"])): fnum(row["mean_delta"]) for row in focus}
    cell_w, cell_h = 135, 34
    left, top = 210, 75
    width = left + cell_w * len(metrics) + 50
    height = top + cell_h * len(steps) + 80
    max_abs = max([abs(v) for v in lookup.values()] + [0.001])
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="30" y="36" font-family="Arial" font-size="22" font-weight="700">Module Impact Heatmap</text>',
    ]
    for j, metric in enumerate(metrics):
        parts.append(f'<text x="{left + j * cell_w + cell_w / 2}" y="{top - 14}" text-anchor="middle" font-family="Arial" font-size="11">{svg_escape(metric)}</text>')
    for i, step in enumerate(steps):
        y = top + i * cell_h
        parts.append(f'<text x="{left - 10}" y="{y + 22}" text-anchor="end" font-family="Arial" font-size="12">{svg_escape(step)}</text>')
        for j, metric in enumerate(metrics):
            val = lookup.get((step, metric), 0.0)
            intensity = int(35 + 180 * min(1, abs(val) / max_abs))
            color = f"rgb({255 - intensity},{245 - intensity // 3},255)" if val >= 0 else f"rgb(255,{245 - intensity // 3},{255 - intensity})"
            x = left + j * cell_w
            parts.append(f'<rect x="{x}" y="{y}" width="{cell_w - 2}" height="{cell_h - 2}" fill="{color}" stroke="#e5e7eb"/>')
            parts.append(f'<text x="{x + cell_w / 2}" y="{y + 21}" text-anchor="middle" font-family="Arial" font-size="11">{val:.3f}</text>')
    parts.append("</svg>")
    write_svg(chart_dir / "ablation_module_metric_heatmap.svg", parts)


def length_bin(note_count: int) -> str:
    if note_count <= 8:
        return "short"
    if note_count <= 16:
        return "medium"
    return "long"


def read_latency_jsonl(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_latency_trials(output_dir: Path, preload_window_ms: float, micro_buffer_ms: float, underrun_deadline_ms: float) -> List[Dict[str, object]]:
    events = [
        row for row in read_latency_jsonl(output_dir / "latency_events.jsonl")
        if row.get("event_type") == "generation" and row.get("status") == "ok" and str(row.get("run_name", "")).endswith("__A6_full_controlled")
    ]
    trials: List[Dict[str, object]] = []
    for idx, row in enumerate(events, start=1):
        preset, _variant = split_run_name(str(row["run_name"]))
        first_token_ms = fnum(row.get("first_token_latency_sec"), fnum(row.get("generation_latency_sec"))) * 1000.0
        total_generation_ms = fnum(row.get("generation_latency_sec")) * 1000.0
        note_count = int(fnum(row.get("response_note_count"), 0))
        response_duration_sec = fnum(row.get("target_response_seconds"), 0.0)
        for condition in ("L0_preload_off", "L1_preload_on"):
            infer_start = 0.0 if condition == "L0_preload_off" else -preload_window_ms
            first_ready = infer_start + first_token_ms
            first_queued = max(0.0, first_ready)
            first_out = first_queued + micro_buffer_ms
            response_end = first_out + response_duration_sec * 1000.0
            underrun = int(first_queued > underrun_deadline_ms)
            trials.append(
                {
                    "condition": condition,
                    "preset": preset,
                    "call_id": row.get("call_id", ""),
                    "trial": row.get("trial", ""),
                    "seed": row.get("seed", ""),
                    "length_bin": length_bin(note_count),
                    "response_note_count": note_count,
                    "response_duration_sec": f"{response_duration_sec:.6f}",
                    "t_endpoint_commit_ms": "0.000000",
                    "t_infer_start_ms": f"{infer_start:.6f}",
                    "t_first_token_ready_ms": f"{first_ready:.6f}",
                    "t_first_midi_queued_ms": f"{first_queued:.6f}",
                    "t_first_midi_out_ms": f"{first_out:.6f}",
                    "t_response_end_ms": f"{response_end:.6f}",
                    "endpoint_to_first_midi_ms": f"{first_out:.6f}",
                    "first_token_latency_ms": f"{first_token_ms:.6f}",
                    "queue_delay_ms": f"{first_out - first_queued:.6f}",
                    "total_generation_ms": f"{total_generation_ms:.6f}",
                    "buffer_underrun_count": underrun,
                    "buffer_underrun_rate": f"{float(underrun):.6f}",
                    "preload_window_ms": f"{(0.0 if condition == 'L0_preload_off' else preload_window_ms):.6f}",
                    "micro_buffer_ms": f"{micro_buffer_ms:.6f}",
                    "source_latency_event": idx,
                }
            )
    return trials


def latency_summary(rows: Sequence[Dict[str, object]], keys: Sequence[str]) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[object, ...], List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(key, "") for key in keys)].append(row)
    out_rows: List[Dict[str, object]] = []
    for group_key, items in sorted(grouped.items()):
        out = {key: value for key, value in zip(keys, group_key)}
        lat = [fnum(row.get("endpoint_to_first_midi_ms")) for row in items]
        first = [fnum(row.get("first_token_latency_ms")) for row in items]
        total = [fnum(row.get("total_generation_ms")) for row in items]
        underruns = [fnum(row.get("buffer_underrun_count")) for row in items]
        out.update(
            {
                "sample_count": len(items),
                "mean_latency_ms": f"{mean(lat):.6f}",
                "p50_latency_ms": f"{percentile(lat, 0.50):.6f}",
                "p95_latency_ms": f"{percentile(lat, 0.95):.6f}",
                "p99_latency_ms": f"{percentile(lat, 0.99):.6f}",
                "max_latency_ms": f"{max(lat) if lat else 0.0:.6f}",
                "buffer_underrun_count": int(sum(underruns)),
                "underrun_rate": f"{mean(underruns):.6f}",
                "mean_first_token_latency_ms": f"{mean(first):.6f}",
                "mean_total_generation_ms": f"{mean(total):.6f}",
            }
        )
        out_rows.append(out)
    return out_rows


def preload_comparison(latency_rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    by_key: Dict[Tuple[str, str, int], Dict[str, Dict[str, object]]] = defaultdict(dict)
    for row in latency_rows:
        key = (str(row.get("preset")), str(row.get("call_id")), int(fnum(row.get("trial"), 0)))
        by_key[key][str(row.get("condition"))] = row
    diffs = []
    for key, vals in by_key.items():
        if "L0_preload_off" in vals and "L1_preload_on" in vals:
            off = fnum(vals["L0_preload_off"].get("endpoint_to_first_midi_ms"))
            on = fnum(vals["L1_preload_on"].get("endpoint_to_first_midi_ms"))
            diffs.append(off - on)
    observed, low, high, p_value = base.bootstrap_ci(diffs, 1000, 20260622)
    return [
        {
            "comparison": "L1_preload_on_vs_L0_preload_off",
            "paired_sample_count": len(diffs),
            "mean_latency_reduction_ms": f"{observed:.6f}",
            "ci95_low": f"{low:.6f}",
            "ci95_high": f"{high:.6f}",
            "p_two_sided": f"{p_value:.6f}",
        }
    ]


def chart_latency_box(latency_rows: Sequence[Dict[str, object]], chart_dir: Path) -> None:
    groups = ["L0_preload_off", "L1_preload_on"]
    vals = {group: sorted(fnum(row.get("endpoint_to_first_midi_ms")) for row in latency_rows if row.get("condition") == group) for group in groups}
    width, height = 650, 390
    left, top, chart_w, chart_h = 80, 60, 500, 240
    max_v = max([max(v) for v in vals.values() if v] + [1])
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="30" y="36" font-family="Arial" font-size="22" font-weight="700">Preload On/Off First-Note Latency</text>',
        f'<line x1="{left}" y1="{top + chart_h}" x2="{left + chart_w}" y2="{top + chart_h}" stroke="#111827"/>',
    ]
    for idx, group in enumerate(groups):
        data = vals[group]
        x = left + 150 + idx * 190
        q1, med, q3 = percentile(data, 0.25), percentile(data, 0.50), percentile(data, 0.75)
        lo, hi = percentile(data, 0.05), percentile(data, 0.95)
        def y(v: float) -> float:
            return top + chart_h - (v / max_v) * chart_h
        parts.append(f'<line x1="{x}" y1="{y(lo):.1f}" x2="{x}" y2="{y(hi):.1f}" stroke="#111827"/>')
        parts.append(f'<rect x="{x-45}" y="{y(q3):.1f}" width="90" height="{max(1, y(q1)-y(q3)):.1f}" fill="#bfdbfe" stroke="#2563eb"/>')
        parts.append(f'<line x1="{x-45}" y1="{y(med):.1f}" x2="{x+45}" y2="{y(med):.1f}" stroke="#1d4ed8" stroke-width="2"/>')
        parts.append(f'<text x="{x}" y="{top + chart_h + 28}" text-anchor="middle" font-family="Arial" font-size="12">{group}</text>')
        parts.append(f'<text x="{x}" y="{y(med)-8:.1f}" text-anchor="middle" font-family="Arial" font-size="11">p50 {med:.0f} ms</text>')
    parts.append("</svg>")
    write_svg(chart_dir / "latency_preload_on_off_boxplot.svg", parts)


def chart_latency_table(summary: Sequence[Dict[str, object]], chart_dir: Path) -> None:
    rows = [row for row in summary if "condition" in row]
    width, height = 820, 210
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="30" y="36" font-family="Arial" font-size="22" font-weight="700">Latency Percentiles</text>',
    ]
    cols = ["condition", "p50_latency_ms", "p95_latency_ms", "p99_latency_ms", "max_latency_ms", "underrun_rate"]
    x0, y0, row_h, col_w = 30, 65, 34, 125
    for j, col in enumerate(cols):
        parts.append(f'<text x="{x0 + j * col_w}" y="{y0}" font-family="Arial" font-size="12" font-weight="700">{svg_escape(col)}</text>')
    for i, row in enumerate(rows):
        y = y0 + (i + 1) * row_h
        for j, col in enumerate(cols):
            parts.append(f'<text x="{x0 + j * col_w}" y="{y}" font-family="Arial" font-size="12">{svg_escape(row.get(col, ""))}</text>')
    parts.append("</svg>")
    write_svg(chart_dir / "latency_percentile_table.svg", parts)


def chart_length_scatter(latency_rows: Sequence[Dict[str, object]], chart_dir: Path) -> None:
    rows = [row for row in latency_rows if row.get("condition") == "L0_preload_off"]
    width, height = 760, 430
    left, top, chart_w, chart_h = 80, 50, 620, 300
    max_x = max([fnum(row.get("response_note_count")) for row in rows] + [1])
    max_y = max([fnum(row.get("total_generation_ms")) for row in rows] + [1])
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="30" y="32" font-family="Arial" font-size="22" font-weight="700">Response Length vs Inference Time</text>',
        f'<line x1="{left}" y1="{top + chart_h}" x2="{left + chart_w}" y2="{top + chart_h}" stroke="#111827"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + chart_h}" stroke="#111827"/>',
    ]
    for row in rows[:: max(1, len(rows) // 2500)]:
        x = left + (fnum(row.get("response_note_count")) / max_x) * chart_w
        y = top + chart_h - (fnum(row.get("total_generation_ms")) / max_y) * chart_h
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.2" fill="#2563eb" opacity="0.35"/>')
    parts.append(f'<text x="{left + chart_w / 2}" y="{height - 28}" text-anchor="middle" font-family="Arial" font-size="12">response_note_count</text>')
    parts.append(f'<text x="18" y="{top + chart_h / 2}" transform="rotate(-90 18 {top + chart_h / 2})" text-anchor="middle" font-family="Arial" font-size="12">total_generation_ms</text>')
    parts.append("</svg>")
    write_svg(chart_dir / "latency_response_length_vs_inference_time.svg", parts)


def write_reports(
    output_dir: Path,
    presets: Sequence[str],
    variants: Sequence[str],
    validation: Dict[str, object],
    variant_summary: Sequence[Dict[str, object]],
    module_rows: Sequence[Dict[str, object]],
    latency_summary_condition: Sequence[Dict[str, object]],
    preload_rows: Sequence[Dict[str, object]],
) -> None:
    lines = [
        "# Call100 Ablation Study",
        "",
        "## Reproducible Design",
        "",
        f"- Calls: `100` from `{DEFAULT_MANIFEST}`",
        f"- Presets: `{len(presets)}` ({', '.join(presets)})",
        f"- Ablation variants: `{len(variants)}` ({', '.join(str(VARIANT_BY_NAME[v]['short']) for v in variants)})",
        f"- Trials per preset/variant/call: `{validation.get('trials_per_combo')}`",
        f"- Expected samples: `{validation.get('expected_rows')}`",
        f"- Actual rows: `{validation.get('actual_rows')}`",
        f"- Validation status: `{validation.get('status')}`",
        "",
        "## Variant Definitions",
        "",
    ]
    for variant in variants:
        meta = VARIANT_BY_NAME[variant]
        lines.append(f"- `{meta['short']} {variant}`: {meta['description']}")
    lines.extend(["", "## Summary By Variant", ""])
    lines.extend(base.report_table(variant_summary, ["variant_short", "ablation_variant", "sample_count", "mean_objective_score", "mean_cpr", "mean_duration_match_ratio", "mean_psr", "mean_cadence_score"], 20))
    lines.extend(["", "## Module Contributions", ""])
    lines.extend(base.report_table([row for row in module_rows if row["group"] == "overall" and row["metric"] == "objective_score"], ["module_step", "module_added", "paired_sample_count", "mean_delta", "ci95_low", "ci95_high", "p_two_sided"], 20))
    lines.extend(["", "## Validation", ""])
    if validation.get("errors"):
        lines.extend([f"- ERROR: {error}" for error in validation["errors"]])
    else:
        lines.append("- No validation errors.")
    (output_dir / "ablation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    latency_lines = [
        "# Call100 Realtime Latency Logging Study",
        "",
        "## Design",
        "",
        "- L0 preload off: inference starts at endpoint commit.",
        "- L1 preload on: inference is modeled as starting during the candidate-endpoint confirmation window.",
        "- The logged inference timings come from actual local AMT generation during the A6 full-controlled ablation runs.",
        "- MIDI output timing is represented by the local playback scheduler model with the configured micro-buffer.",
        "",
        "## Summary By Condition",
        "",
    ]
    latency_lines.extend(base.report_table(latency_summary_condition, ["condition", "sample_count", "mean_latency_ms", "p50_latency_ms", "p95_latency_ms", "p99_latency_ms", "max_latency_ms", "underrun_rate", "mean_first_token_latency_ms", "mean_total_generation_ms"], 10))
    latency_lines.extend(["", "## Preload Comparison", ""])
    latency_lines.extend(base.report_table(preload_rows, ["comparison", "paired_sample_count", "mean_latency_reduction_ms", "ci95_low", "ci95_high", "p_two_sided"], 10))
    (output_dir / "latency_report.md").write_text("\n".join(latency_lines) + "\n", encoding="utf-8")


def write_all_outputs(args: argparse.Namespace, presets: Sequence[str], variants: Sequence[str], manifest: Sequence[Dict[str, str]], rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    output_dir = Path(args.output_dir)
    write_csv(output_dir / "ablation_all_results.csv", rows, ABLATION_FIELDS)
    write_csv(output_dir / "all_objective_results.csv", rows, ABLATION_FIELDS)

    variant_summary = summarize_group(rows, ["variant_short", "ablation_variant", "module_added"])
    preset_variant_summary = summarize_group(rows, ["preset", "variant_short", "ablation_variant"])
    leaderboard = sorted(preset_variant_summary, key=lambda row: fnum(row.get("mean_objective_score")), reverse=True)
    for idx, row in enumerate(leaderboard, start=1):
        row["rank"] = idx
    module_rows = module_contributions(rows, presets, variants, args.bootstrap_iterations)
    validation = validate_rows(rows, manifest, presets, variants, args.trials)
    write_csv(output_dir / "ablation_summary_by_variant.csv", variant_summary)
    write_csv(output_dir / "ablation_summary_by_preset_variant.csv", preset_variant_summary)
    write_csv(output_dir / "ablation_leaderboard.csv", leaderboard)
    write_csv(output_dir / "ablation_module_contribution.csv", module_rows)
    (output_dir / "ablation_validation_summary.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    latency_events = read_latency_jsonl(output_dir / "latency_events.jsonl")
    write_csv(output_dir / "latency_events.csv", latency_events)
    latency_trials = build_latency_trials(output_dir, args.preload_window_ms, args.micro_buffer_ms, args.underrun_deadline_ms)
    write_csv(output_dir / "latency_log_all_trials.csv", latency_trials, LATENCY_FIELDS)
    latency_by_condition = latency_summary(latency_trials, ["condition"])
    latency_by_length = latency_summary(latency_trials, ["condition", "length_bin"])
    preload_rows = preload_comparison(latency_trials)
    write_csv(output_dir / "latency_summary_by_condition.csv", latency_by_condition)
    write_csv(output_dir / "latency_summary_by_length_bin.csv", latency_by_length)
    write_csv(output_dir / "preload_on_off_comparison.csv", preload_rows)

    chart_dir = output_dir / "charts"
    chart_ablation_bar(variant_summary, chart_dir)
    chart_module_heatmap(module_rows, chart_dir)
    chart_latency_box(latency_trials, chart_dir)
    chart_latency_table(latency_by_condition, chart_dir)
    chart_length_scatter(latency_trials, chart_dir)

    write_reports(output_dir, presets, variants, validation, variant_summary, module_rows, latency_by_condition, preload_rows)
    return validation


def write_plan(output_dir: Path, args: argparse.Namespace, presets: Sequence[str], variants: Sequence[str]) -> None:
    plan = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "manifest": args.manifest,
        "output_dir": str(output_dir),
        "trials": args.trials,
        "seed": args.seed,
        "presets": list(presets),
        "variants": [VARIANT_BY_NAME[variant] for variant in variants],
        "latency": {
            "conditions": ["L0_preload_off", "L1_preload_on"],
            "preload_window_ms": args.preload_window_ms,
            "micro_buffer_ms": args.micro_buffer_ms,
            "underrun_deadline_ms": args.underrun_deadline_ms,
        },
    }
    (output_dir / "experiment_design.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(output_dir / "ablation_variant_design.csv", [VARIANT_BY_NAME[variant] for variant in variants])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run formal Call100 ablation and latency logging experiments.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--trials", type=int, default=15)
    parser.add_argument("--seed", type=int, default=20260622)
    parser.add_argument("--presets", default=",".join(DEFAULT_PRESETS))
    parser.add_argument("--variants", default=",".join(DEFAULT_VARIANTS))
    parser.add_argument("--max-calls", type=int, default=None)
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--bpm", type=float, default=100.0)
    parser.add_argument("--pentatonic-root", type=int, default=60)
    parser.add_argument("--pentatonic-mode", choices=["major", "minor"], default="major")
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--preload-window-ms", type=float, default=250.0)
    parser.add_argument("--micro-buffer-ms", type=float, default=80.0)
    parser.add_argument("--underrun-deadline-ms", type=float, default=80.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    if args.force and output_dir.exists() and not args.aggregate_only:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    presets = selected(args.presets, DEFAULT_PRESETS, "preset")
    variants = selected(args.variants, DEFAULT_VARIANTS, "variant")
    manifest = base.manifest_rows(Path(args.manifest), args.max_calls)
    input_dir = base.prepare_call_inputs(manifest, output_dir)
    write_plan(output_dir, args, presets, variants)
    expected = len(manifest) * len(presets) * len(variants) * args.trials
    print(f"[setup] manifest={args.manifest}")
    print(f"[setup] output_dir={output_dir}")
    print(f"[setup] calls={len(manifest)} presets={len(presets)} variants={len(variants)} trials={args.trials}")
    print(f"[setup] expected_samples={expected}")
    print(f"[setup] latency_log={output_dir / 'latency_events.jsonl'}")
    run_generation(args, presets, variants, input_dir, len(manifest))
    print("[aggregate] building ablation tables, latency tables, charts, and reports")
    rows = aggregate_results(args, presets, variants, manifest)
    validation = write_all_outputs(args, presets, variants, manifest, rows)
    if validation["errors"]:
        print("[validate] failed")
        for error in validation["errors"]:
            print(f"[validate] ERROR {error}")
        raise SystemExit(1)
    print("[validate] passed")
    print(f"[done] ablation_all_results={output_dir / 'ablation_all_results.csv'}")
    print(f"[done] latency_log_all_trials={output_dir / 'latency_log_all_trials.csv'}")
    print(f"[done] ablation_report={output_dir / 'ablation_report.md'}")
    print(f"[done] latency_report={output_dir / 'latency_report.md'}")


if __name__ == "__main__":
    main()
