from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Dict, Iterable, List, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = ROOT / "code"
AB_DIR = ROOT / "ab_tests"
DEFAULT_MANIFEST = AB_DIR / "calls_100_public_final" / "call100_manifest.csv"
DEFAULT_OUTPUT_DIR = AB_DIR / "objective_search_call100_trials15"

PRESETS: Dict[str, Dict[str, object]] = {
    "pentatonic_no_theory": {
        "style": "pentatonic_2bar_4_4",
        "theory_control": False,
        "top_p": 0.98,
        "temperature": 0.9,
        "max_melodic_leap": 0,
    },
    "pentatonic_balanced": {
        "style": "pentatonic_2bar_4_4",
        "theory_control": True,
        "top_p": 0.95,
        "temperature": 0.85,
        "max_melodic_leap": 7,
        "cadence_degree": "root",
    },
    "pentatonic_conservative": {
        "style": "pentatonic_2bar_4_4",
        "theory_control": True,
        "top_p": 0.90,
        "temperature": 0.75,
        "max_melodic_leap": 5,
        "cadence_degree": "root",
    },
    "pentatonic_creative": {
        "style": "pentatonic_2bar_4_4",
        "theory_control": True,
        "top_p": 0.98,
        "temperature": 1.00,
        "max_melodic_leap": 9,
        "cadence_degree": "root",
    },
    "pentatonic_creative_wide": {
        "style": "pentatonic_2bar_4_4",
        "theory_control": True,
        "top_p": 0.99,
        "temperature": 1.10,
        "max_melodic_leap": 12,
        "cadence_degree": "root",
    },
    "pentatonic_low_temp_no_strongbeat": {
        "style": "pentatonic_2bar_4_4",
        "theory_control": True,
        "strong_beat_stable": False,
        "top_p": 0.92,
        "temperature": 0.75,
        "max_melodic_leap": 7,
        "cadence_degree": "root",
    },
}

DEFAULT_PRESET_NAMES = list(PRESETS)
DEFAULT_CANDIDATES = [
    "amt_small_raw",
    "amt_small_controlled",
    "motif_transform_baseline",
]

ALL_RESULT_FIELDS = [
    "call_id",
    "origin",
    "source_dataset",
    "category",
    "sub_category",
    "preset",
    "candidate",
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
    "qualified_note_rate",
    "qualified_rhythm_rate",
    "groove_similarity",
    "cadence_score",
    "max_abs_interval",
    "note_count",
    "raw_response_seconds",
    "actual_response_seconds",
    "duration_match_ratio",
    "duration_stretch_factor",
    "fallback_used",
    "prompt_cleaning_applied",
    "repetition_resample_count",
    "response_midi_path",
]

METRICS = [
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
    "qualified_note_rate",
    "qualified_rhythm_rate",
    "groove_similarity",
    "cadence_score",
    "max_abs_interval",
    "note_count",
    "duration_match_ratio",
]


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Dict[str, object]], fields: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        ordered: List[str] = []
        for row in rows:
            for key in row:
                if key not in ordered:
                    ordered.append(key)
        fields = ordered
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: format_value(row.get(field, "")) for field in fields})


def format_value(value: object) -> object:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return ""
        return f"{value:.6f}"
    return value


def fnum(value: object, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def boolish(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def selected_names(raw: str | None, default: Sequence[str], known: Dict[str, object] | None = None) -> List[str]:
    if raw:
        names = [item.strip() for item in raw.split(",") if item.strip()]
    else:
        names = list(default)
    if known is not None:
        unknown = [name for name in names if name not in known]
        if unknown:
            raise SystemExit(f"Unknown preset(s): {unknown}. Available: {sorted(known)}")
    return names


def bool_flag(name: str, value: bool) -> List[str]:
    return [f"--{name}"] if value else [f"--no-{name}"]


def apply_preset(command: List[str], preset: Dict[str, object]) -> List[str]:
    result = list(command)
    for key, value in preset.items():
        option = key.replace("_", "-")
        if key == "theory_control":
            result.extend(bool_flag("theory-control", bool(value)))
        elif key == "strong_beat_stable":
            result.extend(bool_flag("strong-beat-stable", bool(value)))
        else:
            result.extend([f"--{option}", str(value)])
    return result


def manifest_rows(manifest: Path, max_calls: int | None = None) -> List[Dict[str, str]]:
    rows = read_csv(manifest)
    if max_calls is not None:
        rows = rows[:max_calls]
    if not rows:
        raise SystemExit(f"No rows in manifest: {manifest}")
    ids = [row.get("call_id", "") for row in rows]
    dupes = [key for key, count in Counter(ids).items() if count > 1]
    if dupes:
        raise SystemExit(f"Duplicate call_id in manifest: {dupes[:10]}")
    return rows


def prepare_call_inputs(rows: Sequence[Dict[str, str]], output_dir: Path) -> Path:
    input_dir = output_dir / "call_inputs"
    if input_dir.exists():
        shutil.rmtree(input_dir)
    input_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        call_id = row["call_id"]
        source = Path(row["midi_path"])
        if not source.exists():
            raise FileNotFoundError(f"Missing MIDI for {call_id}: {source}")
        shutil.copy2(source, input_dir / f"{call_id}.mid")
    return input_dir


def count_csv_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return max(0, sum(1 for _ in handle) - 1)


def preset_complete(run_dir: Path, expected_rows: int) -> bool:
    metrics = run_dir / "objective_metrics.csv"
    answer_key = run_dir / "answer_key.csv"
    responses = run_dir / "responses"
    if count_csv_rows(metrics) != expected_rows:
        return False
    if count_csv_rows(answer_key) != expected_rows:
        return False
    if not responses.exists():
        return False
    response_count = len([p for p in responses.iterdir() if p.suffix.lower() in {".mid", ".midi"}])
    return response_count == expected_rows


def run_logged(command: List[str], cwd: Path, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\n[command] " + " ".join(command) + "\n")
        log.flush()
        proc = subprocess.run(command, cwd=str(cwd), stdout=log, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, command)


def run_generation(
    args: argparse.Namespace,
    preset_names: Sequence[str],
    candidate_names: Sequence[str],
    input_dir: Path,
    call_count: int,
) -> None:
    expected_per_preset = call_count * args.trials * len(candidate_names)
    for preset_index, preset_name in enumerate(preset_names, start=1):
        run_dir = Path(args.output_dir) / preset_name
        if args.aggregate_only:
            continue
        if run_dir.exists() and args.force:
            shutil.rmtree(run_dir)
        if preset_complete(run_dir, expected_per_preset):
            print(f"[skip] {preset_name}: already complete ({expected_per_preset} responses)")
            continue
        if run_dir.exists():
            shutil.rmtree(run_dir)
        preset_seed = args.seed + preset_index * 10_000_000
        command = [
            sys.executable,
            str(CODE_DIR / "offline_ab_test.py"),
            "--input-dir",
            str(input_dir),
            "--trials",
            str(args.trials),
            "--seed",
            str(preset_seed),
            "--output-dir",
            str(run_dir),
            "--candidates",
            ",".join(candidate_names),
            "--offline",
            "--bpm",
            str(args.bpm),
            "--pentatonic-root",
            str(args.pentatonic_root),
            "--pentatonic-mode",
            args.pentatonic_mode,
        ]
        command = apply_preset(command, PRESETS[preset_name])
        log_path = Path(args.output_dir) / "logs" / f"{preset_name}.log"
        print(f"[run] {preset_name}: generating {expected_per_preset} responses seed={preset_seed}")
        run_logged(command, ROOT, log_path)
        eval_command = [
            sys.executable,
            str(CODE_DIR / "evaluate_melody_metrics.py"),
            "--run-dir",
            str(run_dir),
            "--scale-root",
            str(args.pentatonic_root),
            "--scale-mode",
            args.pentatonic_mode,
            "--beats-per-bar",
            "4",
        ]
        print(f"[eval] {preset_name}")
        run_logged(eval_command, ROOT, log_path)
        if not preset_complete(run_dir, expected_per_preset):
            raise RuntimeError(f"Preset did not complete cleanly: {preset_name}")


def duration_ratio(actual: float, target: float) -> Tuple[float, float]:
    if actual <= 0 or target <= 0:
        return 0.0, 0.0
    return min(actual, target) / max(actual, target), actual / target


def call_prompt_cleaning_applied(row: Dict[str, str]) -> bool:
    return boolish(row.get("has_repetition_tail", ""))


def macro_category(row: Dict[str, str]) -> str:
    if row.get("split") == "base_call50":
        return f"base_{row.get('origin', 'unknown')}"
    category = row.get("category", "")
    if "chord" in category or "polyphonic" in category:
        return "public_polyphonic"
    if "dense" in category or "fast" in category:
        return "public_dense_fast"
    if "chromatic" in category or "outside" in category:
        return "public_chromatic"
    if "sparse" in category or "pause" in category or "false_ending" in category:
        return "public_sparse_pause"
    if "repetition" in category:
        return "public_repetition"
    if "expressive" in category or "rubato" in category:
        return "public_expressive"
    return "public_melody"


def aggregate_all_results(
    args: argparse.Namespace,
    preset_names: Sequence[str],
    candidate_names: Sequence[str],
    manifest: Sequence[Dict[str, str]],
) -> List[Dict[str, object]]:
    manifest_by_id = {row["call_id"]: row for row in manifest}
    rows: List[Dict[str, object]] = []
    for preset_name in preset_names:
        run_dir = Path(args.output_dir) / preset_name
        answers = {row["sample_id"]: row for row in read_csv(run_dir / "answer_key.csv")}
        metrics_rows = read_csv(run_dir / "objective_metrics.csv")
        for metric in metrics_rows:
            sample_id = metric["sample_id"]
            answer = answers.get(sample_id, {})
            call_id = answer.get("call_id") or metric.get("call_id")
            if call_id not in manifest_by_id:
                raise RuntimeError(f"Unknown call_id in {preset_name}: {call_id}")
            source = manifest_by_id[call_id]
            raw_seconds = fnum(answer.get("response_seconds"))
            actual_seconds = fnum(metric.get("duration_seconds"))
            match_ratio, stretch_factor = duration_ratio(actual_seconds, raw_seconds)
            fallback_count = int(fnum(answer.get("fallback_count"), 0))
            generation_error = answer.get("generation_error", "")
            out = {
                "call_id": call_id,
                "origin": source.get("origin", ""),
                "source_dataset": source.get("source_dataset", ""),
                "category": source.get("category", ""),
                "sub_category": source.get("sub_category", ""),
                "preset": preset_name,
                "candidate": answer.get("candidate") or metric.get("candidate", ""),
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
                "qualified_note_rate": fnum(metric.get("qualified_note_rate")),
                "qualified_rhythm_rate": fnum(metric.get("qualified_rhythm_rate")),
                "groove_similarity": fnum(metric.get("groove_similarity")),
                "cadence_score": fnum(metric.get("cadence_score")),
                "max_abs_interval": fnum(metric.get("max_abs_interval")),
                "note_count": fnum(metric.get("note_count")),
                "raw_response_seconds": raw_seconds,
                "actual_response_seconds": actual_seconds,
                "duration_match_ratio": match_ratio,
                "duration_stretch_factor": stretch_factor,
                "fallback_used": fallback_count > 0 or bool(generation_error),
                "prompt_cleaning_applied": call_prompt_cleaning_applied(source),
                "repetition_resample_count": int(fnum(answer.get("rejected_repeat_count"), 0)),
                "response_midi_path": answer.get("response_only_midi") or metric.get("midi_path", ""),
            }
            if out["candidate"] not in candidate_names:
                raise RuntimeError(f"Unexpected candidate in metrics: {out['candidate']}")
            rows.append(out)
    return rows


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lower = int(math.floor(pos))
    upper = int(math.ceil(pos))
    if lower == upper:
        return ordered[lower]
    weight = pos - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def stddev(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mu = mean(values)
    return math.sqrt(sum((v - mu) ** 2 for v in values) / (len(values) - 1))


def sem_ci(values: Sequence[float]) -> Tuple[float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0
    mu = mean(values)
    se = stddev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0
    return mu, mu - 1.96 * se, mu + 1.96 * se


def summarize_group(rows: Sequence[Dict[str, object]], group_fields: Sequence[str]) -> List[Dict[str, object]]:
    groups: Dict[Tuple[object, ...], List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(field, "") for field in group_fields)].append(row)
    summaries: List[Dict[str, object]] = []
    for key, items in groups.items():
        out: Dict[str, object] = {field: key[idx] for idx, field in enumerate(group_fields)}
        scores = [fnum(item.get("objective_score")) for item in items]
        out["sample_count"] = len(items)
        out["mean_objective_score"], out["ci95_low_objective_score"], out["ci95_high_objective_score"] = sem_ci(scores)
        out["median_objective_score"] = median(scores) if scores else 0.0
        out["std_objective_score"] = stddev(scores)
        for metric in METRICS:
            vals = [fnum(item.get(metric)) for item in items]
            if vals:
                out[f"mean_{metric}"] = mean(vals)
        summaries.append(out)
    summaries.sort(key=lambda row: fnum(row.get("mean_objective_score")), reverse=True)
    for rank, row in enumerate(summaries, start=1):
        row["rank"] = rank
    return summaries


def add_macro_category(rows: Sequence[Dict[str, object]], manifest: Sequence[Dict[str, str]]) -> List[Dict[str, object]]:
    by_id = {row["call_id"]: row for row in manifest}
    enriched = []
    for row in rows:
        copy = dict(row)
        copy["macro_category"] = macro_category(by_id[str(row["call_id"])])
        enriched.append(copy)
    return enriched


def paired_differences(rows: Sequence[Dict[str, object]], candidate_a: str, candidate_b: str, preset: str | None) -> List[float]:
    buckets: Dict[Tuple[str, str, int], Dict[str, float]] = defaultdict(dict)
    for row in rows:
        if preset is not None and row.get("preset") != preset:
            continue
        candidate = str(row.get("candidate", ""))
        if candidate not in {candidate_a, candidate_b}:
            continue
        key = (str(row.get("preset", "")), str(row.get("call_id", "")), int(fnum(row.get("trial"))))
        buckets[key][candidate] = fnum(row.get("objective_score"))
    diffs = []
    for values in buckets.values():
        if candidate_a in values and candidate_b in values:
            diffs.append(values[candidate_a] - values[candidate_b])
    return diffs


def bootstrap_ci(diffs: Sequence[float], iterations: int, seed: int) -> Tuple[float, float, float, float]:
    if not diffs:
        return 0.0, 0.0, 0.0, 1.0
    observed = mean(diffs)
    rng = random.Random(seed)
    n = len(diffs)
    samples = []
    for _ in range(iterations):
        total = 0.0
        for _ in range(n):
            total += diffs[rng.randrange(n)]
        samples.append(total / n)
    low = percentile(samples, 0.025)
    high = percentile(samples, 0.975)
    le_zero = sum(1 for value in samples if value <= 0) / len(samples)
    ge_zero = sum(1 for value in samples if value >= 0) / len(samples)
    p_two_sided = min(1.0, 2.0 * min(le_zero, ge_zero))
    return observed, low, high, p_two_sided


def write_pairwise(rows: Sequence[Dict[str, object]], preset_names: Sequence[str], output_dir: Path, iterations: int) -> List[Dict[str, object]]:
    comparisons = [
        ("controlled_minus_raw", "amt_small_controlled", "amt_small_raw"),
        ("controlled_minus_motif", "amt_small_controlled", "motif_transform_baseline"),
        ("motif_minus_raw", "motif_transform_baseline", "amt_small_raw"),
    ]
    result: List[Dict[str, object]] = []
    groups: List[Tuple[str, str | None]] = [("overall", None)] + [(preset, preset) for preset in preset_names]
    seed_base = 20260618
    for group_index, (group_name, preset) in enumerate(groups):
        for comp_index, (label, a, b) in enumerate(comparisons):
            diffs = paired_differences(rows, a, b, preset)
            observed, low, high, p = bootstrap_ci(diffs, iterations, seed_base + group_index * 100 + comp_index)
            result.append(
                {
                    "group": group_name,
                    "comparison": label,
                    "candidate_a": a,
                    "candidate_b": b,
                    "paired_sample_count": len(diffs),
                    "mean_diff_a_minus_b": observed,
                    "ci95_low": low,
                    "ci95_high": high,
                    "p_two_sided_bootstrap": p,
                    "bootstrap_iterations": iterations,
                }
            )
    write_csv(output_dir / "pairwise_bootstrap_tests.csv", result)
    return result


def write_failure_cases(rows: Sequence[Dict[str, object]], output_dir: Path) -> List[Dict[str, object]]:
    scores = [fnum(row.get("objective_score")) for row in rows]
    threshold = percentile(scores, 0.05)
    failures: List[Dict[str, object]] = []
    for row in rows:
        reasons = []
        if boolish(row.get("fallback_used")) or row.get("fallback_used") is True:
            reasons.append("fallback_used")
        if fnum(row.get("objective_score")) <= threshold:
            reasons.append("bottom_5pct_objective_score")
        if fnum(row.get("note_count")) < 3:
            reasons.append("very_low_note_count")
        if fnum(row.get("duration_match_ratio")) < 0.55:
            reasons.append("poor_duration_match")
        if not Path(str(row.get("response_midi_path", ""))).exists():
            reasons.append("missing_response_midi")
        if reasons:
            failures.append(
                {
                    "reason": ";".join(reasons),
                    "call_id": row.get("call_id"),
                    "origin": row.get("origin"),
                    "source_dataset": row.get("source_dataset"),
                    "category": row.get("category"),
                    "sub_category": row.get("sub_category"),
                    "preset": row.get("preset"),
                    "candidate": row.get("candidate"),
                    "trial": row.get("trial"),
                    "seed": row.get("seed"),
                    "objective_score": row.get("objective_score"),
                    "duration_match_ratio": row.get("duration_match_ratio"),
                    "note_count": row.get("note_count"),
                    "response_midi_path": row.get("response_midi_path"),
                }
            )
    failures.sort(key=lambda item: fnum(item.get("objective_score")))
    write_csv(output_dir / "failure_cases.csv", failures)
    return failures


def validate_all_results(
    rows: Sequence[Dict[str, object]],
    manifest: Sequence[Dict[str, str]],
    preset_names: Sequence[str],
    candidate_names: Sequence[str],
    trials: int,
) -> Dict[str, object]:
    errors: List[str] = []
    manifest_ids = {row["call_id"] for row in manifest}
    expected_rows = len(manifest_ids) * len(preset_names) * len(candidate_names) * trials
    if len(rows) != expected_rows:
        errors.append(f"all_objective_results.csv row count expected {expected_rows}, got {len(rows)}")
    combos = Counter((row.get("preset"), row.get("candidate"), row.get("call_id")) for row in rows)
    expected_combo_count = len(preset_names) * len(candidate_names) * len(manifest_ids)
    if len(combos) != expected_combo_count:
        errors.append(f"preset/candidate/call_id combo count expected {expected_combo_count}, got {len(combos)}")
    bad_combo = [key for key, count in combos.items() if count != trials]
    if bad_combo:
        errors.append(f"{len(bad_combo)} preset/candidate/call_id combos do not have {trials} rows")
    empty_scores = sum(1 for row in rows if row.get("objective_score") in {"", None})
    if empty_scores:
        errors.append(f"objective_score has {empty_scores} empty values")
    missing_midis = sum(1 for row in rows if not Path(str(row.get("response_midi_path", ""))).exists())
    if missing_midis:
        errors.append(f"response_midi_path missing for {missing_midis} rows")
    unknown_calls = sorted({str(row.get("call_id")) for row in rows} - manifest_ids)
    if unknown_calls:
        errors.append(f"rows contain call_id values not in manifest: {unknown_calls[:10]}")
    missing_calls = sorted(manifest_ids - {str(row.get("call_id")) for row in rows})
    if missing_calls:
        errors.append(f"manifest call_id values missing from results: {missing_calls[:10]}")
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "expected_rows": expected_rows,
        "actual_rows": len(rows),
        "expected_combo_count": expected_combo_count,
        "actual_combo_count": len(combos),
        "trials_per_combo": trials,
        "empty_objective_scores": empty_scores,
        "missing_response_midis": missing_midis,
        "unknown_call_ids": unknown_calls,
        "missing_call_ids": missing_calls,
        "errors": errors,
        "status": "passed" if not errors else "failed",
    }
    return payload


def svg_escape(text: object) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def write_svg(path: Path, parts: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def chart_top_scores(leaderboard: Sequence[Dict[str, object]], chart_dir: Path) -> None:
    rows = list(leaderboard)[:10]
    width, row_h, left, right, top = 1250, 42, 500, 80, 70
    height = top + row_h * len(rows) + 60
    bar_w = width - left - right
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="28" y="38" font-family="Arial" font-size="24" font-weight="700">Top Objective Scores - Call100</text>',
    ]
    for idx, row in enumerate(rows):
        y = top + idx * row_h
        score = fnum(row.get("mean_objective_score"))
        label = f"#{idx + 1} {row.get('preset')} / {row.get('candidate')}"
        parts.append(f'<text x="28" y="{y + 24}" font-family="Arial" font-size="14">{svg_escape(label)}</text>')
        parts.append(f'<rect x="{left}" y="{y + 8}" width="{bar_w}" height="22" fill="#e5e7eb"/>')
        parts.append(f'<rect x="{left}" y="{y + 8}" width="{bar_w * max(0, min(1, score)):.2f}" height="22" fill="#2563eb"/>')
        parts.append(f'<text x="{left + bar_w + 12}" y="{y + 24}" font-family="Arial" font-size="13">{score:.3f}</text>')
    parts.append("</svg>")
    write_svg(chart_dir / "top_objective_scores_call100.svg", parts)


def chart_candidate_ci(summary: Sequence[Dict[str, object]], chart_dir: Path) -> None:
    rows = list(summary)
    width, height = 900, 420
    left, top, chart_w, chart_h = 90, 70, 720, 250
    max_score = 1.0
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="30" y="36" font-family="Arial" font-size="23" font-weight="700">Candidate Mean Score with 95% CI</text>',
    ]
    slot = chart_w / max(1, len(rows))
    colors = ["#2563eb", "#16a34a", "#dc2626"]
    for i in range(6):
        y = top + chart_h - i * chart_h / 5
        parts.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + chart_w}" y2="{y:.2f}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="50" y="{y + 4:.2f}" font-family="Arial" font-size="12">{i / 5:.1f}</text>')
    for idx, row in enumerate(rows):
        cx = left + slot * idx + slot / 2
        mean_v = fnum(row.get("mean_objective_score"))
        low = fnum(row.get("ci95_low_objective_score"))
        high = fnum(row.get("ci95_high_objective_score"))
        bar_h = chart_h * mean_v / max_score
        y = top + chart_h - bar_h
        ci_y1 = top + chart_h - chart_h * high / max_score
        ci_y2 = top + chart_h - chart_h * low / max_score
        parts.append(f'<rect x="{cx - 45}" y="{y:.2f}" width="90" height="{bar_h:.2f}" fill="{colors[idx % len(colors)]}"/>')
        parts.append(f'<line x1="{cx}" y1="{ci_y1:.2f}" x2="{cx}" y2="{ci_y2:.2f}" stroke="#111827" stroke-width="2"/>')
        parts.append(f'<line x1="{cx - 12}" y1="{ci_y1:.2f}" x2="{cx + 12}" y2="{ci_y1:.2f}" stroke="#111827" stroke-width="2"/>')
        parts.append(f'<line x1="{cx - 12}" y1="{ci_y2:.2f}" x2="{cx + 12}" y2="{ci_y2:.2f}" stroke="#111827" stroke-width="2"/>')
        parts.append(f'<text x="{cx}" y="{top + chart_h + 28}" text-anchor="middle" font-family="Arial" font-size="12">{svg_escape(row.get("candidate"))}</text>')
        parts.append(f'<text x="{cx}" y="{y - 8:.2f}" text-anchor="middle" font-family="Arial" font-size="12">{mean_v:.3f}</text>')
    parts.append("</svg>")
    write_svg(chart_dir / "candidate_mean_score_with_ci.svg", parts)


def chart_controlled_vs_raw(pairwise: Sequence[Dict[str, object]], chart_dir: Path) -> None:
    rows = [row for row in pairwise if row.get("comparison") == "controlled_minus_raw" and row.get("group") != "overall"]
    width, height = 1150, 430
    left, top, chart_w, chart_h = 110, 70, 930, 250
    vals = [fnum(row.get("mean_diff_a_minus_b")) for row in rows]
    max_abs = max([0.02] + [abs(v) for v in vals])
    zero_y = top + chart_h / 2
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="30" y="36" font-family="Arial" font-size="23" font-weight="700">Controlled vs Raw by Preset</text>',
        f'<line x1="{left}" y1="{zero_y}" x2="{left + chart_w}" y2="{zero_y}" stroke="#111827"/>',
    ]
    slot = chart_w / max(1, len(rows))
    for idx, row in enumerate(rows):
        diff = fnum(row.get("mean_diff_a_minus_b"))
        low = fnum(row.get("ci95_low"))
        high = fnum(row.get("ci95_high"))
        cx = left + slot * idx + slot / 2
        y = zero_y - diff / max_abs * (chart_h / 2)
        low_y = zero_y - low / max_abs * (chart_h / 2)
        high_y = zero_y - high / max_abs * (chart_h / 2)
        color = "#16a34a" if diff >= 0 else "#dc2626"
        parts.append(f'<line x1="{cx}" y1="{high_y:.2f}" x2="{cx}" y2="{low_y:.2f}" stroke="#111827" stroke-width="2"/>')
        parts.append(f'<circle cx="{cx}" cy="{y:.2f}" r="7" fill="{color}"/>')
        parts.append(f'<text x="{cx}" y="{top + chart_h + 28}" transform="rotate(25 {cx} {top + chart_h + 28})" font-family="Arial" font-size="11">{svg_escape(row.get("group"))}</text>')
        parts.append(f'<text x="{cx}" y="{y - 12:.2f}" text-anchor="middle" font-family="Arial" font-size="11">{diff:.3f}</text>')
    parts.append("</svg>")
    write_svg(chart_dir / "controlled_vs_raw_by_preset.svg", parts)


def chart_boxplot(rows: Sequence[Dict[str, object]], candidate_names: Sequence[str], chart_dir: Path) -> None:
    width, height = 900, 460
    left, top, chart_w, chart_h = 90, 70, 720, 280
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="30" y="36" font-family="Arial" font-size="23" font-weight="700">Objective Score Boxplot by Candidate</text>',
    ]
    for i in range(6):
        y = top + chart_h - i * chart_h / 5
        parts.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + chart_w}" y2="{y:.2f}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="50" y="{y + 4:.2f}" font-family="Arial" font-size="12">{i / 5:.1f}</text>')
    slot = chart_w / max(1, len(candidate_names))
    for idx, candidate in enumerate(candidate_names):
        scores = [fnum(row.get("objective_score")) for row in rows if row.get("candidate") == candidate]
        q1, q2, q3 = percentile(scores, 0.25), percentile(scores, 0.50), percentile(scores, 0.75)
        lo, hi = percentile(scores, 0.05), percentile(scores, 0.95)
        cx = left + slot * idx + slot / 2
        def y_of(v: float) -> float:
            return top + chart_h - chart_h * max(0, min(1, v))
        parts.append(f'<line x1="{cx}" y1="{y_of(hi):.2f}" x2="{cx}" y2="{y_of(lo):.2f}" stroke="#111827" stroke-width="2"/>')
        parts.append(f'<rect x="{cx - 48}" y="{y_of(q3):.2f}" width="96" height="{max(1, y_of(q1) - y_of(q3)):.2f}" fill="#bfdbfe" stroke="#1d4ed8"/>')
        parts.append(f'<line x1="{cx - 48}" y1="{y_of(q2):.2f}" x2="{cx + 48}" y2="{y_of(q2):.2f}" stroke="#1d4ed8" stroke-width="3"/>')
        parts.append(f'<text x="{cx}" y="{top + chart_h + 30}" text-anchor="middle" font-family="Arial" font-size="12">{svg_escape(candidate)}</text>')
    parts.append("</svg>")
    write_svg(chart_dir / "objective_score_boxplot_by_candidate.svg", parts)


def chart_category_heatmap(rows: Sequence[Dict[str, object]], candidate_names: Sequence[str], chart_dir: Path) -> None:
    categories = sorted({str(row.get("category")) for row in rows})
    cell_w, cell_h = 165, 26
    left, top = 300, 70
    width = left + cell_w * len(candidate_names) + 80
    height = top + cell_h * len(categories) + 70
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="30" y="36" font-family="Arial" font-size="23" font-weight="700">Category Performance Heatmap</text>',
    ]
    for j, candidate in enumerate(candidate_names):
        parts.append(f'<text x="{left + j * cell_w + cell_w / 2}" y="{top - 16}" text-anchor="middle" font-family="Arial" font-size="12">{svg_escape(candidate)}</text>')
    for i, category in enumerate(categories):
        y = top + i * cell_h
        parts.append(f'<text x="{left - 12}" y="{y + 18}" text-anchor="end" font-family="Arial" font-size="12">{svg_escape(category)}</text>')
        for j, candidate in enumerate(candidate_names):
            vals = [fnum(row.get("objective_score")) for row in rows if row.get("category") == category and row.get("candidate") == candidate]
            value = mean(vals) if vals else 0.0
            intensity = int(255 - 120 * max(0, min(1, value)))
            color = f"rgb({intensity},{min(255, intensity + 35)},255)"
            x = left + j * cell_w
            parts.append(f'<rect x="{x}" y="{y}" width="{cell_w - 2}" height="{cell_h - 2}" fill="{color}" stroke="#ffffff"/>')
            parts.append(f'<text x="{x + cell_w / 2}" y="{y + 17}" text-anchor="middle" font-family="Arial" font-size="11">{value:.3f}</text>')
    parts.append("</svg>")
    write_svg(chart_dir / "category_performance_heatmap.svg", parts)


def chart_metric_breakdown(leaderboard: Sequence[Dict[str, object]], chart_dir: Path) -> None:
    rows = list(leaderboard)[:6]
    metric_keys = [
        "mean_tonality_score",
        "mean_rhythm_score",
        "mean_interval_score",
        "mean_repetition_score",
        "mean_pitch_diversity_score",
        "mean_compression_score",
    ]
    metric_labels = ["Tonality", "Rhythm", "Interval", "Repetition", "PitchDiv", "Compression"]
    width, height = 1280, 720
    left, top, chart_w, chart_h = 90, 90, 850, 450
    colors = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2"]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="30" y="40" font-family="Arial" font-size="24" font-weight="700">Metric Breakdown for Top Settings - Call100</text>',
    ]
    for i in range(6):
        y = top + chart_h - i * chart_h / 5
        parts.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + chart_w}" y2="{y:.2f}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="50" y="{y + 4:.2f}" font-family="Arial" font-size="12">{i / 5:.1f}</text>')
    group_w = chart_w / len(metric_keys)
    bar_w = group_w / (len(rows) + 1)
    for metric_idx, metric in enumerate(metric_keys):
        base_x = left + metric_idx * group_w
        parts.append(f'<text x="{base_x + 8:.2f}" y="{top + chart_h + 28}" font-family="Arial" font-size="12">{metric_labels[metric_idx]}</text>')
        for row_idx, row in enumerate(rows):
            val = fnum(row.get(metric))
            x = base_x + 6 + row_idx * bar_w
            h = chart_h * max(0, min(1, val))
            y = top + chart_h - h
            parts.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_w * 0.82:.2f}" height="{h:.2f}" fill="{colors[row_idx % len(colors)]}"/>')
    legend_x, legend_y = 970, 95
    for idx, row in enumerate(rows):
        y = legend_y + idx * 48
        label = f"#{idx + 1} {row.get('preset')}/{row.get('candidate')}"
        parts.append(f'<rect x="{legend_x}" y="{y}" width="16" height="16" fill="{colors[idx % len(colors)]}"/>')
        parts.append(f'<text x="{legend_x + 24}" y="{y + 13}" font-family="Arial" font-size="12">{svg_escape(label)}</text>')
    parts.append("</svg>")
    write_svg(chart_dir / "metric_breakdown_top_settings_call100.svg", parts)


def chart_top2_difference(leaderboard: Sequence[Dict[str, object]], chart_dir: Path) -> None:
    rows = list(leaderboard)[:2]
    if len(rows) < 2:
        return
    metric_keys = [
        "mean_tonality_score",
        "mean_rhythm_score",
        "mean_interval_score",
        "mean_repetition_score",
        "mean_pitch_diversity_score",
        "mean_compression_score",
    ]
    labels = ["Tonality", "Rhythm", "Interval", "Repetition", "PitchDiv", "Compression"]
    diffs = [fnum(rows[0].get(key)) - fnum(rows[1].get(key)) for key in metric_keys]
    max_abs = max([0.01] + [abs(v) for v in diffs])
    width, height = 1000, 430
    left, top, chart_w, chart_h = 180, 70, 720, 250
    zero_x = left + chart_w / 2
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="30" y="36" font-family="Arial" font-size="23" font-weight="700">Top 2 Metric Difference - Call100</text>',
        f'<text x="30" y="58" font-family="Arial" font-size="12">{svg_escape(rows[0].get("preset"))}/{svg_escape(rows[0].get("candidate"))} minus {svg_escape(rows[1].get("preset"))}/{svg_escape(rows[1].get("candidate"))}</text>',
        f'<line x1="{zero_x}" y1="{top}" x2="{zero_x}" y2="{top + chart_h}" stroke="#111827"/>',
    ]
    row_h = chart_h / len(labels)
    for idx, label in enumerate(labels):
        y = top + idx * row_h + row_h / 2
        diff = diffs[idx]
        x = zero_x if diff >= 0 else zero_x + diff / max_abs * (chart_w / 2)
        w = abs(diff) / max_abs * (chart_w / 2)
        color = "#2563eb" if diff >= 0 else "#dc2626"
        parts.append(f'<text x="{left - 12}" y="{y + 4:.2f}" text-anchor="end" font-family="Arial" font-size="13">{label}</text>')
        parts.append(f'<rect x="{x:.2f}" y="{y - 9:.2f}" width="{w:.2f}" height="18" fill="{color}"/>')
        parts.append(f'<text x="{x + (w + 8 if diff >= 0 else -8):.2f}" y="{y + 4:.2f}" text-anchor="{"start" if diff >= 0 else "end"}" font-family="Arial" font-size="12">{diff:.3f}</text>')
    parts.append("</svg>")
    write_svg(chart_dir / "top2_metric_difference_call100.svg", parts)


def write_charts(
    rows: Sequence[Dict[str, object]],
    leaderboard: Sequence[Dict[str, object]],
    summary_by_candidate: Sequence[Dict[str, object]],
    pairwise: Sequence[Dict[str, object]],
    candidate_names: Sequence[str],
    output_dir: Path,
) -> None:
    chart_dir = output_dir / "charts"
    chart_top_scores(leaderboard, chart_dir)
    chart_candidate_ci(summary_by_candidate, chart_dir)
    chart_controlled_vs_raw(pairwise, chart_dir)
    chart_boxplot(rows, candidate_names, chart_dir)
    chart_category_heatmap(rows, candidate_names, chart_dir)
    chart_metric_breakdown(leaderboard, chart_dir)
    chart_top2_difference(leaderboard, chart_dir)


def report_table(rows: Sequence[Dict[str, object]], fields: Sequence[str], limit: int = 10) -> List[str]:
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join("---" for _ in fields) + " |"]
    for row in list(rows)[:limit]:
        values = []
        for field in fields:
            value = row.get(field, "")
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(svg_escape(value) for value in values) + " |")
    return lines


def write_report(
    output_dir: Path,
    manifest: Sequence[Dict[str, str]],
    rows: Sequence[Dict[str, object]],
    leaderboard: Sequence[Dict[str, object]],
    summaries: Dict[str, Sequence[Dict[str, object]]],
    pairwise: Sequence[Dict[str, object]],
    failures: Sequence[Dict[str, object]],
    validation: Dict[str, object],
) -> None:
    source_dist = Counter(row.get("source_dataset", "") for row in manifest)
    origin_dist = Counter(row.get("origin", "") for row in manifest)
    category_dist = Counter(row.get("category", "") for row in manifest)
    controlled_raw = next((row for row in pairwise if row["group"] == "overall" and row["comparison"] == "controlled_minus_raw"), None)
    controlled_motif = next((row for row in pairwise if row["group"] == "overall" and row["comparison"] == "controlled_minus_motif"), None)
    lines = [
        "# Call100 Objective Search Report",
        "",
        "## Experiment Scale",
        "",
        f"- Output directory: `{output_dir}`",
        f"- Calls: `{len(manifest)}`",
        f"- Presets: `{len({row['preset'] for row in rows})}`",
        f"- Candidates: `{len({row['candidate'] for row in rows})}`",
        f"- Trials per preset/candidate/call: `{validation.get('trials_per_combo')}`",
        f"- all_objective_results rows: `{len(rows)}`",
        f"- Validation status: `{validation.get('status')}`",
        "",
        "## Dataset Distribution",
        "",
        "### Source Dataset",
        "",
    ]
    lines.extend([f"- `{key}`: {value}" for key, value in sorted(source_dist.items())])
    lines.extend(["", "### Origin", ""])
    lines.extend([f"- `{key}`: {value}" for key, value in sorted(origin_dist.items())])
    lines.extend(["", "### Category", ""])
    lines.extend([f"- `{key}`: {value}" for key, value in sorted(category_dist.items())])
    lines.extend(["", "## Top 10 Preset/Candidate", ""])
    lines.extend(report_table(leaderboard, ["rank", "preset", "candidate", "sample_count", "mean_objective_score", "ci95_low_objective_score", "ci95_high_objective_score"], 10))
    lines.extend(["", "## Controlled vs Raw", ""])
    if controlled_raw:
        lines.append(
            "Overall controlled-minus-raw mean difference: "
            f"`{fnum(controlled_raw['mean_diff_a_minus_b']):.6f}` "
            f"95% CI `[{fnum(controlled_raw['ci95_low']):.6f}, {fnum(controlled_raw['ci95_high']):.6f}]`, "
            f"bootstrap p=`{fnum(controlled_raw['p_two_sided_bootstrap']):.6f}`."
        )
    lines.extend(["", "## Motif Baseline vs Controlled", ""])
    if controlled_motif:
        lines.append(
            "Overall controlled-minus-motif mean difference: "
            f"`{fnum(controlled_motif['mean_diff_a_minus_b']):.6f}` "
            f"95% CI `[{fnum(controlled_motif['ci95_low']):.6f}, {fnum(controlled_motif['ci95_high']):.6f}]`, "
            f"bootstrap p=`{fnum(controlled_motif['p_two_sided_bootstrap']):.6f}`."
        )
    lines.extend(["", "## Performance by Origin", ""])
    lines.extend(report_table(summaries["origin"], ["rank", "origin", "sample_count", "mean_objective_score", "ci95_low_objective_score", "ci95_high_objective_score"], 20))
    lines.extend(["", "## Performance by Category", ""])
    lines.extend(report_table(summaries["category"], ["rank", "category", "sample_count", "mean_objective_score", "ci95_low_objective_score", "ci95_high_objective_score"], 30))
    lines.extend(["", "## Performance by Sub Category", ""])
    lines.extend(report_table(summaries["sub_category"], ["rank", "sub_category", "sample_count", "mean_objective_score", "ci95_low_objective_score", "ci95_high_objective_score"], 30))
    lines.extend(["", "## Failure Cases", ""])
    lines.append(f"- Failure-case rows written: `{len(failures)}`")
    lines.extend(report_table(failures, ["reason", "call_id", "preset", "candidate", "trial", "objective_score", "duration_match_ratio", "note_count"], 20))
    lines.extend(["", "## Statistical Confidence Intervals", ""])
    lines.append("- Summary CSV files use normal-approximation 95% CI over objective scores.")
    lines.append("- `pairwise_bootstrap_tests.csv` uses paired bootstrap over preset/call/trial matched candidate differences.")
    lines.extend(["", "## Validation Details", ""])
    for error in validation.get("errors", []):
        lines.append(f"- ERROR: {error}")
    if not validation.get("errors"):
        lines.append("- No validation errors.")
    (output_dir / "report_call100_objective_search.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_all_outputs(
    args: argparse.Namespace,
    preset_names: Sequence[str],
    candidate_names: Sequence[str],
    manifest: Sequence[Dict[str, str]],
    rows: Sequence[Dict[str, object]],
) -> Dict[str, object]:
    output_dir = Path(args.output_dir)
    all_path = output_dir / "all_objective_results.csv"
    write_csv(all_path, rows, ALL_RESULT_FIELDS)
    leaderboard = summarize_group(rows, ["preset", "candidate"])
    write_csv(output_dir / "objective_leaderboard.csv", leaderboard)

    rows_with_macro = add_macro_category(rows, manifest)
    macro_leaderboard = summarize_group(rows_with_macro, ["macro_category", "preset", "candidate"])
    write_csv(output_dir / "objective_leaderboard_macro_category.csv", macro_leaderboard)

    summary_by_candidate = summarize_group(rows, ["candidate"])
    summary_by_preset_candidate = summarize_group(rows, ["preset", "candidate"])
    summary_by_origin = summarize_group(rows, ["origin"])
    summary_by_source_dataset = summarize_group(rows, ["source_dataset"])
    summary_by_category = summarize_group(rows, ["category"])
    summary_by_sub_category = summarize_group(rows, ["sub_category"])
    write_csv(output_dir / "summary_by_candidate.csv", summary_by_candidate)
    write_csv(output_dir / "summary_by_preset_candidate.csv", summary_by_preset_candidate)
    write_csv(output_dir / "summary_by_origin.csv", summary_by_origin)
    write_csv(output_dir / "summary_by_source_dataset.csv", summary_by_source_dataset)
    write_csv(output_dir / "summary_by_category.csv", summary_by_category)
    write_csv(output_dir / "summary_by_sub_category.csv", summary_by_sub_category)

    pairwise = write_pairwise(rows, preset_names, output_dir, args.bootstrap_iterations)
    failures = write_failure_cases(rows, output_dir)
    validation = validate_all_results(rows, manifest, preset_names, candidate_names, args.trials)
    (output_dir / "validation_summary.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_charts(rows, leaderboard, summary_by_candidate, pairwise, candidate_names, output_dir)
    write_report(
        output_dir=output_dir,
        manifest=manifest,
        rows=rows,
        leaderboard=leaderboard,
        summaries={
            "origin": summary_by_origin,
            "category": summary_by_category,
            "sub_category": summary_by_sub_category,
        },
        pairwise=pairwise,
        failures=failures,
        validation=validation,
    )
    return validation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the formal Call100 objective search and build full outputs.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--trials", type=int, default=15)
    parser.add_argument("--seed", type=int, default=20260618)
    parser.add_argument("--presets", default=",".join(DEFAULT_PRESET_NAMES))
    parser.add_argument("--candidates", default=",".join(DEFAULT_CANDIDATES))
    parser.add_argument("--max-calls", type=int, default=None)
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--bpm", type=float, default=100.0)
    parser.add_argument("--pentatonic-root", type=int, default=60)
    parser.add_argument("--pentatonic-mode", choices=["major", "minor"], default="major")
    parser.add_argument("--bootstrap-iterations", type=int, default=300)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.trials < 1:
        raise SystemExit("--trials must be at least 1.")
    if args.max_calls is not None and args.max_calls < 1:
        raise SystemExit("--max-calls must be at least 1.")
    if args.bootstrap_iterations < 1:
        raise SystemExit("--bootstrap-iterations must be at least 1.")


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    preset_names = selected_names(args.presets, DEFAULT_PRESET_NAMES, PRESETS)
    candidate_names = selected_names(args.candidates, DEFAULT_CANDIDATES)
    manifest = manifest_rows(Path(args.manifest), args.max_calls)
    input_dir = prepare_call_inputs(manifest, output_dir)
    print(f"[setup] manifest={args.manifest}")
    print(f"[setup] output_dir={output_dir}")
    print(f"[setup] calls={len(manifest)} presets={len(preset_names)} candidates={len(candidate_names)} trials={args.trials}")
    print(f"[setup] expected_responses={len(manifest) * len(preset_names) * len(candidate_names) * args.trials}")
    run_generation(args, preset_names, candidate_names, input_dir, len(manifest))
    print("[aggregate] building all_objective_results.csv and summaries")
    rows = aggregate_all_results(args, preset_names, candidate_names, manifest)
    validation = write_all_outputs(args, preset_names, candidate_names, manifest, rows)
    if validation["errors"]:
        print("[validate] failed")
        for error in validation["errors"]:
            print(f"[validate] ERROR {error}")
        raise SystemExit(1)
    print("[validate] passed")
    print(f"[done] all_objective_results={output_dir / 'all_objective_results.csv'}")
    print(f"[done] report={output_dir / 'report_call100_objective_search.md'}")


if __name__ == "__main__":
    main()
