"""First-commit Call100 benchmark for the MIDI-VAD endpoint detector.

The final Note-Off in each isolated Call100 file is used as a reproducible
proxy reference boundary. A first commit more than 100 ms before that boundary
is a premature failure. Post-boundary deadlines of 0.5, 1.0, and 2.0 seconds
are all reported so the result does not depend on one selectively chosen
tolerance. Replay stops at the first commit, matching the deployed
Listening -> Candidate -> Commit -> Inference state transition. The reported
PS-F1 is a custom premature-sensitive first-commit score, not conventional
event-detection F1.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, median
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
DEPS = ROOT / ".python_deps"
if DEPS.exists():
    sys.path.insert(0, str(DEPS))
sys.path.insert(0, str(ROOT / "code"))

import mido  # type: ignore  # noqa: E402

from midi_vad_endpoint import (  # noqa: E402
    EndpointCancel,
    EndpointCandidate,
    EndpointDecision,
    MidiEndpointVAD,
)


PRIMARY_DEADLINE_SECONDS = 2.0
DEADLINES_SECONDS = (0.5, 1.0, PRIMARY_DEADLINE_SECONDS)
EARLY_TOLERANCE_SECONDS = 0.1
TICK_SECONDS = 0.005
TRAILING_SIMULATION_SECONDS = 15.0
BOOTSTRAP_ITERATIONS = 2000
BOOTSTRAP_BASE_SEED = 20260809


@dataclass(frozen=True)
class Condition:
    name: str
    detector: str
    chord_cluster_window: float
    confirmation_delay: float
    fixed_cutoff: Optional[float] = None
    family: str = "main"
    theta: float = 0.05
    window_size: int = 8
    min_intensity: float = 0.25
    min_cutoff: Optional[float] = None
    max_cutoff: Optional[float] = None


CONDITIONS = (
    Condition("adaptive_full", "adaptive", 0.08, 0.15),
    Condition("adaptive_no_clustering", "adaptive", 0.0, 0.15),
    Condition("adaptive_no_confirmation", "adaptive", 0.08, 0.0),
    Condition("adaptive_neither", "adaptive", 0.0, 0.0),
    Condition("fixed_300ms", "fixed", 0.08, 0.15, 0.3),
    Condition("fixed_500ms", "fixed", 0.08, 0.15, 0.5),
    Condition("fixed_800ms", "fixed", 0.08, 0.15, 0.8),
    Condition("cluster_40ms", "adaptive", 0.04, 0.15, family="sensitivity"),
    Condition("cluster_120ms", "adaptive", 0.12, 0.15, family="sensitivity"),
    Condition("confirm_75ms", "adaptive", 0.08, 0.075, family="sensitivity"),
    Condition("confirm_250ms", "adaptive", 0.08, 0.25, family="sensitivity"),
    Condition("theta_0p02", "adaptive", 0.08, 0.15, family="theta", theta=0.02),
    Condition("theta_0p10", "adaptive", 0.08, 0.15, family="theta", theta=0.10),
    Condition("theta_0p20", "adaptive", 0.08, 0.15, family="theta", theta=0.20),
    Condition("window_4", "adaptive", 0.08, 0.15, family="window", window_size=4),
    Condition("window_12", "adaptive", 0.08, 0.15, family="window", window_size=12),
    Condition("floor_0p125", "adaptive", 0.08, 0.15, family="floor", min_intensity=0.125),
    Condition("floor_0p50", "adaptive", 0.08, 0.15, family="floor", min_intensity=0.50),
    Condition("max_cutoff_2s", "adaptive", 0.08, 0.15, family="clamp", max_cutoff=2.0),
    Condition("max_cutoff_4s", "adaptive", 0.08, 0.15, family="clamp", max_cutoff=4.0),
)


class FixedCutoffVAD(MidiEndpointVAD):
    def __init__(self, fixed_cutoff: float, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.fixed_cutoff = fixed_cutoff

    def tau_cutoff(self) -> float:
        return self.fixed_cutoff


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def provenance_path(path: Path) -> str:
    """Return a repository-relative path, or a filename for external inputs."""

    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return resolved.name


def read_manifest(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 100:
        raise ValueError(f"Call100 manifest must contain 100 rows, found {len(rows)}")
    return rows


def read_midi_timeline(path: Path) -> Tuple[List[Tuple[float, int, int]], float]:
    midi = mido.MidiFile(path)
    tempo = 500000
    current_time = 0.0
    note_ons: List[Tuple[float, int, int]] = []
    final_note_off = 0.0

    for message in mido.merge_tracks(midi.tracks):
        current_time += mido.tick2second(message.time, midi.ticks_per_beat, tempo)
        if message.type == "set_tempo":
            tempo = message.tempo
        elif message.type == "note_on" and message.velocity > 0:
            note_ons.append((current_time, int(message.note), int(message.velocity)))
        elif message.type == "note_off" or (
            message.type == "note_on" and message.velocity == 0
        ):
            final_note_off = max(final_note_off, current_time)

    if not note_ons:
        raise ValueError(f"No Note-On events in {path}")
    if final_note_off <= 0:
        final_note_off = note_ons[-1][0]
    return note_ons, final_note_off


def make_detector(
    condition: Condition,
    candidates: List[EndpointCandidate],
    cancels: List[EndpointCancel],
    decisions: List[EndpointDecision],
) -> MidiEndpointVAD:
    kwargs = dict(
        theta=condition.theta,
        window_size=condition.window_size,
        min_intensity=condition.min_intensity,
        min_cutoff=condition.min_cutoff,
        max_cutoff=condition.max_cutoff,
        chord_cluster_window=condition.chord_cluster_window,
        endpoint_confirm_delay=condition.confirmation_delay,
        on_candidate_endpoint=candidates.append,
        on_candidate_cancel=cancels.append,
        on_endpoint=decisions.append,
    )
    if condition.fixed_cutoff is not None:
        return FixedCutoffVAD(condition.fixed_cutoff, **kwargs)
    return MidiEndpointVAD(**kwargs)


def simulate_call(
    call_id: str,
    midi_path: Path,
    condition: Condition,
) -> Dict[str, object]:
    note_ons, reference_time = read_midi_timeline(midi_path)
    candidates: List[EndpointCandidate] = []
    cancels: List[EndpointCancel] = []
    decisions: List[EndpointDecision] = []
    detector = make_detector(condition, candidates, cancels, decisions)

    clock = 0.0
    committed = False
    for event_time, pitch, velocity in note_ons:
        while clock + TICK_SECONDS < event_time - 1e-12:
            clock += TICK_SECONDS
            detector.tick(clock)
            if decisions:
                committed = True
                break
        if committed:
            break
        clock = event_time
        detector.observe_note_on(pitch, velocity, event_time)

    if not decisions:
        stop_time = reference_time + TRAILING_SIMULATION_SECONDS
        while clock < stop_time:
            clock = min(clock + TICK_SECONDS, stop_time)
            detector.tick(clock)
            if decisions:
                break

    cut_times = [decision.cut_time for decision in decisions]
    first_cut = cut_times[0] if cut_times else None
    row: Dict[str, object] = {
        "condition": condition.name,
        "condition_family": condition.family,
        "detector": condition.detector,
        "chord_cluster_window_s": condition.chord_cluster_window,
        "confirmation_delay_s": condition.confirmation_delay,
        "fixed_cutoff_s": "" if condition.fixed_cutoff is None else condition.fixed_cutoff,
        "theta": condition.theta,
        "window_size": condition.window_size,
        "min_intensity": condition.min_intensity,
        "min_cutoff_s": "" if condition.min_cutoff is None else condition.min_cutoff,
        "max_cutoff_s": "" if condition.max_cutoff is None else condition.max_cutoff,
        "call_id": call_id,
        "midi_file": midi_path.name,
        "midi_sha256": sha256(midi_path),
        "note_on_count": len(note_ons),
        "reference_final_note_off_s": reference_time,
        "candidate_count": len(candidates),
        "cancel_count": len(cancels),
        "decision_count": len(decisions),
        "premature_decision_count": int(
            first_cut is not None
            and first_cut < reference_time - EARLY_TOLERANCE_SECONDS
        ),
        "late_decision_count_2s": int(
            first_cut is not None
            and first_cut > reference_time + PRIMARY_DEADLINE_SECONDS
        ),
        "candidate_cancel_rate": len(cancels) / len(candidates) if candidates else 0.0,
        "decision_times_s": "" if first_cut is None else f"{first_cut:.6f}",
    }

    for deadline in DEADLINES_SECONDS:
        if first_cut is None:
            outcome = "missed"
            matched = 0
            match_index = -1
            error: object = ""
            false_positive_count = 0
            false_negative_count = 1
        elif first_cut < reference_time - EARLY_TOLERANCE_SECONDS:
            outcome = "premature"
            matched = 0
            match_index = -1
            error = ""
            false_positive_count = 1
            false_negative_count = 1
        elif first_cut <= reference_time + deadline:
            outcome = "success"
            matched = 1
            match_index = 0
            error = first_cut - reference_time
            false_positive_count = 0
            false_negative_count = 0
        else:
            outcome = "late"
            match_index = -1
            matched = 0
            error = ""
            false_positive_count = 0
            false_negative_count = 1
        suffix = str(int(deadline * 1000))
        row[f"first_commit_outcome_{suffix}ms"] = outcome
        row[f"matched_{suffix}ms"] = matched
        row[f"matched_decision_index_{suffix}ms"] = match_index
        row[f"endpoint_error_s_{suffix}ms"] = error
        row[f"false_positive_count_{suffix}ms"] = false_positive_count
        row[f"false_negative_count_{suffix}ms"] = false_negative_count
    return row


def percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def metric_bundle(rows: Sequence[Dict[str, object]], deadline: float) -> Dict[str, float]:
    suffix = str(int(deadline * 1000))
    outcome_counts = {
        outcome: sum(
            str(row[f"first_commit_outcome_{suffix}ms"]) == outcome for row in rows
        )
        for outcome in ("success", "premature", "late", "missed")
    }
    true_positives = sum(int(row[f"matched_{suffix}ms"]) for row in rows)
    false_positives = sum(int(row[f"false_positive_count_{suffix}ms"]) for row in rows)
    false_negatives = sum(int(row[f"false_negative_count_{suffix}ms"]) for row in rows)
    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives else 0.0
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    errors = [
        float(row[f"endpoint_error_s_{suffix}ms"])
        for row in rows
        if row[f"endpoint_error_s_{suffix}ms"] != ""
    ]
    return {
        "true_positive_count": float(true_positives),
        "false_positive_count": float(false_positives),
        "false_negative_count": float(false_negatives),
        "precision": precision,
        "recall": recall,
        "premature_sensitive_first_commit_f1": f1,
        # Legacy field retained for compatibility; see the score definition above.
        "f1": f1,
        "success_count": float(outcome_counts["success"]),
        "premature_count": float(outcome_counts["premature"]),
        "late_count": float(outcome_counts["late"]),
        "missed_count": float(outcome_counts["missed"]),
        "signed_error_effective_n": float(len(errors)),
        "median_signed_error_s": median(errors) if errors else math.nan,
        "mean_absolute_error_s": mean(abs(value) for value in errors) if errors else math.nan,
        "p90_absolute_error_s": percentile([abs(value) for value in errors], 0.9),
    }


def bootstrap_interval(
    rows: Sequence[Dict[str, object]], deadline: float, metric: str, seed: int
) -> Tuple[float, float]:
    rng = random.Random(seed)
    estimates: List[float] = []
    for _ in range(BOOTSTRAP_ITERATIONS):
        sample = [rows[rng.randrange(len(rows))] for _ in rows]
        estimates.append(metric_bundle(sample, deadline)[metric])
    return percentile(estimates, 0.025), percentile(estimates, 0.975)


def summarize(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    groups: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[str(row["condition"])].append(row)
    condition_map = {condition.name: condition for condition in CONDITIONS}
    summaries: List[Dict[str, object]] = []
    for condition in CONDITIONS:
        items = groups[condition.name]
        for deadline in DEADLINES_SECONDS:
            bundle = metric_bundle(items, deadline)
            f1_low, f1_high = bootstrap_interval(
                items, deadline, "f1", seed=BOOTSTRAP_BASE_SEED + int(deadline * 1000)
            )
            summary: Dict[str, object] = {
                "condition": condition.name,
                "condition_family": condition.family,
                "detector": condition.detector,
                "chord_cluster_window_s": condition.chord_cluster_window,
                "confirmation_delay_s": condition.confirmation_delay,
                "fixed_cutoff_s": "" if condition.fixed_cutoff is None else condition.fixed_cutoff,
                "theta": condition.theta,
                "window_size": condition.window_size,
                "min_intensity": condition.min_intensity,
                "min_cutoff_s": "" if condition.min_cutoff is None else condition.min_cutoff,
                "max_cutoff_s": "" if condition.max_cutoff is None else condition.max_cutoff,
                "deadline_s": deadline,
                "call_count": len(items),
                **bundle,
                "f1_ci95_low": f1_low,
                "f1_ci95_high": f1_high,
                "candidate_count": sum(int(row["candidate_count"]) for row in items),
                "cancel_count": sum(int(row["cancel_count"]) for row in items),
                "candidate_cancel_rate": (
                    sum(int(row["cancel_count"]) for row in items)
                    / sum(int(row["candidate_count"]) for row in items)
                    if sum(int(row["candidate_count"]) for row in items)
                    else 0.0
                ),
                "premature_decision_count": sum(
                    int(row["premature_decision_count"]) for row in items
                ),
            }
            summaries.append(summary)
    return summaries


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0].keys()), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, summaries: Sequence[Dict[str, object]]) -> None:
    primary = [
        row
        for row in summaries
        if float(row["deadline_s"]) == PRIMARY_DEADLINE_SECONDS
    ]
    lines = [
        "# Call100 MIDI-VAD Endpoint Benchmark",
        "",
        "Reference boundary: final Note-Off in each isolated Call100 MIDI file.",
        "Replay stops at the first commit. A first commit earlier than 100 ms before",
        "the reference is a premature failure; commits after the first are never counted. Results are",
        "reported at 0.5, 1.0, and 2.0 s post-boundary deadlines; the table below uses 2.0 s.",
        "This file-end proxy is reproducible but is not a substitute for human boundary annotation.",
        "",
        "PS-F1 is the custom premature-sensitive first-commit score: TP=S, FP=P,",
        "and FN=P+L+M. Signed error is commit minus final Note-Off and is computed",
        "only for successful commits. S/P/L/M are mutually exclusive and sum to 100.",
        "",
        "| condition | precision | recall | PS-F1 [95% CI] | S/P/L/M | median error (s; N) | MAE (s) | cancel rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in primary:
        lines.append(
            f"| {row['condition']} | {float(row['precision']):.3f} | "
            f"{float(row['recall']):.3f} | {float(row['f1']):.3f} "
            f"[{float(row['f1_ci95_low']):.3f}, {float(row['f1_ci95_high']):.3f}] | "
            f"{int(float(row['success_count']))}/{int(float(row['premature_count']))}/"
            f"{int(float(row['late_count']))}/{int(float(row['missed_count']))} | "
            f"{float(row['median_signed_error_s']):.3f} ({int(float(row['signed_error_effective_n']))}) | "
            f"{float(row['mean_absolute_error_s']):.3f} | "
            f"{float(row['candidate_cancel_rate']):.3f} |"
        )
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "ab_tests" / "calls_100_public_final" / "call100_manifest.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "ab_tests" / "endpoint_benchmark_call100",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = read_manifest(args.manifest)
    results: List[Dict[str, object]] = []
    for condition in CONDITIONS:
        for index, manifest_row in enumerate(manifest_rows, start=1):
            midi_path = Path(manifest_row["midi_path"])
            results.append(simulate_call(manifest_row["call_id"], midi_path, condition))
        print(f"[condition] {condition.name}: {index} calls")

    summaries = summarize(results)
    write_csv(args.output_dir / "endpoint_call_level_results.csv", results)
    write_csv(args.output_dir / "endpoint_condition_summary.csv", summaries)
    write_report(args.output_dir / "report.md", summaries)
    provenance = {
        "benchmark_definition": {
            "reference": "final Note-Off in each isolated Call100 file",
            "early_tolerance_s": EARLY_TOLERANCE_SECONDS,
            "reported_deadlines_s": DEADLINES_SECONDS,
            "primary_deadline_s": PRIMARY_DEADLINE_SECONDS,
            "tick_s": TICK_SECONDS,
            "trailing_simulation_s": TRAILING_SIMULATION_SECONDS,
            "ground_truth_limitation": "file-end proxy; no human boundary annotation",
            "state_machine_semantics": "replay terminates after the first commit",
            "replay_horizon_after_reference_s": TRAILING_SIMULATION_SECONDS,
            "score_name": "premature-sensitive first-commit F1 (PS-F1)",
            "score_counts": "TP=success; FP=premature; FN=premature+late+missed",
            "signed_error_scope": "commit minus final Note-Off; successful first commits only",
            "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
            "bootstrap_seed_by_deadline": "20260809 + int(deadline_seconds * 1000); shared across conditions at the same deadline",
            "bootstrap_interval": "linearly interpolated 2.5th and 97.5th percentiles at ordered-sample position (B-1)*q",
        },
        "detector_defaults": {
            "theta": 0.05,
            "window_size": 8,
            "min_intensity": 0.25,
            "chord_cluster_window_s": 0.08,
            "confirmation_delay_s": 0.15,
        },
        "conditions": [asdict(condition) for condition in CONDITIONS],
        "benchmark_input_manifest": {
            "manifest_file": provenance_path(args.manifest),
            "manifest_sha256": sha256(args.manifest),
            "path_policy": "the published release records a path-redacted manifest separately when the benchmark was run from a private absolute-path manifest",
        },
        "detector_sha256": sha256(ROOT / "code" / "midi_vad_endpoint.py"),
        "benchmark_sha256": sha256(Path(__file__)),
        "call_count": len(manifest_rows),
        "row_count": len(results),
    }
    (args.output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    print(f"[done] output={args.output_dir}")


if __name__ == "__main__":
    main()
