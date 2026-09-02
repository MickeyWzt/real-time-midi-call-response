"""Recompute the scheduler replay from published first-event timings.

The replay uses a commit-anchored first-event target

    s = max(B, C1 - H)

where B is the commit-relative readiness target and H is the pre-commit
overlap. The source field ``first_token_latency_sec`` is a legacy name: it was
recorded when the generator first yielded a complete GeneratedEvent. This
script performs no AMT generation, MIDI transmission, audio measurement, or
hypothesis test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List

from run_call100_ablation_latency import (
    LATENCY_FIELDS,
    fnum,
    latency_summary,
    preload_comparison,
    read_csv,
    write_csv,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    ROOT / "results" / "call100_ablation_latency" / "latency_log_all_trials.csv"
)
DEFAULT_OUTPUT = ROOT / "results" / "call100_ablation_latency"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def recompute(
    rows: List[Dict[str, str]],
    preload_window_ms: float,
    startup_target_ms: float,
) -> List[Dict[str, object]]:
    output: List[Dict[str, object]] = []
    for source in rows:
        row: Dict[str, object] = dict(source)
        condition = str(source["condition"])
        overlap = preload_window_ms if condition == "L1_preload_on" else 0.0
        first_generated_event_ms = fnum(
            source.get(
                "first_generated_event_latency_ms",
                source.get("first_token_latency_ms", ""),
            )
        )
        first_ready = first_generated_event_ms - overlap
        first_queued = max(0.0, first_ready)
        first_out = max(startup_target_ms, first_ready)
        response_duration_ms = fnum(source["response_duration_sec"]) * 1000.0
        startup_miss = int(first_ready > startup_target_ms)

        row.update(
            {
                "t_endpoint_commit_ms": "0.000000",
                "t_infer_start_ms": f"{-overlap:.6f}",
                "t_first_generated_event_ready_ms": f"{first_ready:.6f}",
                "commit_to_first_event_readiness_lower_bound_ms": f"{first_out:.6f}",
                "first_generated_event_latency_ms": f"{first_generated_event_ms:.6f}",
                "modeled_response_end_lower_bound_ms": f"{first_out + response_duration_ms:.6f}",
                # Legacy aliases: these are modeled readiness values, not
                # measured MIDI transmission or audio-onset timestamps.
                "t_first_token_ready_ms": f"{first_ready:.6f}",
                "t_first_midi_queued_ms": f"{first_queued:.6f}",
                "t_first_midi_out_ms": f"{first_out:.6f}",
                "t_response_end_ms": f"{first_out + response_duration_ms:.6f}",
                "endpoint_to_first_midi_ms": f"{first_out:.6f}",
                "queue_delay_ms": f"{first_out - first_queued:.6f}",
                "startup_deadline_miss": startup_miss,
                "startup_deadline_miss_rate": f"{float(startup_miss):.6f}",
                "startup_target_ms": f"{startup_target_ms:.6f}",
                # Backward-compatible aliases; these are not streaming underruns.
                "buffer_underrun_count": startup_miss,
                "buffer_underrun_rate": f"{float(startup_miss):.6f}",
                "preload_window_ms": f"{overlap:.6f}",
                "micro_buffer_ms": f"{startup_target_ms:.6f}",
            }
        )
        output.append(row)
    return output


def markdown_table(rows: List[Dict[str, object]], fields: List[str]) -> List[str]:
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join("---:" if field not in {"condition", "comparison"} else "---" for field in fields) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |")
    return lines


def write_report(
    path: Path,
    summaries: List[Dict[str, object]],
    paired: List[Dict[str, object]],
) -> None:
    summary_fields = [
        "condition",
        "sample_count",
        "mean_first_event_readiness_lower_bound_ms",
        "p50_first_event_readiness_lower_bound_ms",
        "p95_first_event_readiness_lower_bound_ms",
        "startup_deadline_miss_rate",
        "mean_first_generated_event_latency_ms",
        "mean_total_generation_ms",
    ]
    paired_fields = [
        "comparison",
        "paired_sample_count",
        "mean_readiness_lower_bound_reduction_ms",
        "positive_pairs",
        "negative_pairs",
        "tied_pairs",
        "inference",
    ]
    lines = [
        "# Call100 First-Event Readiness Replay",
        "",
        "The source timing is elapsed local AMT inference until the first complete `GeneratedEvent` is yielded; its legacy field name is `first_token_latency_sec`. The commit-anchored quantity `max(B, C1-H)` is an earliest readiness/release lower bound, not measured MIDI-send, host-receive, or audio-onset latency. Full-stream underrun is also unmeasured.",
        "",
        "## Summary By Condition",
        "",
        *markdown_table(summaries, summary_fields),
        "",
        "## Paired Preload Comparison",
        "",
        *markdown_table(paired, paired_fields),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--preload-window-ms", type=float, default=150.0)
    parser.add_argument("--startup-target-ms", type=float, default=80.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.preload_window_ms < 0:
        raise SystemExit("--preload-window-ms cannot be negative")
    if args.startup_target_ms <= 0:
        raise SystemExit("--startup-target-ms must be positive")
    source_hash = sha256(args.input)
    rows = recompute(
        read_csv(args.input), args.preload_window_ms, args.startup_target_ms
    )
    if len(rows) != 18000:
        raise RuntimeError(f"Expected 18000 replay rows, found {len(rows)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "latency_log_all_trials.csv", rows, LATENCY_FIELDS)
    by_condition = latency_summary(rows, ["condition"])
    by_length = latency_summary(rows, ["condition", "length_bin"])
    paired = preload_comparison(rows)
    write_csv(args.output_dir / "latency_summary_by_condition.csv", by_condition)
    write_csv(args.output_dir / "latency_summary_by_length_bin.csv", by_length)
    write_csv(args.output_dir / "preload_on_off_comparison.csv", paired)
    write_report(args.output_dir / "latency_report.md", by_condition, paired)
    provenance = {
        "formula": "s=max(B,C1-H)",
        "canonical_quantity": "commit-to-first-generated-event readiness lower bound",
        "source_timing_semantics": "elapsed inference time until the first complete GeneratedEvent was yielded",
        "not_measured": [
            "queue send time",
            "operating-system MIDI send/receive time",
            "host or DAW receive time",
            "audio onset",
        ],
        "legacy_aliases": {
            "first_token_latency_ms": "first_generated_event_latency_ms",
            "t_first_token_ready_ms": "t_first_generated_event_ready_ms",
            "endpoint_to_first_midi_ms": "commit_to_first_event_readiness_lower_bound_ms; not measured MIDI output",
            "t_first_midi_queued_ms": "commit_to_first_event_readiness_lower_bound_ms; modeled alias, not a queue timestamp",
            "t_first_midi_out_ms": "commit_to_first_event_readiness_lower_bound_ms; modeled alias, not a transmitted MIDI timestamp",
            "t_response_end_ms": "modeled_response_end_lower_bound_ms; modeled alias, not an observed response-end timestamp",
        },
        "startup_target_ms": args.startup_target_ms,
        "preload_window_ms": args.preload_window_ms,
        "input_file": args.input.name,
        "input_sha256_before_recompute": source_hash,
        "row_count": len(rows),
        "inference": "deterministic scheduler replay; no hypothesis test",
    }
    (args.output_dir / "scheduler_replay_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[done] rows={len(rows)} output={args.output_dir}")


if __name__ == "__main__":
    main()
