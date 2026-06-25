"""
Objective melody metrics for offline Call-and-Response A/B runs.

The metrics are intentionally lightweight and thesis-friendly: they quantify
pitch diversity, scale adherence, melodic intervals, repetition, rhythm
regularity, groove similarity, cadence, and symbolic compressibility.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import zlib
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Optional


ROOT = Path(__file__).resolve().parents[1]
PROJECT_DEPS = ROOT / ".python_deps"
if PROJECT_DEPS.exists():
    sys.path.insert(0, str(PROJECT_DEPS))


@dataclass(frozen=True)
class Note:
    onset_ticks: int
    duration_ticks: int
    onset_seconds: float
    duration_seconds: float
    pitch: int
    velocity: int


def require_runtime() -> None:
    try:
        import mido  # noqa: F401
    except ImportError as exc:
        raise SystemExit("Missing mido. Use Python 3.12 with project .python_deps.") from exc


def read_notes(path: Path) -> tuple[List[Note], int]:
    import mido

    midi = mido.MidiFile(path)
    tempo = 500000
    abs_ticks = 0
    abs_seconds = 0.0
    active: Dict[tuple[int, int], tuple[int, float, int]] = {}
    notes: List[Note] = []

    for msg in mido.merge_tracks(midi.tracks):
        abs_ticks += msg.time
        abs_seconds += mido.tick2second(msg.time, midi.ticks_per_beat, tempo)
        if msg.type == "set_tempo":
            tempo = msg.tempo
            continue
        if msg.type == "note_on" and msg.velocity > 0:
            active[(msg.channel, msg.note)] = (abs_ticks, abs_seconds, msg.velocity)
            continue
        is_note_off = msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0)
        if is_note_off:
            start = active.pop((msg.channel, msg.note), None)
            if start is None:
                continue
            start_ticks, start_seconds, velocity = start
            notes.append(
                Note(
                    onset_ticks=start_ticks,
                    duration_ticks=max(1, abs_ticks - start_ticks),
                    onset_seconds=start_seconds,
                    duration_seconds=max(0.001, abs_seconds - start_seconds),
                    pitch=msg.note,
                    velocity=velocity,
                )
            )

    notes.sort(key=lambda item: (item.onset_ticks, item.pitch))
    return notes, midi.ticks_per_beat


def entropy(values: Iterable[int]) -> float:
    counts = Counter(values)
    total = sum(counts.values())
    if total == 0:
        return 0.0
    result = 0.0
    for count in counts.values():
        p = count / total
        result -= p * math.log2(p)
    return result


def scale_degrees(mode: str) -> set[int]:
    if mode == "minor":
        return {0, 3, 5, 7, 10}
    return {0, 2, 4, 7, 9}


def stable_degrees(mode: str) -> set[int]:
    if mode == "minor":
        return {0, 3, 7}
    return {0, 4, 7}


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def closeness_score(value: float, target: float, tolerance: float) -> float:
    if tolerance <= 0:
        return 1.0 if value == target else 0.0
    return clamp01(1.0 - abs(value - target) / tolerance)


def compression_ratio(notes: List[Note], ticks_per_beat: int) -> float:
    if not notes:
        return 1.0
    items = [
        f"{round(note.onset_ticks / ticks_per_beat, 3)}:{round(note.duration_ticks / ticks_per_beat, 3)}:{note.pitch}"
        for note in notes
    ]
    raw = " ".join(items).encode("utf-8")
    if not raw:
        return 1.0
    return len(zlib.compress(raw)) / len(raw)


def groove_similarity(notes: List[Note], ticks_per_beat: int, beats_per_bar: int, slots_per_bar: int) -> float:
    if beats_per_bar <= 0 or slots_per_bar <= 0:
        return 0.0
    bar_ticks = ticks_per_beat * beats_per_bar
    if bar_ticks <= 0:
        return 0.0
    bars: Dict[int, List[int]] = defaultdict(lambda: [0] * slots_per_bar)
    for note in notes:
        bar_idx = note.onset_ticks // bar_ticks
        pos = note.onset_ticks % bar_ticks
        slot = int(round(pos / bar_ticks * slots_per_bar)) % slots_per_bar
        bars[bar_idx][slot] = 1
    ordered = [bars[idx] for idx in sorted(bars)]
    if len(ordered) < 2:
        return 1.0 if ordered else 0.0
    similarities = []
    for left, right in zip(ordered, ordered[1:]):
        mismatches = sum(1 for a, b in zip(left, right) if a != b)
        similarities.append(1.0 - mismatches / slots_per_bar)
    return mean(similarities)


def max_repeat_run(pitches: List[int]) -> int:
    if not pitches:
        return 0
    best = 1
    run = 1
    for prev, current in zip(pitches, pitches[1:]):
        if current == prev:
            run += 1
        else:
            best = max(best, run)
            run = 1
    return max(best, run)


def strong_beat_stable_rate(
    notes: List[Note],
    ticks_per_beat: int,
    beats_per_bar: int,
    root_pc: int,
    mode: str,
) -> float:
    strong_total = 0
    stable_total = 0
    stable = stable_degrees(mode)
    tolerance = max(1, ticks_per_beat // 16)
    strong_positions = [0]
    if beats_per_bar >= 4:
        strong_positions.append(2 * ticks_per_beat)
    bar_ticks = ticks_per_beat * beats_per_bar
    for note in notes:
        position = note.onset_ticks % bar_ticks
        if any(abs(position - strong) <= tolerance for strong in strong_positions):
            strong_total += 1
            if (note.pitch - root_pc) % 12 in stable:
                stable_total += 1
    if strong_total == 0:
        return 1.0
    return stable_total / strong_total


def cadence_score(notes: List[Note], root_pc: int, mode: str) -> float:
    if not notes:
        return 0.0
    pc = (notes[-1].pitch - root_pc) % 12
    if pc == 0:
        return 1.0
    if pc == 7:
        return 0.85
    if pc in stable_degrees(mode):
        return 0.7
    if pc in scale_degrees(mode):
        return 0.45
    return 0.0


def evaluate_notes(notes: List[Note], ticks_per_beat: int, args: argparse.Namespace) -> Dict[str, float]:
    if not notes:
        return {
            "note_count": 0,
            "objective_score": 0.0,
        }

    pitches = [note.pitch for note in notes]
    pcs = [pitch % 12 for pitch in pitches]
    intervals = [b - a for a, b in zip(pitches, pitches[1:])]
    abs_intervals = [abs(item) for item in intervals]
    root_pc = args.scale_root % 12
    allowed = scale_degrees(args.scale_mode)

    note_count = len(notes)
    transition_count = max(1, note_count - 1)
    duration_seconds = max(note.onset_seconds + note.duration_seconds for note in notes) - min(
        note.onset_seconds for note in notes
    )
    pche = entropy(pcs)
    upc = len(set(pcs))
    psr = sum(1 for pitch in pitches if (pitch - root_pc) % 12 in allowed) / note_count
    tone_span_ratio = sum(1 for item in abs_intervals if item > args.large_interval_threshold) / transition_count
    mean_abs_interval = mean(abs_intervals) if abs_intervals else 0.0
    max_abs_interval = max(abs_intervals) if abs_intervals else 0.0
    stepwise_rate = sum(1 for item in abs_intervals if item <= args.stepwise_threshold) / transition_count
    cpr = sum(1 for a, b in zip(pitches, pitches[1:]) if a == b) / transition_count
    longest_run = max_repeat_run(pitches)
    qn = sum(1 for note in notes if note.duration_ticks / ticks_per_beat >= args.qualified_note_beats) / note_count

    standard_durations = [0.125, 0.25, 0.375, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]
    qrf = 0
    for note in notes:
        duration_beats = note.duration_ticks / ticks_per_beat
        if any(abs(duration_beats - standard) <= args.rhythm_tolerance_beats for standard in standard_durations):
            qrf += 1
    qrf_rate = qrf / note_count
    groove = groove_similarity(notes, ticks_per_beat, args.beats_per_bar, args.groove_slots)
    strong_stable = strong_beat_stable_rate(notes, ticks_per_beat, args.beats_per_bar, root_pc, args.scale_mode)
    cadence = cadence_score(notes, root_pc, args.scale_mode)
    comp = compression_ratio(notes, ticks_per_beat)

    pitch_diversity_score = 0.55 * closeness_score(pche, args.target_pche, args.pche_tolerance) + 0.45 * closeness_score(
        upc, args.target_upc, args.upc_tolerance
    )
    interval_score = 0.6 * stepwise_rate + 0.4 * (1.0 - tone_span_ratio)
    repetition_score = clamp01(1.0 - cpr * 2.0 - max(0, longest_run - 2) * 0.2)
    rhythm_score = (qn + qrf_rate + groove) / 3.0
    tonality_score = (psr + cadence + strong_stable) / 3.0
    compression_score = closeness_score(comp, args.target_compression_ratio, args.compression_tolerance)
    objective_score = (
        0.22 * tonality_score
        + 0.20 * rhythm_score
        + 0.18 * interval_score
        + 0.16 * repetition_score
        + 0.14 * pitch_diversity_score
        + 0.10 * compression_score
    )

    return {
        "note_count": float(note_count),
        "duration_seconds": duration_seconds,
        "pitch_range": float(max(pitches) - min(pitches)),
        "unique_pitch_count": float(len(set(pitches))),
        "upc": float(upc),
        "pche": pche,
        "psr": psr,
        "tone_span_ratio": tone_span_ratio,
        "mean_abs_interval": mean_abs_interval,
        "max_abs_interval": max_abs_interval,
        "stepwise_rate": stepwise_rate,
        "cpr": cpr,
        "longest_repeat_run": float(longest_run),
        "qualified_note_rate": qn,
        "qualified_rhythm_rate": qrf_rate,
        "groove_similarity": groove,
        "strong_beat_stable_rate": strong_stable,
        "cadence_score": cadence,
        "compression_ratio": comp,
        "pitch_diversity_score": pitch_diversity_score,
        "interval_score": interval_score,
        "repetition_score": repetition_score,
        "rhythm_score": rhythm_score,
        "tonality_score": tonality_score,
        "compression_score": compression_score,
        "objective_score": objective_score,
    }


def read_answer_key(path: Optional[Path]) -> Dict[str, Dict[str, str]]:
    if path is None or not path.exists():
        return {}
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return {row["sample_id"]: row for row in csv.DictReader(handle)}


def collect_midis(args: argparse.Namespace) -> tuple[Path, List[Path], Optional[Path]]:
    if args.run_dir:
        run_dir = Path(args.run_dir)
        midi_dir = run_dir / args.subdir
        answer_key = run_dir / "answer_key.csv"
    else:
        midi_dir = Path(args.midi_dir)
        run_dir = midi_dir.parent
        answer_key = Path(args.answer_key) if args.answer_key else None
    if not midi_dir.exists():
        raise SystemExit(f"MIDI directory does not exist: {midi_dir}")
    files = sorted([item for item in midi_dir.iterdir() if item.suffix.lower() in {".mid", ".midi"}])
    if not files:
        raise SystemExit(f"No MIDI files found in: {midi_dir}")
    return run_dir, files, answer_key


def format_row(row: Dict[str, object]) -> Dict[str, object]:
    formatted: Dict[str, object] = {}
    for key, value in row.items():
        if isinstance(value, float):
            formatted[key] = f"{value:.6f}"
        else:
            formatted[key] = value
    return formatted


def write_rows(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(format_row(row))


def summarize(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    groups: Dict[tuple[str, str], List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[(str(row.get("candidate", "unknown")), str(row.get("model_id", "")))].append(row)

    metric_names = [
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
    ]
    summaries: List[Dict[str, object]] = []
    for (candidate, model_id), items in groups.items():
        summary: Dict[str, object] = {
            "candidate": candidate,
            "model_id": model_id,
            "sample_count": len(items),
        }
        for metric in metric_names:
            values = [float(item[metric]) for item in items if metric in item]
            if values:
                summary[f"mean_{metric}"] = mean(values)
        summaries.append(summary)
    summaries.sort(key=lambda item: float(item.get("mean_objective_score", 0.0)), reverse=True)
    for rank, item in enumerate(summaries, start=1):
        item["rank"] = rank
    return summaries


def evaluate(args: argparse.Namespace) -> tuple[Path, Path]:
    require_runtime()
    run_dir, files, answer_key_path = collect_midis(args)
    answer_key = read_answer_key(answer_key_path)
    rows: List[Dict[str, object]] = []

    for path in files:
        sample_id = path.stem.replace("_response_only", "")
        notes, ticks_per_beat = read_notes(path)
        metrics = evaluate_notes(notes, ticks_per_beat, args)
        metadata = answer_key.get(sample_id, {})
        row: Dict[str, object] = {
            "sample_id": sample_id,
            "midi_path": str(path),
            "candidate": metadata.get("candidate", "unknown"),
            "model_id": metadata.get("model_id", ""),
            "call_id": metadata.get("call_id", ""),
            "trial": metadata.get("trial", ""),
        }
        row.update(metrics)
        rows.append(row)

    output_csv = Path(args.output_csv) if args.output_csv else run_dir / "objective_metrics.csv"
    summary_csv = Path(args.summary_csv) if args.summary_csv else run_dir / "objective_summary.csv"
    write_rows(output_csv, rows)
    write_rows(summary_csv, summarize(rows))
    print(f"[metrics] sample_metrics={output_csv}")
    print(f"[metrics] summary={summary_csv}")
    return output_csv, summary_csv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate generated MIDI melodies with objective metrics")
    parser.add_argument("--run-dir", default=None, help="A/B run directory containing responses/ and answer_key.csv")
    parser.add_argument("--subdir", default="responses", help="MIDI subdirectory inside --run-dir")
    parser.add_argument("--midi-dir", default=None, help="standalone MIDI directory")
    parser.add_argument("--answer-key", default=None, help="optional answer_key.csv for standalone MIDI directory")
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--summary-csv", default=None)
    parser.add_argument("--scale-root", type=int, default=60, help="MIDI tonic; 60 means C")
    parser.add_argument("--scale-mode", choices=["major", "minor"], default="major")
    parser.add_argument("--beats-per-bar", type=int, default=4)
    parser.add_argument("--groove-slots", type=int, default=16)
    parser.add_argument("--large-interval-threshold", type=int, default=7)
    parser.add_argument("--stepwise-threshold", type=int, default=3)
    parser.add_argument("--qualified-note-beats", type=float, default=0.125)
    parser.add_argument("--rhythm-tolerance-beats", type=float, default=0.065)
    parser.add_argument("--target-pche", type=float, default=2.0)
    parser.add_argument("--pche-tolerance", type=float, default=0.65)
    parser.add_argument("--target-upc", type=float, default=5.0)
    parser.add_argument("--upc-tolerance", type=float, default=2.0)
    parser.add_argument("--target-compression-ratio", type=float, default=0.65)
    parser.add_argument("--compression-tolerance", type=float, default=0.35)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if bool(args.run_dir) == bool(args.midi_dir):
        raise SystemExit("Pass exactly one of --run-dir or --midi-dir.")
    if args.beats_per_bar < 1:
        raise SystemExit("--beats-per-bar must be at least 1.")
    if args.groove_slots < 1:
        raise SystemExit("--groove-slots must be at least 1.")


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    evaluate(args)


if __name__ == "__main__":
    main()
