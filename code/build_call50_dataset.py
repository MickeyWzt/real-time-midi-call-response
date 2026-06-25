"""
Build the final 50-Call dataset for objective evaluation.

Composition:
  - 10 artificial stress-test Calls from generate_call_dataset.py
  - 40 real two-bar monophonic melody snippets extracted from a public dataset,
    preferably POP909.

The extractor is conservative: it searches melody-like MIDI tracks, slides
two-bar windows, keeps windows with enough notes, limited overlap, and a clear
major/minor scale fit, then samples a reproducible subset.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


ROOT = Path(__file__).resolve().parents[1]
PROJECT_DEPS = ROOT / ".python_deps"
if PROJECT_DEPS.exists():
    sys.path.insert(0, str(PROJECT_DEPS))
sys.path.insert(0, str(ROOT / "code"))

from generate_call_dataset import call_specs, write_midi  # noqa: E402


MAJOR_SCALE = {0, 2, 4, 5, 7, 9, 11}
MINOR_SCALE = {0, 2, 3, 5, 7, 8, 10}


@dataclass(frozen=True)
class MidiNote:
    onset_ticks: int
    duration_ticks: int
    pitch: int
    velocity: int
    channel: int


@dataclass(frozen=True)
class RealSnippet:
    source_file: Path
    track_index: int
    track_name: str
    start_tick: int
    end_tick: int
    key_root: int
    key_mode: str
    scale_fit: float
    note_count: int
    notes: List[MidiNote]
    ticks_per_beat: int
    tempo: int


def require_runtime() -> None:
    try:
        import mido  # noqa: F401
    except ImportError as exc:
        raise SystemExit("Missing mido. Use C:\\Python312\\python.exe with project .python_deps.") from exc


def maybe_extract_zip(zip_path: Optional[str], extract_dir: Path) -> Optional[Path]:
    if not zip_path:
        return None
    archive = Path(zip_path)
    if not archive.exists():
        raise SystemExit(f"POP909 zip does not exist: {archive}")
    extract_dir.mkdir(parents=True, exist_ok=True)
    marker = extract_dir / ".extracted"
    if not marker.exists():
        print(f"[extract] {archive} -> {extract_dir}")
        with zipfile.ZipFile(archive, "r") as handle:
            handle.extractall(extract_dir)
        marker.write_text(str(archive), encoding="utf-8")
    return extract_dir


def find_midi_files(root: Path) -> List[Path]:
    return sorted(
        [
            item
            for item in root.rglob("*")
            if item.is_file() and item.suffix.lower() in {".mid", ".midi"}
        ]
    )


def track_name(track) -> str:
    for msg in track:
        if msg.type == "track_name":
            return str(msg.name)
    return ""


def notes_from_track(track, ticks_per_beat: int) -> Tuple[List[MidiNote], int]:
    abs_tick = 0
    tempo = 500000
    active: Dict[Tuple[int, int], Tuple[int, int]] = {}
    notes: List[MidiNote] = []

    for msg in track:
        abs_tick += msg.time
        if msg.type == "set_tempo":
            tempo = msg.tempo
            continue
        if msg.type == "note_on" and msg.velocity > 0:
            active[(msg.channel, msg.note)] = (abs_tick, msg.velocity)
            continue
        is_note_off = msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0)
        if is_note_off:
            start = active.pop((msg.channel, msg.note), None)
            if start is None:
                continue
            onset, velocity = start
            notes.append(
                MidiNote(
                    onset_ticks=onset,
                    duration_ticks=max(1, abs_tick - onset),
                    pitch=msg.note,
                    velocity=velocity,
                    channel=msg.channel,
                )
            )
    notes.sort(key=lambda item: (item.onset_ticks, item.pitch))
    return notes, tempo


def is_melody_track(name: str, index: int, total_tracks: int) -> bool:
    lowered = name.lower()
    if any(key in lowered for key in ["melody", "main", "vocal", "lead"]):
        return True
    if any(key in lowered for key in ["chord", "bridge", "piano left", "bass", "drum"]):
        return False
    return index <= 1 or total_tracks <= 2


def overlap_ratio(notes: List[MidiNote]) -> float:
    if len(notes) < 2:
        return 0.0
    overlaps = 0
    for left, right in zip(notes, notes[1:]):
        if left.onset_ticks + left.duration_ticks > right.onset_ticks:
            overlaps += 1
    return overlaps / max(1, len(notes) - 1)


def best_major_minor_fit(notes: List[MidiNote]) -> Tuple[int, str, float]:
    pcs = [note.pitch % 12 for note in notes]
    if not pcs:
        return 0, "major", 0.0
    best = (0, "major", -1.0)
    for root in range(12):
        major_fit = sum(1 for pc in pcs if (pc - root) % 12 in MAJOR_SCALE) / len(pcs)
        minor_fit = sum(1 for pc in pcs if (pc - root) % 12 in MINOR_SCALE) / len(pcs)
        if major_fit > best[2]:
            best = (root, "major", major_fit)
        if minor_fit > best[2]:
            best = (root, "minor", minor_fit)
    return best


def window_candidates(
    path: Path,
    min_notes: int,
    max_notes: int,
    min_scale_fit: float,
    max_overlap_ratio: float,
) -> List[RealSnippet]:
    import mido

    snippets: List[RealSnippet] = []
    try:
        midi = mido.MidiFile(path)
    except Exception as exc:
        print(f"[warn] cannot read {path}: {exc}")
        return []

    ticks_per_beat = midi.ticks_per_beat
    if ticks_per_beat <= 0:
        return []
    window_ticks = ticks_per_beat * 8
    hop_ticks = ticks_per_beat * 2
    total_tracks = len(midi.tracks)

    for idx, track in enumerate(midi.tracks):
        name = track_name(track)
        if not is_melody_track(name, idx, total_tracks):
            continue
        notes, tempo = notes_from_track(track, ticks_per_beat)
        if len(notes) < min_notes:
            continue
        if overlap_ratio(notes) > max_overlap_ratio:
            continue
        min_tick = notes[0].onset_ticks
        max_tick = max(note.onset_ticks + note.duration_ticks for note in notes)
        start = (min_tick // hop_ticks) * hop_ticks
        while start + window_ticks <= max_tick:
            end = start + window_ticks
            window_notes = [
                note
                for note in notes
                if start <= note.onset_ticks < end
            ]
            if min_notes <= len(window_notes) <= max_notes and overlap_ratio(window_notes) <= max_overlap_ratio:
                root, mode, fit = best_major_minor_fit(window_notes)
                if fit >= min_scale_fit:
                    normalized = [
                        MidiNote(
                            onset_ticks=note.onset_ticks - start,
                            duration_ticks=min(note.duration_ticks, max(1, end - note.onset_ticks)),
                            pitch=note.pitch,
                            velocity=note.velocity,
                            channel=note.channel,
                        )
                        for note in window_notes
                    ]
                    snippets.append(
                        RealSnippet(
                            source_file=path,
                            track_index=idx,
                            track_name=name,
                            start_tick=start,
                            end_tick=end,
                            key_root=root,
                            key_mode=mode,
                            scale_fit=fit,
                            note_count=len(normalized),
                            notes=normalized,
                            ticks_per_beat=ticks_per_beat,
                            tempo=tempo,
                        )
                    )
            start += hop_ticks
    return snippets


def write_real_snippet(snippet: RealSnippet, path: Path) -> None:
    import mido

    path.parent.mkdir(parents=True, exist_ok=True)
    mid = mido.MidiFile(ticks_per_beat=snippet.ticks_per_beat)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=snippet.tempo, time=0))
    track.append(mido.MetaMessage("time_signature", numerator=4, denominator=4, time=0))
    track.append(mido.Message("program_change", program=0, channel=0, time=0))

    events: List[Tuple[int, int, str, MidiNote]] = []
    for note in snippet.notes:
        events.append((note.onset_ticks, 0, "on", note))
        events.append((note.onset_ticks + note.duration_ticks, 1, "off", note))
    events.sort(key=lambda item: (item[0], item[1]))

    last_tick = 0
    for tick, _, kind, note in events:
        delta = max(0, tick - last_tick)
        if kind == "on":
            track.append(mido.Message("note_on", note=note.pitch, velocity=note.velocity, channel=0, time=delta))
        else:
            track.append(mido.Message("note_off", note=note.pitch, velocity=0, channel=0, time=delta))
        last_tick = tick

    end_tick = snippet.ticks_per_beat * 8
    track.append(mido.MetaMessage("end_of_track", time=max(0, end_tick - last_tick)))
    mid.save(path)


def write_manifest(output_dir: Path, rows: List[Dict[str, object]]) -> None:
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with (output_dir / "call50_manifest.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_dataset(args: argparse.Namespace) -> None:
    require_runtime()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    for idx, spec in enumerate(call_specs(), start=1):
        call_id = f"A{idx:02d}"
        path = output_dir / f"{call_id}_{spec.name}.mid"
        write_midi(spec, path)
        rows.append(
            {
                "call_id": call_id,
                "origin": "artificial_stress",
                "name": spec.name,
                "category": spec.category,
                "description": spec.description,
                "note_count": len(spec.notes),
                "midi_path": str(path),
            }
        )

    dataset_root = Path(args.dataset_root) if args.dataset_root else maybe_extract_zip(args.dataset_zip, ROOT / "ab_tests" / "pop909_extracted")
    if dataset_root is None:
        raise SystemExit("Pass --dataset-root for an extracted POP909/Lakh MIDI directory, or --dataset-zip for a local zip.")
    if not dataset_root.exists():
        raise SystemExit(f"Dataset root does not exist: {dataset_root}")

    midi_files = find_midi_files(dataset_root)
    if not midi_files:
        raise SystemExit(f"No MIDI files found under dataset root: {dataset_root}")
    print(f"[scan] midi_files={len(midi_files)} root={dataset_root}")

    snippets: List[RealSnippet] = []
    for midi_path in midi_files:
        snippets.extend(
            window_candidates(
                midi_path,
                min_notes=args.min_notes,
                max_notes=args.max_notes,
                min_scale_fit=args.min_scale_fit,
                max_overlap_ratio=args.max_overlap_ratio,
            )
        )
    if len(snippets) < args.real_count:
        raise SystemExit(
            f"Only found {len(snippets)} usable real snippets, need {args.real_count}. "
            "Try lowering --min-scale-fit or pass a larger dataset."
        )

    rng = random.Random(args.seed)
    rng.shuffle(snippets)
    selected = snippets[: args.real_count]
    for idx, snippet in enumerate(selected, start=1):
        call_id = f"R{idx:02d}"
        safe_source = snippet.source_file.stem.replace(" ", "_")
        path = output_dir / f"{call_id}_{safe_source}.mid"
        write_real_snippet(snippet, path)
        rows.append(
            {
                "call_id": call_id,
                "origin": "real_public_dataset",
                "name": safe_source,
                "category": "real_two_bar_monophonic_major_minor",
                "description": "Two-bar monophonic melody snippet sampled from public MIDI dataset.",
                "source_file": str(snippet.source_file),
                "track_index": snippet.track_index,
                "track_name": snippet.track_name,
                "start_tick": snippet.start_tick,
                "end_tick": snippet.end_tick,
                "key_root_pc": snippet.key_root,
                "key_mode": snippet.key_mode,
                "scale_fit": f"{snippet.scale_fit:.6f}",
                "note_count": snippet.note_count,
                "midi_path": str(path),
            }
        )

    write_manifest(output_dir, rows)
    print(f"[done] output_dir={output_dir}")
    print(f"[done] artificial=10 real={len(selected)} total={len(rows)}")
    print(f"[done] manifest={output_dir / 'call50_manifest.csv'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build 50-call dataset from artificial stress tests plus public MIDI snippets")
    parser.add_argument("--dataset-root", default=None, help="extracted POP909/Lakh MIDI directory")
    parser.add_argument("--dataset-zip", default=None, help="local POP909/Lakh zip to extract and scan")
    parser.add_argument("--output-dir", default=str(ROOT / "ab_tests" / "calls_50"))
    parser.add_argument("--seed", type=int, default=20260529)
    parser.add_argument("--real-count", type=int, default=40)
    parser.add_argument("--min-notes", type=int, default=6)
    parser.add_argument("--max-notes", type=int, default=24)
    parser.add_argument("--min-scale-fit", type=float, default=0.85)
    parser.add_argument("--max-overlap-ratio", type=float, default=0.10)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    build_dataset(args)


if __name__ == "__main__":
    main()
