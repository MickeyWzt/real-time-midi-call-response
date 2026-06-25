"""
Generate a 10-Call MIDI test dataset for objective Call-and-Response experiments.

The calls cover different prompt conditions: pentatonic, blues/chromatic
tension, antecedent half cadence, repetition tail, syncopation, sparse long
tones, dense motion, arch contour, descending contour, and leap gap-fill.
"""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List


ROOT = Path(__file__).resolve().parents[1]
PROJECT_DEPS = ROOT / ".python_deps"
if PROJECT_DEPS.exists():
    sys.path.insert(0, str(PROJECT_DEPS))


@dataclass(frozen=True)
class CallNote:
    onset_beats: float
    pitch: int
    duration_beats: float
    velocity: int = 90


@dataclass(frozen=True)
class CallSpec:
    call_id: str
    name: str
    category: str
    description: str
    bpm: float
    notes: List[CallNote]


def call_specs() -> List[CallSpec]:
    return [
        CallSpec(
            "C01",
            "c_pentatonic_balanced",
            "pentatonic_clear_motif",
            "Two-bar C major pentatonic motif with balanced upward and downward motion.",
            100.0,
            [
                CallNote(0.0, 60, 0.5),
                CallNote(0.5, 62, 0.5),
                CallNote(1.0, 64, 0.75),
                CallNote(1.875, 67, 0.375),
                CallNote(2.25, 69, 0.625),
                CallNote(3.0, 67, 0.5),
                CallNote(3.5, 64, 0.5),
                CallNote(4.0, 62, 0.625),
                CallNote(4.75, 60, 0.375),
                CallNote(5.125, 62, 0.375),
                CallNote(5.5, 64, 0.5),
                CallNote(6.0, 67, 0.75),
                CallNote(6.875, 69, 0.375),
                CallNote(7.25, 67, 0.75),
            ],
        ),
        CallSpec(
            "C02",
            "a_minor_pentatonic_question",
            "modal_question",
            "A minor pentatonic call ending high, designed as an unanswered question.",
            92.0,
            [
                CallNote(0.0, 57, 0.5),
                CallNote(0.5, 60, 0.5),
                CallNote(1.0, 62, 1.0),
                CallNote(2.0, 64, 0.5),
                CallNote(2.5, 67, 0.5),
                CallNote(3.0, 69, 1.0),
                CallNote(4.0, 67, 0.5),
                CallNote(4.5, 64, 0.5),
                CallNote(5.0, 62, 0.75),
                CallNote(5.875, 64, 0.375),
                CallNote(6.25, 67, 0.5),
                CallNote(6.75, 69, 1.0),
            ],
        ),
        CallSpec(
            "C03",
            "a_blues_tension",
            "blues_chromatic",
            "A blues-oriented call with flat-third and blue-note chromatic tension.",
            108.0,
            [
                CallNote(0.0, 57, 0.5),
                CallNote(0.5, 60, 0.5),
                CallNote(1.0, 62, 0.5),
                CallNote(1.5, 63, 0.25),
                CallNote(1.75, 62, 0.75),
                CallNote(2.5, 60, 0.5),
                CallNote(3.0, 57, 1.0),
                CallNote(4.0, 60, 0.5),
                CallNote(4.5, 62, 0.5),
                CallNote(5.0, 63, 0.25),
                CallNote(5.25, 64, 0.25),
                CallNote(5.5, 67, 0.75),
                CallNote(6.5, 60, 1.0),
            ],
        ),
        CallSpec(
            "C04",
            "classical_antecedent",
            "antecedent_half_cadence",
            "Classical antecedent phrase ending on the dominant to invite resolution.",
            100.0,
            [
                CallNote(0.0, 60, 1.0),
                CallNote(1.0, 62, 1.0),
                CallNote(2.0, 64, 1.5),
                CallNote(3.5, 65, 0.5),
                CallNote(4.0, 64, 0.5),
                CallNote(4.5, 62, 0.5),
                CallNote(5.0, 60, 0.75),
                CallNote(5.875, 62, 0.375),
                CallNote(6.25, 64, 0.5),
                CallNote(6.75, 67, 1.0),
            ],
        ),
        CallSpec(
            "C05",
            "repetition_tail",
            "repetition_stress_test",
            "Normal opening followed by a repeated A tail to test anti-copy controls.",
            104.0,
            [
                CallNote(0.0, 60, 0.5),
                CallNote(0.5, 64, 0.5),
                CallNote(1.0, 67, 0.75),
                CallNote(2.0, 69, 0.375),
                CallNote(2.375, 69, 0.375),
                CallNote(2.75, 69, 0.375),
                CallNote(3.125, 69, 0.375),
                CallNote(3.5, 69, 1.0),
                CallNote(5.0, 67, 0.5),
                CallNote(5.5, 69, 0.375),
                CallNote(5.875, 69, 0.375),
                CallNote(6.25, 69, 1.0),
            ],
        ),
        CallSpec(
            "C06",
            "syncopated_pop",
            "syncopation",
            "Pentatonic call with off-beat entries and tied-feeling durations.",
            106.0,
            [
                CallNote(0.0, 60, 0.375),
                CallNote(0.75, 62, 0.5),
                CallNote(1.5, 64, 0.375),
                CallNote(2.0, 67, 0.75),
                CallNote(2.875, 69, 0.375),
                CallNote(3.5, 67, 0.5),
                CallNote(4.25, 64, 0.375),
                CallNote(4.75, 62, 0.5),
                CallNote(5.5, 60, 0.375),
                CallNote(6.0, 62, 0.75),
                CallNote(6.875, 64, 0.375),
                CallNote(7.25, 67, 0.5),
            ],
        ),
        CallSpec(
            "C07",
            "sparse_long_tones",
            "sparse_cadential",
            "Sparse long-tone call testing whether responses avoid over-fragmentation.",
            88.0,
            [
                CallNote(0.0, 60, 1.25),
                CallNote(1.5, 64, 1.0),
                CallNote(3.0, 67, 1.25),
                CallNote(4.75, 69, 0.75),
                CallNote(5.75, 67, 1.0),
                CallNote(7.0, 60, 1.0),
            ],
        ),
        CallSpec(
            "C08",
            "dense_eighth_motion",
            "dense_motion",
            "Dense eighth-note pentatonic motion testing rhythmic stability and note count control.",
            112.0,
            [
                CallNote(0.0, 60, 0.5),
                CallNote(0.5, 62, 0.5),
                CallNote(1.0, 64, 0.5),
                CallNote(1.5, 67, 0.5),
                CallNote(2.0, 69, 0.5),
                CallNote(2.5, 67, 0.5),
                CallNote(3.0, 64, 0.5),
                CallNote(3.5, 62, 0.5),
                CallNote(4.0, 60, 0.5),
                CallNote(4.5, 62, 0.5),
                CallNote(5.0, 64, 0.5),
                CallNote(5.5, 67, 0.5),
                CallNote(6.0, 69, 0.5),
                CallNote(6.5, 67, 0.5),
                CallNote(7.0, 64, 0.5),
                CallNote(7.5, 60, 0.5),
            ],
        ),
        CallSpec(
            "C09",
            "arched_contour",
            "arched_contour",
            "Clear arch contour rising to a peak then returning, testing contour continuation.",
            96.0,
            [
                CallNote(0.0, 55, 0.75),
                CallNote(0.75, 60, 0.5),
                CallNote(1.25, 62, 0.5),
                CallNote(1.75, 64, 0.75),
                CallNote(2.5, 67, 0.5),
                CallNote(3.0, 72, 1.0),
                CallNote(4.25, 69, 0.5),
                CallNote(4.75, 67, 0.5),
                CallNote(5.25, 64, 0.75),
                CallNote(6.0, 62, 0.5),
                CallNote(6.5, 60, 1.0),
            ],
        ),
        CallSpec(
            "C10",
            "leap_gap_fill",
            "large_leap_gap_fill",
            "Large leaps followed by opposite stepwise gap-fill motion.",
            100.0,
            [
                CallNote(0.0, 60, 0.5),
                CallNote(0.5, 72, 0.5),
                CallNote(1.0, 69, 0.5),
                CallNote(1.5, 67, 0.5),
                CallNote(2.0, 64, 0.75),
                CallNote(3.0, 55, 0.5),
                CallNote(3.5, 60, 0.5),
                CallNote(4.0, 62, 0.5),
                CallNote(4.5, 64, 0.75),
                CallNote(5.5, 76, 0.5),
                CallNote(6.0, 72, 0.5),
                CallNote(6.5, 69, 0.5),
                CallNote(7.0, 67, 0.75),
            ],
        ),
    ]


def write_midi(spec: CallSpec, path: Path, ticks_per_beat: int = 480) -> None:
    import mido

    path.parent.mkdir(parents=True, exist_ok=True)
    mid = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(spec.bpm), time=0))
    track.append(mido.MetaMessage("time_signature", numerator=4, denominator=4, time=0))
    track.append(mido.Message("program_change", program=0, channel=0, time=0))

    events: List[tuple[int, int, str, int, int]] = []
    for note in spec.notes:
        onset = int(round(note.onset_beats * ticks_per_beat))
        duration = max(1, int(round(note.duration_beats * ticks_per_beat)))
        events.append((onset, 0, "on", note.pitch, note.velocity))
        events.append((onset + duration, 1, "off", note.pitch, 0))
    events.sort()

    last_tick = 0
    for tick, _, kind, pitch, velocity in events:
        delta = max(0, tick - last_tick)
        if kind == "on":
            track.append(mido.Message("note_on", note=pitch, velocity=velocity, channel=0, time=delta))
        else:
            track.append(mido.Message("note_off", note=pitch, velocity=0, channel=0, time=delta))
        last_tick = tick

    end_tick = ticks_per_beat * 8
    track.append(mido.MetaMessage("end_of_track", time=max(0, end_tick - last_tick)))
    mid.save(path)


def write_metadata(specs: List[CallSpec], output_dir: Path) -> None:
    fields = [
        "call_id",
        "name",
        "category",
        "description",
        "bpm",
        "note_count",
        "duration_beats",
        "midi_path",
    ]
    with (output_dir / "call_dataset_manifest.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for spec in specs:
            end_beat = max(note.onset_beats + note.duration_beats for note in spec.notes)
            writer.writerow(
                {
                    "call_id": spec.call_id,
                    "name": spec.name,
                    "category": spec.category,
                    "description": spec.description,
                    "bpm": spec.bpm,
                    "note_count": len(spec.notes),
                    "duration_beats": f"{end_beat:.3f}",
                    "midi_path": str(output_dir / f"{spec.call_id}_{spec.name}.mid"),
                }
            )


def main() -> None:
    output_dir = ROOT / "ab_tests" / "calls_10"
    specs = call_specs()
    output_dir.mkdir(parents=True, exist_ok=True)
    for spec in specs:
        write_midi(spec, output_dir / f"{spec.call_id}_{spec.name}.mid")
    write_metadata(specs, output_dir)
    print(f"[done] call_dataset_dir={output_dir}")
    print(f"[done] calls={len(specs)}")
    print(f"[done] manifest={output_dir / 'call_dataset_manifest.csv'}")


if __name__ == "__main__":
    main()
