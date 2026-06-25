from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import mido


OUT_DIR = Path(__file__).resolve().parent
AB_DIR = OUT_DIR.parent
MFP_DIR = AB_DIR.parent
CALL50_DIR = AB_DIR / "calls_50"
CALL50_MANIFEST = CALL50_DIR / "call50_manifest.csv"
CALLS_DIR = OUT_DIR / "calls"

DATASET_VERSION = "call100_public_final_v1"

FIELDS = [
    "call_id",
    "origin",
    "source_dataset",
    "source_group",
    "name",
    "category",
    "sub_category",
    "description",
    "midi_path",
    "source_file",
    "track_index",
    "track_name",
    "start_sec",
    "end_sec",
    "start_tick",
    "end_tick",
    "bpm",
    "meter",
    "bars",
    "duration_sec",
    "note_count",
    "pitch_min",
    "pitch_max",
    "pitch_range",
    "onset_density",
    "mean_ioi",
    "std_ioi",
    "polyphony_rate",
    "has_chord",
    "has_pause",
    "max_internal_pause",
    "has_repetition_tail",
    "has_chromatic_outside",
    "key_root_pc",
    "key_mode",
    "scale_fit",
    "license_note",
    "citation_key",
    "split",
    "fingerprint_sha1",
]

CATEGORY_QUOTAS = {
    "clear_motif_melody": 12,
    "expressive_rubato": 8,
    "chord_arpeggio_polyphonic": 8,
    "sparse_short": 5,
    "dense_fast": 5,
    "chromatic_outside": 5,
    "false_ending_pause": 4,
    "repetition_tail": 3,
}

EXPECTED_PUBLIC_SOURCES = {
    "POP909": 10,
    "MAESTRO_v3_midi_only": 15,
    "ASAP": 10,
    "Lakh_MIDI_Clean_or_Slakh2100": 10,
    "GiantMIDI_or_MAESTRO_complex": 5,
}

SOURCE_LICENSES = {
    "POP909": "POP909 local repository includes MIT LICENSE; cite POP909 ISMIR 2020.",
    "MAESTRO_v3_midi_only": "MAESTRO v3.0.0 MIDI-only source not present locally in this build.",
    "ASAP": "ASAP source not present locally in this build.",
    "Lakh_MIDI_Clean_or_Slakh2100": "Lakh MIDI Clean or Slakh2100 source not present locally in this build.",
    "GiantMIDI_or_MAESTRO_complex": "GiantMIDI-Piano or MAESTRO complex source not present locally in this build.",
    "calls_50": "Existing local Call50 benchmark material copied without modifying calls_50.",
}

CITATIONS = {
    "POP909": "pop909-ismir2020",
    "MAESTRO_v3_midi_only": "maestro-v3.0.0",
    "ASAP": "asap-dataset",
    "Lakh_MIDI_Clean_or_Slakh2100": "lakh-midi-clean-or-slakh2100",
    "GiantMIDI_or_MAESTRO_complex": "giantmidi-piano-or-maestro-complex",
    "calls_50": "call50_manifest",
}

MAJOR_SCALE = {0, 2, 4, 5, 7, 9, 11}
MINOR_SCALE = {0, 2, 3, 5, 7, 8, 10}


@dataclass(frozen=True)
class Note:
    start_tick: int
    end_tick: int
    start_sec: float
    end_sec: float
    pitch: int
    velocity: int
    channel: int


@dataclass
class Candidate:
    source_dataset: str
    source_group: str
    source_file: Path
    track_index: int
    track_name: str
    start_tick: int
    end_tick: int
    start_sec: float
    end_sec: float
    bpm: float
    meter: str
    bars: float
    ticks_per_beat: int
    tempo: int
    numerator: int
    denominator: int
    notes: List[Note]
    features: Dict[str, object]
    score_by_category: Dict[str, float]
    sub_category: str
    source_song: str
    midi_bytes: bytes = field(default=b"")
    fingerprint_sha1: str = ""

    @property
    def key(self) -> Tuple[str, int, int, int]:
        return (str(self.source_file), self.track_index, self.start_tick, self.end_tick)


def fmt(value: object, digits: int = 6) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return ""
        return f"{value:.{digits}f}".rstrip("0").rstrip(".")
    return str(value)


def slugify(value: str, max_len: int = 64) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    value = re.sub(r"_+", "_", value).strip("_")
    return (value or "item")[:max_len]


def sha1_file(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha1_bytes(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()


def load_midi(path: Path) -> Optional[mido.MidiFile]:
    try:
        return mido.MidiFile(path)
    except Exception:
        return None


def absolute_track_ticks(track: mido.MidiTrack) -> Iterable[Tuple[int, mido.Message]]:
    tick = 0
    for msg in track:
        tick += msg.time
        yield tick, msg


def tempo_events(mid: mido.MidiFile) -> List[Tuple[int, int]]:
    events = [(0, 500000)]
    for track in mid.tracks:
        for tick, msg in absolute_track_ticks(track):
            if msg.type == "set_tempo":
                events.append((tick, int(msg.tempo)))
    events.sort(key=lambda item: item[0])
    deduped: List[Tuple[int, int]] = []
    for tick, tempo in events:
        if deduped and deduped[-1][0] == tick:
            deduped[-1] = (tick, tempo)
        else:
            deduped.append((tick, tempo))
    return deduped


def tick_to_seconds_func(mid: mido.MidiFile):
    events = tempo_events(mid)
    ticks_per_beat = mid.ticks_per_beat

    def tick_to_seconds(tick: int) -> float:
        tick = max(0, int(tick))
        total = 0.0
        last_tick = 0
        current_tempo = events[0][1]
        for event_tick, event_tempo in events[1:]:
            if tick <= event_tick:
                break
            total += mido.tick2second(event_tick - last_tick, ticks_per_beat, current_tempo)
            last_tick = event_tick
            current_tempo = event_tempo
        total += mido.tick2second(tick - last_tick, ticks_per_beat, current_tempo)
        return total

    return tick_to_seconds


def first_tempo(mid: mido.MidiFile) -> int:
    return tempo_events(mid)[0][1]


def first_meter(mid: mido.MidiFile) -> Tuple[int, int]:
    for track in mid.tracks:
        for _, msg in absolute_track_ticks(track):
            if msg.type == "time_signature":
                return int(msg.numerator), int(msg.denominator)
    return 4, 4


def max_track_tick(mid: mido.MidiFile) -> int:
    return max((sum(msg.time for msg in track) for track in mid.tracks), default=0)


def track_name(track: mido.MidiTrack, fallback: str) -> str:
    for msg in track:
        if msg.type == "track_name":
            return str(msg.name)
    return fallback


def track_program(track: mido.MidiTrack) -> int:
    for msg in track:
        if msg.type == "program_change" and getattr(msg, "channel", 0) != 9:
            return int(msg.program)
    return 0


def notes_from_track(mid: mido.MidiFile, track_index: int) -> List[Note]:
    if track_index >= len(mid.tracks):
        return []
    tick_to_seconds = tick_to_seconds_func(mid)
    active: Dict[Tuple[int, int], List[Tuple[int, int]]] = defaultdict(list)
    notes: List[Note] = []
    for tick, msg in absolute_track_ticks(mid.tracks[track_index]):
        if msg.type not in ("note_on", "note_off"):
            continue
        channel = int(getattr(msg, "channel", 0))
        if channel == 9:
            continue
        pitch = int(msg.note)
        key = (channel, pitch)
        if msg.type == "note_on" and int(msg.velocity) > 0:
            active[key].append((tick, int(msg.velocity)))
        else:
            if active[key]:
                start_tick, velocity = active[key].pop(0)
                if tick > start_tick:
                    notes.append(
                        Note(
                            start_tick=start_tick,
                            end_tick=tick,
                            start_sec=tick_to_seconds(start_tick),
                            end_sec=tick_to_seconds(tick),
                            pitch=pitch,
                            velocity=velocity,
                            channel=channel,
                        )
                    )
    return notes


def notes_from_midi(mid: mido.MidiFile) -> List[Note]:
    notes: List[Note] = []
    for idx in range(len(mid.tracks)):
        notes.extend(notes_from_track(mid, idx))
    return sorted(notes, key=lambda n: (n.start_sec, n.pitch, n.end_sec))


def has_any_drum_notes(mid: mido.MidiFile) -> bool:
    for track in mid.tracks:
        for _, msg in absolute_track_ticks(track):
            if msg.type in ("note_on", "note_off") and getattr(msg, "channel", -1) == 9:
                return True
    return False


def estimate_duration(mid: mido.MidiFile, notes: Optional[Sequence[Note]] = None) -> float:
    try:
        length = float(mid.length)
    except Exception:
        tick_to_seconds = tick_to_seconds_func(mid)
        length = tick_to_seconds(max_track_tick(mid))
    if notes:
        length = max(length, max(n.end_sec for n in notes))
    return max(0.0, length)


def estimate_bars(mid: mido.MidiFile) -> float:
    numerator, denominator = first_meter(mid)
    bar_ticks = mid.ticks_per_beat * numerator * 4.0 / denominator
    if bar_ticks <= 0:
        return 0.0
    return max_track_tick(mid) / bar_ticks


def key_estimate(pitches: Sequence[int]) -> Tuple[int, str, float]:
    if not pitches:
        return -1, "", 0.0
    pcs = [p % 12 for p in pitches]
    best = (-1, "", -1.0)
    for root in range(12):
        major = {(pc + root) % 12 for pc in MAJOR_SCALE}
        minor = {(pc + root) % 12 for pc in MINOR_SCALE}
        major_fit = sum(1 for pc in pcs if pc in major) / len(pcs)
        minor_fit = sum(1 for pc in pcs if pc in minor) / len(pcs)
        if major_fit > best[2]:
            best = (root, "major", major_fit)
        if minor_fit > best[2]:
            best = (root, "minor", minor_fit)
    return best


def polyphony_rate(notes: Sequence[Note], duration_sec: float) -> float:
    if not notes or duration_sec <= 0:
        return 0.0
    events: List[Tuple[float, int]] = []
    for note in notes:
        events.append((max(0.0, note.start_sec), 1))
        events.append((max(0.0, note.end_sec), -1))
    events.sort(key=lambda item: (item[0], item[1]))
    active = 0
    last_t = 0.0
    poly_time = 0.0
    for t, delta in events:
        t = min(duration_sec, max(0.0, t))
        if t > last_t and active >= 2:
            poly_time += t - last_t
        active += delta
        last_t = t
    if duration_sec > last_t and active >= 2:
        poly_time += duration_sec - last_t
    return max(0.0, min(1.0, poly_time / duration_sec))


def pause_profile(notes: Sequence[Note], duration_sec: float) -> Tuple[float, float, float]:
    if not notes:
        return duration_sec, duration_sec, duration_sec
    ordered = sorted(notes, key=lambda n: (n.start_sec, n.end_sec))
    first_gap = max(0.0, ordered[0].start_sec)
    trailing_gap = max(0.0, duration_sec - max(n.end_sec for n in ordered))
    max_gap = max(first_gap, trailing_gap)
    current_end = ordered[0].end_sec
    for note in ordered[1:]:
        if note.start_sec > current_end:
            max_gap = max(max_gap, note.start_sec - current_end)
        current_end = max(current_end, note.end_sec)
    return max_gap, first_gap, trailing_gap


def has_repetition_tail(notes: Sequence[Note]) -> bool:
    ordered = sorted(notes, key=lambda n: (n.start_sec, n.pitch))
    pitches = [n.pitch for n in ordered]
    if len(pitches) < 3:
        return False
    tail = pitches[-5:]
    if any(tail[i] == tail[i - 1] for i in range(1, len(tail))):
        return True
    if len(tail) >= 4 and tail[-2:] == tail[-4:-2]:
        return True
    intervals = [tail[i] - tail[i - 1] for i in range(1, len(tail))]
    return len(intervals) >= 3 and intervals[-1] == intervals[-2]


def has_chord(notes: Sequence[Note]) -> bool:
    starts = sorted(n.start_sec for n in notes)
    for i in range(1, len(starts)):
        if abs(starts[i] - starts[i - 1]) <= 0.035:
            return True
    ordered = sorted(notes, key=lambda n: n.start_sec)
    for i, note in enumerate(ordered):
        active = 0
        for other in ordered:
            if other.start_sec <= note.start_sec < other.end_sec:
                active += 1
        if active >= 2:
            return True
    return False


def chromatic_signal(notes: Sequence[Note], scale_fit: float) -> bool:
    ordered = sorted(notes, key=lambda n: (n.start_sec, n.pitch))
    pitches = [n.pitch for n in ordered]
    if scale_fit < 0.82:
        return True
    if len(pitches) >= 4:
        semitone_steps = sum(1 for a, b in zip(pitches, pitches[1:]) if abs(a - b) == 1)
        if semitone_steps >= 3:
            return True
    return False


def compute_features(notes: Sequence[Note], duration_sec: float) -> Optional[Dict[str, object]]:
    if duration_sec <= 0 or not notes:
        return None
    pitches = [n.pitch for n in notes]
    if not pitches:
        return None
    pitch_min = min(pitches)
    pitch_max = max(pitches)
    unique_onsets = sorted(set(round(n.start_sec, 6) for n in notes))
    iois = [b - a for a, b in zip(unique_onsets, unique_onsets[1:]) if b >= a]
    mean_ioi = sum(iois) / len(iois) if iois else 0.0
    std_ioi = math.sqrt(sum((x - mean_ioi) ** 2 for x in iois) / len(iois)) if iois else 0.0
    root, mode, scale_fit = key_estimate(pitches)
    max_pause, first_gap, trailing_gap = pause_profile(notes, duration_sec)
    chord = has_chord(notes)
    repetition = has_repetition_tail(notes)
    chromatic = chromatic_signal(notes, scale_fit)
    return {
        "duration_sec": duration_sec,
        "note_count": len(notes),
        "pitch_min": pitch_min,
        "pitch_max": pitch_max,
        "pitch_range": pitch_max - pitch_min,
        "onset_density": len(notes) / duration_sec if duration_sec > 0 else 0.0,
        "mean_ioi": mean_ioi,
        "std_ioi": std_ioi,
        "polyphony_rate": polyphony_rate(notes, duration_sec),
        "has_chord": chord,
        "has_pause": max_pause >= 0.45,
        "max_internal_pause": max_pause,
        "has_repetition_tail": repetition,
        "has_chromatic_outside": chromatic,
        "key_root_pc": root,
        "key_mode": mode,
        "scale_fit": scale_fit,
        "_first_gap": first_gap,
        "_trailing_gap": trailing_gap,
    }


def score_candidate(features: Dict[str, object], track: str, dataset: str) -> Dict[str, float]:
    note_count = float(features["note_count"])
    duration = float(features["duration_sec"])
    pitch_range = float(features["pitch_range"])
    onset_density = float(features["onset_density"])
    mean_ioi = float(features["mean_ioi"])
    std_ioi = float(features["std_ioi"])
    poly = float(features["polyphony_rate"])
    scale_fit = float(features["scale_fit"])
    max_pause = float(features["max_internal_pause"])
    trailing_gap = float(features.get("_trailing_gap", 0.0))
    chord = bool(features["has_chord"])
    chromatic = bool(features["has_chromatic_outside"])
    repetition = bool(features["has_repetition_tail"])
    track_upper = track.upper()
    melodic_track = 1.0 if any(k in track_upper for k in ("MELODY", "BRIDGE", "LEAD")) else 0.0
    piano_track = 1.0 if "PIANO" in track_upper or dataset.startswith("MAESTRO") else 0.0

    def closeness(value: float, target: float, span: float) -> float:
        return max(0.0, 1.0 - abs(value - target) / span)

    return {
        "clear_motif_melody": (
            2.0 * melodic_track
            + 1.2 * closeness(note_count, 10, 12)
            + 1.0 * closeness(pitch_range, 10, 14)
            + 1.2 * scale_fit
            + max(0.0, 1.0 - poly * 3.0)
        ),
        "expressive_rubato": (
            melodic_track
            + min(2.0, std_ioi * 5.0)
            + min(1.0, mean_ioi * 2.0)
            + closeness(onset_density, 3.0, 4.0)
            + 0.3 * scale_fit
        ),
        "chord_arpeggio_polyphonic": (
            1.5 * piano_track
            + (1.5 if chord else 0.0)
            + min(2.0, poly * 5.0)
            + min(1.5, note_count / 12.0)
            + closeness(pitch_range, 22, 24)
        ),
        "sparse_short": (
            max(0.0, 2.0 - note_count / 5.0)
            + max(0.0, 2.0 - onset_density / 1.8)
            + min(1.5, max_pause * 1.5)
            + closeness(duration, 3.0, 4.0)
        ),
        "dense_fast": (
            min(2.5, onset_density / 3.0)
            + min(2.0, note_count / 14.0)
            + max(0.0, 1.2 - mean_ioi * 2.0)
            + min(1.0, pitch_range / 18.0)
        ),
        "chromatic_outside": (
            (2.5 if chromatic else 0.0)
            + max(0.0, 2.0 - scale_fit * 2.0)
            + min(1.0, pitch_range / 18.0)
            + min(1.0, note_count / 12.0)
        ),
        "false_ending_pause": (
            min(2.5, trailing_gap * 2.0)
            + min(1.5, max_pause * 1.2)
            + melodic_track
            + closeness(note_count, 7, 8)
        ),
        "repetition_tail": (
            (3.0 if repetition else 0.0)
            + melodic_track
            + closeness(note_count, 8, 10)
            + 0.5 * scale_fit
        ),
    }


def feature_fields(features: Dict[str, object]) -> Dict[str, str]:
    return {
        "duration_sec": fmt(features.get("duration_sec")),
        "note_count": fmt(features.get("note_count"), 0),
        "pitch_min": fmt(features.get("pitch_min"), 0),
        "pitch_max": fmt(features.get("pitch_max"), 0),
        "pitch_range": fmt(features.get("pitch_range"), 0),
        "onset_density": fmt(features.get("onset_density")),
        "mean_ioi": fmt(features.get("mean_ioi")),
        "std_ioi": fmt(features.get("std_ioi")),
        "polyphony_rate": fmt(features.get("polyphony_rate")),
        "has_chord": fmt(features.get("has_chord")),
        "has_pause": fmt(features.get("has_pause")),
        "max_internal_pause": fmt(features.get("max_internal_pause")),
        "has_repetition_tail": fmt(features.get("has_repetition_tail")),
        "has_chromatic_outside": fmt(features.get("has_chromatic_outside")),
        "key_root_pc": fmt(features.get("key_root_pc"), 0),
        "key_mode": fmt(features.get("key_mode")),
        "scale_fit": fmt(features.get("scale_fit")),
    }


def candidate_to_midi_bytes(candidate: Candidate) -> bytes:
    mid = mido.MidiFile(type=1, ticks_per_beat=candidate.ticks_per_beat)
    meta = mido.MidiTrack()
    meta.append(mido.MetaMessage("set_tempo", tempo=int(candidate.tempo), time=0))
    meta.append(
        mido.MetaMessage(
            "time_signature",
            numerator=int(candidate.numerator),
            denominator=int(candidate.denominator),
            time=0,
        )
    )
    meta.append(mido.MetaMessage("end_of_track", time=max(1, candidate.end_tick - candidate.start_tick)))
    mid.tracks.append(meta)

    track = mido.MidiTrack()
    track.append(mido.MetaMessage("track_name", name=candidate.track_name[:80], time=0))
    program = 0
    source_mid = load_midi(candidate.source_file)
    if source_mid is not None and candidate.track_index < len(source_mid.tracks):
        program = track_program(source_mid.tracks[candidate.track_index])
    track.append(mido.Message("program_change", channel=0, program=program, time=0))

    events: List[Tuple[int, int, mido.Message]] = []
    duration_ticks = max(1, candidate.end_tick - candidate.start_tick)
    for note in candidate.notes:
        start = max(0, min(duration_ticks, int(note.start_tick)))
        end = max(start + 1, min(duration_ticks, int(note.end_tick)))
        events.append((start, 1, mido.Message("note_on", channel=0, note=note.pitch, velocity=max(1, note.velocity), time=0)))
        events.append((end, 0, mido.Message("note_off", channel=0, note=note.pitch, velocity=0, time=0)))
    events.sort(key=lambda item: (item[0], item[1], item[2].note))
    last_tick = 0
    for tick, _, msg in events:
        msg.time = max(0, tick - last_tick)
        track.append(msg)
        last_tick = tick
    track.append(mido.MetaMessage("end_of_track", time=max(1, duration_ticks - last_tick)))
    mid.tracks.append(track)

    buf = io.BytesIO()
    mid.save(file=buf)
    return buf.getvalue()


def relative_notes(notes: Sequence[Note], start_tick: int, end_tick: int, tick_to_sec) -> List[Note]:
    rel: List[Note] = []
    base_sec = tick_to_sec(start_tick)
    for note in notes:
        if note.start_tick < start_tick or note.start_tick >= end_tick:
            continue
        clipped_end = min(note.end_tick, end_tick)
        if clipped_end <= note.start_tick:
            continue
        rel_start = note.start_tick - start_tick
        rel_end = clipped_end - start_tick
        rel.append(
            Note(
                start_tick=rel_start,
                end_tick=rel_end,
                start_sec=max(0.0, tick_to_sec(note.start_tick) - base_sec),
                end_sec=max(0.0, tick_to_sec(clipped_end) - base_sec),
                pitch=note.pitch,
                velocity=note.velocity,
                channel=note.channel,
            )
        )
    return rel


def build_pop909_candidates(pop_root: Path) -> List[Candidate]:
    candidates: List[Candidate] = []
    if not pop_root.exists():
        return candidates
    for song_dir in sorted(p for p in pop_root.iterdir() if p.is_dir()):
        midi_path = song_dir / f"{song_dir.name}.mid"
        if not midi_path.exists():
            continue
        mid = load_midi(midi_path)
        if mid is None:
            continue
        tick_to_sec = tick_to_seconds_func(mid)
        tempo = first_tempo(mid)
        bpm = 60000000.0 / tempo if tempo else 120.0
        numerator, denominator = first_meter(mid)
        meter = f"{numerator}/{denominator}"
        bar_ticks = int(round(mid.ticks_per_beat * numerator * 4.0 / denominator))
        if bar_ticks <= 0:
            continue
        window_ticks = bar_ticks * 2
        max_tick = max_track_tick(mid)
        if max_tick <= window_ticks:
            continue
        for track_index, track in enumerate(mid.tracks):
            name = track_name(track, f"track_{track_index}")
            upper = name.upper()
            if not any(k in upper for k in ("MELODY", "BRIDGE", "PIANO")):
                continue
            notes = notes_from_track(mid, track_index)
            if len(notes) < 3:
                continue
            used_windows = set()
            starts_from_notes = sorted({(n.start_tick // bar_ticks) * bar_ticks for n in notes})
            for start_tick in starts_from_notes[:96]:
                if start_tick in used_windows:
                    continue
                used_windows.add(start_tick)
                end_tick = start_tick + window_ticks
                if end_tick > max_tick:
                    continue
                start_sec = tick_to_sec(start_tick)
                end_sec = tick_to_sec(end_tick)
                duration_sec = end_sec - start_sec
                if not (1.0 <= duration_sec <= 8.0):
                    continue
                seg_notes = relative_notes(notes, start_tick, end_tick, tick_to_sec)
                features = compute_features(seg_notes, duration_sec)
                if features is None:
                    continue
                if int(features["note_count"]) < 3:
                    continue
                scores = score_candidate(features, name, "POP909")
                source_group = f"song_{song_dir.name}"
                candidates.append(
                    Candidate(
                        source_dataset="POP909",
                        source_group=source_group,
                        source_file=midi_path,
                        track_index=track_index,
                        track_name=name,
                        start_tick=start_tick,
                        end_tick=end_tick,
                        start_sec=start_sec,
                        end_sec=end_sec,
                        bpm=bpm,
                        meter=meter,
                        bars=2.0,
                        ticks_per_beat=mid.ticks_per_beat,
                        tempo=tempo,
                        numerator=numerator,
                        denominator=denominator,
                        notes=seg_notes,
                        features=features,
                        score_by_category=scores,
                        sub_category="two_bar_pop909_" + slugify(name.lower()),
                        source_song=song_dir.name,
                    )
                )
    return candidates


def candidate_roots_by_keyword(keywords: Sequence[str]) -> List[Path]:
    roots = []
    search_roots = [AB_DIR, MFP_DIR / "downloads", MFP_DIR]
    seen = set()
    for root in search_roots:
        if not root.exists():
            continue
        try:
            for child in root.iterdir():
                lower = child.name.lower()
                if any(k.lower() in lower for k in keywords):
                    resolved = child.resolve()
                    if resolved not in seen:
                        roots.append(child)
                        seen.add(resolved)
        except OSError:
            continue
    return roots


def build_generic_candidates(dataset: str, roots: Sequence[Path], limit_files: int = 500) -> List[Candidate]:
    candidates: List[Candidate] = []
    midi_files: List[Path] = []
    for root in roots:
        if root.is_file() and root.suffix.lower() in (".mid", ".midi"):
            midi_files.append(root)
        elif root.is_dir():
            for path in root.rglob("*"):
                if path.is_file() and path.suffix.lower() in (".mid", ".midi"):
                    midi_files.append(path)
                    if len(midi_files) >= limit_files:
                        break
        if len(midi_files) >= limit_files:
            break
    for midi_path in sorted(midi_files)[:limit_files]:
        mid = load_midi(midi_path)
        if mid is None:
            continue
        tick_to_sec = tick_to_seconds_func(mid)
        tempo = first_tempo(mid)
        bpm = 60000000.0 / tempo if tempo else 120.0
        numerator, denominator = first_meter(mid)
        meter = f"{numerator}/{denominator}"
        bar_ticks = int(round(mid.ticks_per_beat * numerator * 4.0 / denominator))
        max_tick = max_track_tick(mid)
        for track_index, track in enumerate(mid.tracks):
            name = track_name(track, f"track_{track_index}")
            upper = name.upper()
            if dataset == "Lakh_MIDI_Clean_or_Slakh2100":
                if not any(k in upper for k in ("PIANO", "LEAD", "GUITAR", "MELODY")):
                    continue
            notes = notes_from_track(mid, track_index)
            if len(notes) < 3:
                continue
            starts = [n.start_tick for n in notes[:: max(1, len(notes) // 16)]]
            for start_tick in starts[:16]:
                start_sec = tick_to_sec(start_tick)
                end_tick = min(max_tick, start_tick + max(1, int(mid.ticks_per_beat * 6)))
                end_sec = tick_to_sec(end_tick)
                duration_sec = end_sec - start_sec
                if duration_sec > 6.0:
                    end_sec = start_sec + 6.0
                    # Approximate enough for optional local generic sources.
                    end_tick = start_tick + int(mido.second2tick(6.0, mid.ticks_per_beat, tempo))
                    duration_sec = 6.0
                if not (1.0 <= duration_sec <= 8.0):
                    continue
                seg_notes = relative_notes(notes, start_tick, end_tick, tick_to_sec)
                features = compute_features(seg_notes, duration_sec)
                if features is None or int(features["note_count"]) < 3:
                    continue
                scores = score_candidate(features, name, dataset)
                bars = (end_tick - start_tick) / bar_ticks if bar_ticks > 0 else 0.0
                candidates.append(
                    Candidate(
                        source_dataset=dataset,
                        source_group=midi_path.stem,
                        source_file=midi_path,
                        track_index=track_index,
                        track_name=name,
                        start_tick=start_tick,
                        end_tick=end_tick,
                        start_sec=start_sec,
                        end_sec=end_sec,
                        bpm=bpm,
                        meter=meter,
                        bars=bars,
                        ticks_per_beat=mid.ticks_per_beat,
                        tempo=tempo,
                        numerator=numerator,
                        denominator=denominator,
                        notes=seg_notes,
                        features=features,
                        score_by_category=scores,
                        sub_category="short_public_midi_" + slugify(name.lower()),
                        source_song=midi_path.stem,
                    )
                )
    return candidates


def detect_public_sources() -> Tuple[Dict[str, List[Path]], Dict[str, str]]:
    found: Dict[str, List[Path]] = {}
    missing: Dict[str, str] = {}
    pop_root = AB_DIR / "pop909_extracted" / "POP909-Dataset-master" / "POP909"
    if pop_root.exists():
        found["POP909"] = [pop_root]
    else:
        missing["POP909"] = f"Missing expected local path: {pop_root}"

    source_keywords = {
        "MAESTRO_v3_midi_only": ["maestro"],
        "ASAP": ["asap"],
        "Lakh_MIDI_Clean_or_Slakh2100": ["lakh", "lmd", "slakh"],
        "GiantMIDI_or_MAESTRO_complex": ["giant", "maestro"],
    }
    for dataset, keywords in source_keywords.items():
        roots = candidate_roots_by_keyword(keywords)
        # Avoid counting POP909 or generated experiment outputs as alternate public datasets.
        roots = [
            p
            for p in roots
            if "pop909" not in p.name.lower()
            and "calls_100_public_final" not in str(p).lower()
            and "calls_50" not in str(p).lower()
            and "objective_search" not in str(p).lower()
            and "aria_" not in str(p).lower()
        ]
        if roots:
            found[dataset] = roots
        else:
            missing[dataset] = "No local directory or MIDI file matching " + ", ".join(keywords)
    return found, missing


def build_public_candidates(found_sources: Dict[str, List[Path]]) -> List[Candidate]:
    candidates: List[Candidate] = []
    if "POP909" in found_sources:
        candidates.extend(build_pop909_candidates(found_sources["POP909"][0]))
    for dataset in (
        "MAESTRO_v3_midi_only",
        "ASAP",
        "Lakh_MIDI_Clean_or_Slakh2100",
        "GiantMIDI_or_MAESTRO_complex",
    ):
        roots = found_sources.get(dataset, [])
        if roots:
            candidates.extend(build_generic_candidates(dataset, roots))
    return candidates


def source_targets_for_available(candidates: Sequence[Candidate]) -> Dict[str, int]:
    available = {c.source_dataset for c in candidates}
    targets = {dataset: expected for dataset, expected in EXPECTED_PUBLIC_SOURCES.items() if dataset in available}
    deficit = 50 - sum(targets.values())
    if deficit > 0 and available:
        candidate_counts = Counter(c.source_dataset for c in candidates)
        for dataset, _ in candidate_counts.most_common():
            add = min(deficit, max(0, candidate_counts[dataset] - targets.get(dataset, 0)))
            targets[dataset] = targets.get(dataset, 0) + add
            deficit -= add
            if deficit <= 0:
                break
    return targets


def choose_candidates(candidates: Sequence[Candidate], old_fingerprints: set) -> Tuple[List[Tuple[str, Candidate]], List[str]]:
    selected: List[Tuple[str, Candidate]] = []
    notes: List[str] = []
    used_keys = set()
    used_source_files = set()
    used_fingerprints = set(old_fingerprints)
    source_targets = source_targets_for_available(candidates)
    source_counts: Counter = Counter()
    if sum(source_targets.values()) < 50:
        notes.append("Available public MIDI sources did not expose enough candidates to fill all 50 additions.")

    for category, quota in CATEGORY_QUOTAS.items():
        category_picks = 0
        while category_picks < quota:
            choices: List[Candidate] = []
            for cand in candidates:
                if cand.key in used_keys:
                    continue
                if source_counts[cand.source_dataset] >= source_targets.get(cand.source_dataset, 0):
                    continue
                choices.append(cand)
            if not choices:
                notes.append(f"Relaxed source target while filling category {category}.")
                choices = [cand for cand in candidates if cand.key not in used_keys]
            if not choices:
                raise RuntimeError(f"Could not fill required category {category}; no candidates left.")
            choices.sort(
                key=lambda cand: (
                    cand.score_by_category.get(category, 0.0),
                    str(cand.source_file) not in used_source_files,
                    -source_counts[cand.source_dataset],
                    -cand.start_tick,
                ),
                reverse=True,
            )
            picked = None
            for cand in choices:
                midi_bytes = candidate_to_midi_bytes(cand)
                fp = sha1_bytes(midi_bytes)
                if fp in used_fingerprints:
                    used_keys.add(cand.key)
                    continue
                cand.midi_bytes = midi_bytes
                cand.fingerprint_sha1 = fp
                picked = cand
                break
            if picked is None:
                raise RuntimeError(f"Could not fill {category}; candidate fingerprints were exhausted.")
            selected.append((category, picked))
            used_keys.add(picked.key)
            used_source_files.add(str(picked.source_file))
            used_fingerprints.add(picked.fingerprint_sha1)
            source_counts[picked.source_dataset] += 1
            category_picks += 1

    if len(selected) != 50:
        raise RuntimeError(f"Expected 50 public additions, selected {len(selected)}.")
    return selected, notes


def clean_output() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if CALLS_DIR.exists():
        shutil.rmtree(CALLS_DIR)
    CALLS_DIR.mkdir(parents=True, exist_ok=True)


def old_call_rows() -> Tuple[List[Dict[str, str]], set]:
    rows: List[Dict[str, str]] = []
    fingerprints = set()
    with CALL50_MANIFEST.open("r", encoding="utf-8-sig", newline="") as f:
        for source_row in csv.DictReader(f):
            src = Path(source_row.get("midi_path", "")).expanduser()
            if not src.exists():
                src = CALL50_DIR / Path(source_row.get("midi_path", "")).name
            if not src.exists():
                raise FileNotFoundError(f"Missing call50 MIDI: {source_row.get('midi_path')}")
            dst = CALLS_DIR / src.name
            shutil.copy2(src, dst)
            mid = load_midi(dst)
            if mid is None:
                raise RuntimeError(f"Copied call50 MIDI is unreadable: {dst}")
            notes = notes_from_midi(mid)
            duration = estimate_duration(mid, notes)
            features = compute_features(notes, duration)
            if features is None:
                raise RuntimeError(f"Copied call50 MIDI is empty: {dst}")
            fp = sha1_file(dst)
            if fp in fingerprints:
                raise RuntimeError(f"Duplicate fingerprint inside call50 copy: {dst}")
            fingerprints.add(fp)
            numerator, denominator = first_meter(mid)
            tempo = first_tempo(mid)
            row = {field: "" for field in FIELDS}
            row.update(
                {
                    "call_id": source_row.get("call_id", ""),
                    "origin": source_row.get("origin", "calls_50"),
                    "source_dataset": "calls_50",
                    "source_group": "existing_call50",
                    "name": source_row.get("name", dst.stem),
                    "category": source_row.get("category", "existing_call50"),
                    "sub_category": "copied_from_call50",
                    "description": source_row.get("description", "Existing Call50 call copied into Call100."),
                    "midi_path": str(dst),
                    "source_file": str(src),
                    "track_index": source_row.get("track_index", ""),
                    "track_name": source_row.get("track_name", ""),
                    "start_sec": "0",
                    "end_sec": fmt(duration),
                    "start_tick": "0",
                    "end_tick": fmt(max_track_tick(mid), 0),
                    "bpm": fmt(60000000.0 / tempo if tempo else 120.0),
                    "meter": f"{numerator}/{denominator}",
                    "bars": fmt(estimate_bars(mid)),
                    "license_note": SOURCE_LICENSES["calls_50"],
                    "citation_key": CITATIONS["calls_50"],
                    "split": "base_call50",
                    "fingerprint_sha1": fp,
                }
            )
            row.update(feature_fields(features))
            rows.append(row)
    if len(rows) != 50:
        raise RuntimeError(f"Expected 50 rows in calls_50 manifest, got {len(rows)}.")
    return rows, fingerprints


def public_call_rows(selected: Sequence[Tuple[str, Candidate]]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for idx, (category, cand) in enumerate(selected, start=1):
        call_id = f"P{idx:03d}"
        filename = f"{call_id}_{slugify(cand.source_dataset.lower())}_{slugify(category)}_{slugify(cand.source_song)}_{slugify(cand.track_name.lower())}.mid"
        dst = CALLS_DIR / filename
        dst.write_bytes(cand.midi_bytes)
        features = cand.features
        row = {field: "" for field in FIELDS}
        row.update(
            {
                "call_id": call_id,
                "origin": "public_midi_extract",
                "source_dataset": cand.source_dataset,
                "source_group": cand.source_group,
                "name": f"{cand.source_dataset} {cand.source_song} {cand.track_name} {fmt(cand.start_sec, 3)}s",
                "category": category,
                "sub_category": cand.sub_category,
                "description": (
                    f"{cand.source_dataset} {cand.track_name} segment from {cand.source_group}; "
                    f"selected for {category}."
                ),
                "midi_path": str(dst),
                "source_file": str(cand.source_file),
                "track_index": fmt(cand.track_index, 0),
                "track_name": cand.track_name,
                "start_sec": fmt(cand.start_sec),
                "end_sec": fmt(cand.end_sec),
                "start_tick": fmt(cand.start_tick, 0),
                "end_tick": fmt(cand.end_tick, 0),
                "bpm": fmt(cand.bpm),
                "meter": cand.meter,
                "bars": fmt(cand.bars),
                "license_note": SOURCE_LICENSES.get(cand.source_dataset, "See source dataset terms."),
                "citation_key": CITATIONS.get(cand.source_dataset, cand.source_dataset),
                "split": "public_addition",
                "fingerprint_sha1": cand.fingerprint_sha1,
            }
        )
        row.update(feature_fields(features))
        rows.append(row)
    return rows


def write_manifest(rows: Sequence[Dict[str, str]]) -> Path:
    path = OUT_DIR / "call100_manifest.csv"
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FIELDS})
    return path


def numeric_values(rows: Sequence[Dict[str, str]], field: str) -> List[float]:
    values = []
    for row in rows:
        try:
            values.append(float(row.get(field, "")))
        except ValueError:
            pass
    return values


def stat_line(rows: Sequence[Dict[str, str]], field: str) -> str:
    values = numeric_values(rows, field)
    if not values:
        return f"| {field} | n/a | n/a | n/a |"
    return f"| {field} | {min(values):.4g} | {sum(values) / len(values):.4g} | {max(values):.4g} |"


def markdown_counter(counter: Counter) -> str:
    lines = ["| value | count |", "|---|---:|"]
    for key, value in sorted(counter.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"| {key} | {value} |")
    return "\n".join(lines)


def write_report(
    rows: Sequence[Dict[str, str]],
    found_sources: Dict[str, List[Path]],
    missing_sources: Dict[str, str],
    selection_notes: Sequence[str],
) -> Path:
    source_counts = Counter(row["source_dataset"] for row in rows)
    category_counts = Counter(row["category"] for row in rows)
    public_counts = Counter(row["source_dataset"] for row in rows if row.get("split") == "public_addition")
    poly_bins = Counter()
    for value in numeric_values(rows, "polyphony_rate"):
        if value < 0.01:
            poly_bins["mono_or_near_mono"] += 1
        elif value < 0.10:
            poly_bins["light_polyphony"] += 1
        elif value < 0.35:
            poly_bins["medium_polyphony"] += 1
        else:
            poly_bins["high_polyphony"] += 1
    scale_bins = Counter()
    for value in numeric_values(rows, "scale_fit"):
        if value >= 0.95:
            scale_bins[">=0.95"] += 1
        elif value >= 0.85:
            scale_bins["0.85-0.95"] += 1
        elif value >= 0.70:
            scale_bins["0.70-0.85"] += 1
        else:
            scale_bins["<0.70"] += 1

    lines = [
        "# Call100 Public Final Dataset Report",
        "",
        f"- dataset_version: `{DATASET_VERSION}`",
        f"- num_calls: `{len(rows)}`",
        f"- base copied from calls_50: `{sum(1 for r in rows if r.get('split') == 'base_call50')}`",
        f"- public additions: `{sum(1 for r in rows if r.get('split') == 'public_addition')}`",
        "",
        "## Source Dataset Distribution",
        "",
        markdown_counter(source_counts),
        "",
        "## Public Addition Source Plan vs Actual",
        "",
        "| source_dataset | target | actual | status |",
        "|---|---:|---:|---|",
    ]
    for dataset, target in EXPECTED_PUBLIC_SOURCES.items():
        actual = public_counts.get(dataset, 0)
        if dataset in missing_sources:
            status = "missing locally; skipped and backfilled from available public MIDI"
        elif actual == target:
            status = "met"
        else:
            status = "available but target not exact"
        lines.append(f"| {dataset} | {target} | {actual} | {status} |")
    lines.extend(
        [
            "",
            "## Category Distribution",
            "",
            markdown_counter(category_counts),
            "",
            "## Required Public Category Quotas",
            "",
            "| category | quota | actual_public_additions |",
            "|---|---:|---:|",
        ]
    )
    public_category_counts = Counter(row["category"] for row in rows if row.get("split") == "public_addition")
    for category, quota in CATEGORY_QUOTAS.items():
        lines.append(f"| {category} | {quota} | {public_category_counts.get(category, 0)} |")
    lines.extend(
        [
            "",
            "## Feature Summary",
            "",
            "| feature | min | mean | max |",
            "|---|---:|---:|---:|",
            stat_line(rows, "note_count"),
            stat_line(rows, "duration_sec"),
            stat_line(rows, "pitch_range"),
            stat_line(rows, "onset_density"),
            "",
            "## Polyphony Rate Distribution",
            "",
            markdown_counter(poly_bins),
            "",
            "## Scale Fit Distribution",
            "",
            markdown_counter(scale_bins),
            "",
            "## Missing Sources",
            "",
        ]
    )
    if missing_sources:
        for dataset, reason in missing_sources.items():
            lines.append(f"- `{dataset}`: {reason}")
    else:
        lines.append("- None.")
    lines.extend(["", "## Selection Notes", ""])
    if selection_notes:
        for note in selection_notes:
            lines.append(f"- {note}")
    else:
        lines.append("- Public additions selected without source-target relaxation.")
    lines.extend(
        [
            "",
            "## License And Citation Notes",
            "",
            "- `calls_50`: copied from the existing local benchmark directory; no files in calls_50 were modified.",
            "- `POP909`: local repository includes an MIT LICENSE file and README citation for `pop909-ismir2020`.",
            "- Other requested public datasets were not imported in this run unless listed with nonzero actual counts above.",
            "- Manifest rows carry `license_note` and `citation_key` per call.",
            "",
            "## Found Local Source Roots",
            "",
        ]
    )
    for dataset, roots in found_sources.items():
        for root in roots:
            lines.append(f"- `{dataset}`: `{root}`")

    path = OUT_DIR / "call100_dataset_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_fingerprint(rows: Sequence[Dict[str, str]], manifest_path: Path, found_sources: Dict[str, List[Path]], missing_sources: Dict[str, str]) -> Path:
    source_counts = Counter(row["source_dataset"] for row in rows)
    public_counts = Counter(row["source_dataset"] for row in rows if row.get("split") == "public_addition")
    payload = {
        "dataset_version": DATASET_VERSION,
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "num_calls": len(rows),
        "manifest_sha1": sha1_file(manifest_path),
        "source_datasets": {
            dataset: {
                "total_count": source_counts.get(dataset, 0),
                "public_addition_count": public_counts.get(dataset, 0),
                "expected_public_additions": EXPECTED_PUBLIC_SOURCES.get(dataset),
                "found_roots": [str(p) for p in found_sources.get(dataset, [])],
                "missing_reason": missing_sources.get(dataset, ""),
            }
            for dataset in sorted(set(source_counts) | set(EXPECTED_PUBLIC_SOURCES))
        },
    }
    path = OUT_DIR / "dataset_fingerprint.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def main() -> None:
    if not CALL50_MANIFEST.exists():
        raise FileNotFoundError(f"Missing calls_50 manifest: {CALL50_MANIFEST}")
    clean_output()
    base_rows, old_fingerprints = old_call_rows()
    found_sources, missing_sources = detect_public_sources()
    candidates = build_public_candidates(found_sources)
    if not candidates:
        raise RuntimeError("No valid public MIDI candidates were found; cannot build 100-call dataset.")
    selected, selection_notes = choose_candidates(candidates, old_fingerprints)
    if missing_sources:
        available_names = ", ".join(sorted({cand.source_dataset for cand in candidates}))
        missing_names = ", ".join(sorted(missing_sources))
        selection_notes.insert(
            0,
            f"Skipped missing requested sources ({missing_names}) and redistributed public additions to available source(s): {available_names}.",
        )
    public_rows = public_call_rows(selected)
    rows = base_rows + public_rows
    if len(rows) != 100:
        raise RuntimeError(f"Expected 100 manifest rows, got {len(rows)}.")
    manifest_path = write_manifest(rows)
    report_path = write_report(rows, found_sources, missing_sources, selection_notes)
    fingerprint_path = write_fingerprint(rows, manifest_path, found_sources, missing_sources)
    print(f"built_manifest={manifest_path}")
    print(f"built_calls={CALLS_DIR}")
    print(f"built_report={report_path}")
    print(f"built_fingerprint={fingerprint_path}")
    print(f"public_candidates_scanned={len(candidates)}")


if __name__ == "__main__":
    main()
