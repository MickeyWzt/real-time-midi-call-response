"""
Live MIDI Call-and-Response prototype.

Pipeline:
  VMPK/loopMIDI -> Python_IN -> MIDI VAD -> AMT Transformer -> Python_OUT

This is the realtime-oriented version of the paper prototype. It keeps the
model loaded, records a human Call phrase from live MIDI, cuts the phrase with
the nonhomogeneous-Poisson VAD, generates a Response with a pretrained
Anticipatory Music Transformer, micro-buffers the first events, and sends
note_on/note_off messages to a virtual MIDI output.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import queue
import sys
import threading
import time
import tempfile
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Tuple

from drum_loop import DrumEvent, is_drum_pad_note
from loop_bank import LoopEvent, ResponseLoopBank
from rhythm_guide import RhythmGuideEngine


ROOT = Path(__file__).resolve().parents[1]
PROJECT_DEPS = ROOT / ".python_deps"
ANTICIPATION_REPO = ROOT / "code" / "anticipation"
HF_CACHE = ROOT / "hf_cache"
HF_HUB_CACHE = HF_CACHE / "hub"
HF_MODULES_CACHE = ROOT / "hf_modules_cache"

if PROJECT_DEPS.exists():
    sys.path.insert(0, str(PROJECT_DEPS))
sys.path.insert(0, str(ANTICIPATION_REPO))

os.environ.setdefault("HF_HOME", str(HF_CACHE))
os.environ.setdefault("HF_HUB_CACHE", str(HF_HUB_CACHE))
os.environ.setdefault("HF_MODULES_CACHE", str(HF_MODULES_CACHE))
os.environ.setdefault("HF_ENDPOINT", "https://huggingface.co")

from midi_vad_endpoint import EndpointCancel, EndpointCandidate, EndpointDecision, MidiEndpointVAD


DEFAULT_MODEL_ID = "stanford-crfm/music-small-800k"
DEFAULT_ARIA_MODEL_ID = str(ROOT / "model_weights" / "aria-medium-gen")
DEFAULT_INPUT_PORT = "Python_IN 0"
DEFAULT_OUTPUT_PORT = "Python_OUT"
DEFAULT_MELODY_OUTPUT_PORT = "Logic Pro Virtual In"
TIME_RESOLUTION = 100
LATENCY_PRESETS = {
    # Fast live mode starts after one streamed event and keeps filling the
    # playback queue while the response is heard. A two-second guard buffer
    # made the turn-based interface feel unresponsive.
    "fast": (1, 0.35),
    "balanced": (2, 0.80),
    "classic": (3, 2.00),
    "smooth": (4, 2.00),
}


def should_capture_rhythm_pad(
    message,
    learning: bool,
    drum_pad_min: int,
    drum_pad_max: int,
    drum_channel: int,
) -> bool:
    """Reserve only the MiniLab pad lane for rhythm learning.

    Keyboard notes must never disappear just because a groove is being
    learned.  The web interface sends taps through the control channel, while
    hardware rhythm capture uses the dedicated channel-10 pad lane.
    """

    return learning and is_drum_pad_note(
        message,
        drum_pad_min,
        drum_pad_max,
        drum_channel,
    )


def default_metrics_path() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return ROOT / "logs" / f"live_metrics_{stamp}.csv"


@dataclass
class CapturedNote:
    onset: float
    pitch: int
    velocity: int
    channel: int = 0
    duration: Optional[float] = None


@dataclass(frozen=True)
class GeneratedEvent:
    tick: int
    duration_ticks: int
    pitch: int
    instrument: int
    velocity: int = 92


@dataclass(frozen=True)
class CallProfile:
    duration_seconds: float
    note_count: int
    pitch_min: int
    pitch_max: int
    mean_pitch: float
    dominant_pitch: int
    dominant_pitch_share: float
    contour: str
    note_density: float
    tail_repeat_pitch: Optional[int]
    tail_repeat_count: int
    tail_repeat_share: float
    tail_start_index: Optional[int]
    scale_root: int
    scale_mode: str
    scale_pitch_classes: Tuple[int, ...]


@dataclass(frozen=True)
class ResponsePlan:
    target_seconds: float
    target_notes: int
    pitch_min: int
    pitch_max: int
    same_pitch_limit: int
    dominant_pitch_max_share: float
    resample_attempts: int
    duration_match_min_share: float = 0.80
    stop_on_target_notes: bool = False


@dataclass
class MusicalControlStats:
    rejected_repeat_count: int = 0
    rejected_dominant_count: int = 0
    fallback_count: int = 0


@dataclass
class SpeculativePreloadState:
    preload_id: int
    candidate_time: float
    phrase_note_count: int
    started_at: float
    stop_event: threading.Event
    done_event: threading.Event
    events: List[GeneratedEvent]
    stats: MusicalControlStats
    first_event_time: Optional[float] = None
    ended_at: Optional[float] = None
    error: str = ""
    cancelled: bool = False


@dataclass
class PlaybackStats:
    playback_start_time: Optional[float]
    playback_end_time: Optional[float]
    initial_buffer_wait: float
    initial_buffer_count: int
    played_events: int
    buffer_underruns: int


@dataclass
class RoundMetrics:
    round_id: int
    call_notes: int
    model_id: str
    device: str
    call_start_time: Optional[float] = None
    endpoint_time: Optional[float] = None
    generation_start_time: Optional[float] = None
    first_event_time: Optional[float] = None
    playback_start_time: Optional[float] = None
    playback_end_time: Optional[float] = None
    generated_events: int = 0
    played_events: int = 0
    buffer_underruns: int = 0
    initial_buffer_wait: Optional[float] = None
    initial_buffer_count: int = 0
    endpoint_silence: Optional[float] = None
    endpoint_cutoff: Optional[float] = None
    mu_tempo: Optional[float] = None
    call_duration_seconds: Optional[float] = None
    dominant_pitch: Optional[int] = None
    dominant_pitch_share: Optional[float] = None
    tail_repeat_pitch: Optional[int] = None
    tail_repeat_count: int = 0
    target_response_seconds: Optional[float] = None
    target_response_notes: Optional[int] = None
    raw_response_seconds: Optional[float] = None
    actual_response_seconds: Optional[float] = None
    duration_match_ratio: Optional[float] = None
    duration_stretch_factor: Optional[float] = None
    rejected_repeat_count: int = 0
    rejected_dominant_count: int = 0
    fallback_count: int = 0
    candidate_time: Optional[float] = None
    candidate_confirm_delay: Optional[float] = None
    speculative_preload_used: bool = False
    speculative_cancel_count: int = 0
    preload_latency: Optional[float] = None
    status: str = "ok"
    error: str = ""


class MetricsRecorder:
    FIELDNAMES = [
        "round_id",
        "status",
        "model_id",
        "device",
        "call_notes",
        "generated_events",
        "played_events",
        "buffer_underruns",
        "call_start_time",
        "endpoint_time",
        "generation_start_time",
        "first_event_time",
        "playback_start_time",
        "playback_end_time",
        "endpoint_wait_s",
        "first_event_latency_s",
        "buffer_wait_s",
        "total_response_latency_s",
        "playback_duration_s",
        "endpoint_silence_s",
        "endpoint_cutoff_s",
        "mu_tempo_per_s",
        "call_duration_s",
        "dominant_pitch",
        "dominant_pitch_share",
        "tail_repeat_pitch",
        "tail_repeat_count",
        "target_response_seconds",
        "target_response_notes",
        "raw_response_seconds",
        "actual_response_seconds",
        "duration_match_ratio",
        "duration_stretch_factor",
        "rejected_repeat_count",
        "rejected_dominant_count",
        "fallback_count",
        "candidate_time",
        "candidate_confirm_delay_s",
        "speculative_preload_used",
        "speculative_cancel_count",
        "preload_latency_s",
        "error",
    ]

    def __init__(self, path: Optional[Path], enabled: bool) -> None:
        self.path = path
        self.enabled = enabled
        self.rows: List[Dict[str, object]] = []
        self._mono_epoch = time.monotonic()
        self._wall_epoch = time.time()
        if self.enabled and self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=self.FIELDNAMES)
                writer.writeheader()
            print(f"[metrics] writing CSV to {self.path}")
        else:
            print("[metrics] disabled")

    def _iso_time(self, value: Optional[float]) -> str:
        if value is None:
            return ""
        wall_time = self._wall_epoch + (value - self._mono_epoch)
        return datetime.fromtimestamp(wall_time).isoformat(timespec="milliseconds")

    @staticmethod
    def _duration(start: Optional[float], end: Optional[float]) -> str:
        if start is None or end is None:
            return ""
        return f"{max(0.0, end - start):.6f}"

    @staticmethod
    def _float(value: Optional[float]) -> str:
        if value is None:
            return ""
        return f"{value:.6f}"

    def record(self, metrics: RoundMetrics) -> None:
        row: Dict[str, object] = {
            "round_id": metrics.round_id,
            "status": metrics.status,
            "model_id": metrics.model_id,
            "device": metrics.device,
            "call_notes": metrics.call_notes,
            "generated_events": metrics.generated_events,
            "played_events": metrics.played_events,
            "buffer_underruns": metrics.buffer_underruns,
            "call_start_time": self._iso_time(metrics.call_start_time),
            "endpoint_time": self._iso_time(metrics.endpoint_time),
            "generation_start_time": self._iso_time(metrics.generation_start_time),
            "first_event_time": self._iso_time(metrics.first_event_time),
            "playback_start_time": self._iso_time(metrics.playback_start_time),
            "playback_end_time": self._iso_time(metrics.playback_end_time),
            "endpoint_wait_s": self._duration(metrics.call_start_time, metrics.endpoint_time),
            "first_event_latency_s": self._duration(metrics.generation_start_time, metrics.first_event_time),
            "buffer_wait_s": self._float(metrics.initial_buffer_wait),
            "total_response_latency_s": self._duration(metrics.endpoint_time, metrics.playback_start_time),
            "playback_duration_s": self._duration(metrics.playback_start_time, metrics.playback_end_time),
            "endpoint_silence_s": self._float(metrics.endpoint_silence),
            "endpoint_cutoff_s": self._float(metrics.endpoint_cutoff),
            "mu_tempo_per_s": self._float(metrics.mu_tempo),
            "call_duration_s": self._float(metrics.call_duration_seconds),
            "dominant_pitch": "" if metrics.dominant_pitch is None else metrics.dominant_pitch,
            "dominant_pitch_share": self._float(metrics.dominant_pitch_share),
            "tail_repeat_pitch": "" if metrics.tail_repeat_pitch is None else metrics.tail_repeat_pitch,
            "tail_repeat_count": metrics.tail_repeat_count,
            "target_response_seconds": self._float(metrics.target_response_seconds),
            "target_response_notes": "" if metrics.target_response_notes is None else metrics.target_response_notes,
            "raw_response_seconds": self._float(metrics.raw_response_seconds),
            "actual_response_seconds": self._float(metrics.actual_response_seconds),
            "duration_match_ratio": self._float(metrics.duration_match_ratio),
            "duration_stretch_factor": self._float(metrics.duration_stretch_factor),
            "rejected_repeat_count": metrics.rejected_repeat_count,
            "rejected_dominant_count": metrics.rejected_dominant_count,
            "fallback_count": metrics.fallback_count,
            "candidate_time": self._iso_time(metrics.candidate_time),
            "candidate_confirm_delay_s": self._float(metrics.candidate_confirm_delay),
            "speculative_preload_used": int(metrics.speculative_preload_used),
            "speculative_cancel_count": metrics.speculative_cancel_count,
            "preload_latency_s": self._float(metrics.preload_latency),
            "error": metrics.error,
        }
        self.rows.append(row)
        if self.enabled and self.path is not None:
            with self.path.open("a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=self.FIELDNAMES)
                writer.writerow(row)
        print(
            "[metrics] round={round_id} status={status} first_event={first}s "
            "total_response={total}s underruns={underruns}".format(
                round_id=metrics.round_id,
                status=metrics.status,
                first=row["first_event_latency_s"] or "n/a",
                total=row["total_response_latency_s"] or "n/a",
                underruns=metrics.buffer_underruns,
            )
        )

    def print_summary(self) -> None:
        if not self.rows:
            print("[summary] no completed response rounds")
            return

        def numeric(field: str) -> List[float]:
            values: List[float] = []
            for row in self.rows:
                value = row.get(field)
                if value not in ("", None):
                    values.append(float(value))
            return values

        first_event = numeric("first_event_latency_s")
        total_response = numeric("total_response_latency_s")
        endpoint_wait = numeric("endpoint_wait_s")
        duration_match = numeric("duration_match_ratio")
        total_underruns = sum(int(row["buffer_underruns"]) for row in self.rows)

        print("[summary] rounds={}".format(len(self.rows)))
        if endpoint_wait:
            print(f"[summary] avg_endpoint_wait={mean(endpoint_wait):.3f}s")
        if first_event:
            print(f"[summary] avg_first_event_latency={mean(first_event):.3f}s")
        if total_response:
            print(
                f"[summary] avg_total_response_latency={mean(total_response):.3f}s "
                f"max_total_response_latency={max(total_response):.3f}s"
            )
        if duration_match:
            print(f"[summary] avg_duration_match_ratio={mean(duration_match):.3f}")
        print(f"[summary] total_buffer_underruns={total_underruns}")
        if self.enabled and self.path is not None:
            print(f"[summary] metrics_csv={self.path}")


class PhraseRecorder:
    """Records live Note-On/Note-Off pairs for one human Call phrase."""

    def __init__(self, default_duration: float = 0.25) -> None:
        self.default_duration = default_duration
        self._active: Dict[Tuple[int, int], CapturedNote] = {}
        self._completed: List[CapturedNote] = []
        self._lock = threading.Lock()

    def note_on(self, pitch: int, velocity: int, channel: int, timestamp: float) -> None:
        with self._lock:
            key = (channel, pitch)
            self._active[key] = CapturedNote(
                onset=timestamp,
                pitch=pitch,
                velocity=velocity,
                channel=channel,
            )

    def note_off(self, pitch: int, channel: int, timestamp: float) -> None:
        with self._lock:
            key = (channel, pitch)
            note = self._active.pop(key, None)
            if note is None:
                return
            note.duration = max(timestamp - note.onset, self.default_duration)
            self._completed.append(note)

    def snapshot_phrase(self, cut_time: float) -> List[CapturedNote]:
        with self._lock:
            phrase = [
                CapturedNote(
                    onset=note.onset,
                    pitch=note.pitch,
                    velocity=note.velocity,
                    channel=note.channel,
                    duration=note.duration or self.default_duration,
                )
                for note in self._completed
            ]
            for note in self._active.values():
                elapsed = max(cut_time - note.onset, 0.0)
                phrase.append(
                    CapturedNote(
                        onset=note.onset,
                        pitch=note.pitch,
                        velocity=note.velocity,
                        channel=note.channel,
                        duration=max(elapsed, self.default_duration),
                    )
                )

        phrase.sort(key=lambda item: item.onset)
        if phrase:
            t0 = phrase[0].onset
            phrase = [
                CapturedNote(
                    onset=note.onset - t0,
                    pitch=note.pitch,
                    velocity=note.velocity,
                    channel=note.channel,
                    duration=note.duration or self.default_duration,
                )
                for note in phrase
            ]
        return phrase

    def consume_phrase(self, cut_time: float) -> List[CapturedNote]:
        with self._lock:
            phrase = list(self._completed)
            for note in self._active.values():
                elapsed = max(cut_time - note.onset, 0.0)
                note.duration = max(elapsed, self.default_duration)
                phrase.append(note)
            self._completed.clear()
            self._active.clear()

        phrase.sort(key=lambda item: item.onset)
        if phrase:
            t0 = phrase[0].onset
            phrase = [
                CapturedNote(
                    onset=note.onset - t0,
                    pitch=note.pitch,
                    velocity=note.velocity,
                    channel=note.channel,
                    duration=note.duration or self.default_duration,
                )
                for note in phrase
            ]
        return phrase


def clamp_float(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)


def clamp_int(value: int, lower: int, upper: int) -> int:
    return min(max(value, lower), upper)


def quantize_onset_to_rhythm_grid(
    onset_offset: float,
    bpm: float,
    strength: float = 0.92,
) -> float:
    """Pull a response onset toward the nearest learned 1/16-note grid point."""

    if bpm <= 0:
        return max(0.0, onset_offset)
    grid = (60.0 / bpm) / 4.0
    quantized = round(max(0.0, onset_offset) / grid) * grid
    amount = clamp_float(strength, 0.0, 1.0)
    return max(0.0, onset_offset + (quantized - onset_offset) * amount)


def should_use_motif_fallback(
    *,
    generated_count: int,
    fallback_on_empty: bool,
    backend: str,
    response_strategy: str,
    stopped: bool,
) -> bool:
    """Allow a full motif rescue only when an AMT response is empty."""

    return (
        generated_count == 0
        and fallback_on_empty
        and backend == "amt"
        and response_strategy in {"controlled_amt", "streaming_amt"}
        and not stopped
    )


def analyze_call_phrase(notes: List[CapturedNote]) -> CallProfile:
    if not notes:
        return CallProfile(
            duration_seconds=0.0,
            note_count=0,
            pitch_min=60,
            pitch_max=60,
            mean_pitch=60.0,
            dominant_pitch=60,
            dominant_pitch_share=0.0,
            contour="flat",
            note_density=0.0,
            tail_repeat_pitch=None,
            tail_repeat_count=0,
            tail_repeat_share=0.0,
            tail_start_index=None,
            scale_root=0,
            scale_mode="major",
            scale_pitch_classes=(0, 2, 4, 5, 7, 9, 11),
        )

    pitches = [note.pitch for note in notes]
    duration_seconds = max(
        note.onset + max(note.duration or 0.0, 0.0)
        for note in notes
    )
    duration_seconds = max(duration_seconds, 0.001)
    counts = Counter(pitches)
    dominant_pitch, dominant_count = counts.most_common(1)[0]
    dominant_share = dominant_count / len(notes)
    mean_pitch = sum(pitches) / len(pitches)

    if len(pitches) < 2:
        contour = "flat"
    else:
        delta = pitches[-1] - pitches[0]
        if delta >= 3:
            contour = "ascending"
        elif delta <= -3:
            contour = "descending"
        elif max(pitches) - min(pitches) >= 5:
            contour = "arched"
        else:
            contour = "flat"

    tail_pitch = pitches[-1]
    consecutive_tail = 1
    for pitch in reversed(pitches[:-1]):
        if pitch != tail_pitch:
            break
        consecutive_tail += 1

    tail_repeat_pitch: Optional[int] = None
    tail_repeat_count = 0
    tail_repeat_share = 0.0
    tail_start_index: Optional[int] = None

    if consecutive_tail >= 4:
        tail_repeat_pitch = tail_pitch
        tail_repeat_count = consecutive_tail
        tail_repeat_share = consecutive_tail / len(notes)
        tail_start_index = len(notes) - consecutive_tail
    else:
        window_start = max(0.0, duration_seconds - 1.5)
        window_items = [
            (idx, note.pitch)
            for idx, note in enumerate(notes)
            if note.onset >= window_start
        ]
        if window_items:
            window_counts = Counter(pitch for _, pitch in window_items)
            window_pitch, window_count = window_counts.most_common(1)[0]
            window_share = window_count / len(window_items)
            if window_count >= 4 and window_share >= 0.70:
                tail_repeat_pitch = window_pitch
                tail_repeat_count = window_count
                tail_repeat_share = window_share
                tail_start_index = next(
                    idx for idx, pitch in window_items if pitch == window_pitch
                )

    scale_root, scale_mode, scale_pitch_classes = infer_phrase_scale(notes)
    return CallProfile(
        duration_seconds=duration_seconds,
        note_count=len(notes),
        pitch_min=min(pitches),
        pitch_max=max(pitches),
        mean_pitch=mean_pitch,
        dominant_pitch=dominant_pitch,
        dominant_pitch_share=dominant_share,
        contour=contour,
        note_density=len(notes) / duration_seconds,
        tail_repeat_pitch=tail_repeat_pitch,
        tail_repeat_count=tail_repeat_count,
        tail_repeat_share=tail_repeat_share,
        tail_start_index=tail_start_index,
        scale_root=scale_root,
        scale_mode=scale_mode,
        scale_pitch_classes=scale_pitch_classes,
    )


MAJOR_INTERVALS = (0, 2, 4, 5, 7, 9, 11)
MINOR_INTERVALS = (0, 2, 3, 5, 7, 8, 10)


def infer_phrase_scale(notes: List[CapturedNote]) -> Tuple[int, str, Tuple[int, ...]]:
    """Choose the major/minor collection that best explains the played phrase."""

    if not notes:
        return 0, "major", tuple(MAJOR_INTERVALS)
    pitch_weights: Counter[int] = Counter()
    for note in notes:
        duration = max(0.08, float(note.duration or 0.25))
        pitch_weights[note.pitch % 12] += 1.0 + min(duration, 1.0) * 0.45
    first_pc = notes[0].pitch % 12
    last_pc = notes[-1].pitch % 12
    best: Tuple[float, int, str, Tuple[int, ...]] | None = None
    for root in range(12):
        for mode, intervals in (("major", MAJOR_INTERVALS), ("minor", MINOR_INTERVALS)):
            pcs = tuple((root + interval) % 12 for interval in intervals)
            pentatonic_intervals = (0, 2, 4, 7, 9) if mode == "major" else (0, 3, 5, 7, 10)
            pentatonic_pcs = {(root + interval) % 12 for interval in pentatonic_intervals}
            score = sum(weight for pc, weight in pitch_weights.items() if pc in pcs)
            score -= sum(weight * 1.4 for pc, weight in pitch_weights.items() if pc not in pcs)
            # Relative major/minor and neighbouring modes can explain the same
            # seven pitch classes. Prefer the tonic whose pentatonic core best
            # explains the performed motif, and treat the opening note as a
            # stronger tonal cue than a question-like final note.
            score += sum(
                weight * 0.35
                for pc, weight in pitch_weights.items()
                if pc in pentatonic_pcs
            )
            if last_pc == root:
                score += 0.4
            elif last_pc == (root + 7) % 12:
                score += 0.2
            if first_pc == root:
                score += 1.2
            elif first_pc == (root + 7) % 12:
                score += 0.2
            candidate = (score, -root, mode, pcs)
            if best is None or candidate > best:
                best = candidate
    assert best is not None
    return -best[1], best[2], best[3]


def snap_pitch_to_scale(
    pitch: int,
    pitch_classes: Tuple[int, ...],
    lower: int,
    upper: int,
    preferred: Optional[int] = None,
) -> int:
    candidates = [value for value in range(lower, upper + 1) if value % 12 in pitch_classes]
    if not candidates:
        return clamp_int(pitch, lower, upper)
    reference = pitch if preferred is None else preferred
    return min(candidates, key=lambda value: (abs(value - pitch), abs(value - reference), value))


def estimate_phrase_grid(notes: List[CapturedNote]) -> float:
    intervals = [
        later.onset - earlier.onset
        for earlier, later in zip(notes, notes[1:])
        if 0.07 <= later.onset - earlier.onset <= 1.5
    ]
    if not intervals:
        return 0.25
    ordered = sorted(intervals)
    median = ordered[len(ordered) // 2]
    return clamp_float(median / 2.0 if median > 0.42 else median, 0.10, 0.38)


def clean_prompt_phrase(notes: List[CapturedNote], profile: CallProfile) -> List[CapturedNote]:
    if profile.tail_repeat_pitch is None or profile.tail_start_index is None:
        return notes
    if profile.tail_repeat_count <= 2:
        return notes

    cleaned: List[CapturedNote] = []
    kept_tail_matches = 0
    for idx, note in enumerate(notes):
        is_tail_repeat = (
            idx >= profile.tail_start_index
            and note.pitch == profile.tail_repeat_pitch
        )
        if is_tail_repeat:
            if kept_tail_matches >= 2:
                continue
            kept_tail_matches += 1
        cleaned.append(note)
    return cleaned or notes[: min(2, len(notes))]


def build_response_plan(
    profile: CallProfile,
    response_seconds: float,
    max_events: int,
    pitch_min: int,
    pitch_max: int,
    response_length_ratio: float,
    response_note_ratio: float,
    same_pitch_limit: int,
    dominant_pitch_max_share: float,
    resample_attempts: int,
    duration_match_min_share: float = 0.80,
    stop_on_target_notes: bool = False,
) -> ResponsePlan:
    max_seconds = max(response_seconds, 0.1)
    # A one-note or two-note Call should receive a compact answer rather than
    # being padded to two seconds before it can be played.
    min_seconds = min(0.85, max_seconds)
    target_seconds = clamp_float(
        profile.duration_seconds * response_length_ratio,
        min_seconds,
        max_seconds,
    )

    max_notes = max(max_events, 1)
    min_notes = min(4, max_notes)
    target_notes = clamp_int(
        int(round(max(profile.note_count, 1) * response_note_ratio)),
        min_notes,
        max_notes,
    )

    call_margin = 12
    planned_pitch_min = clamp_int(min(profile.pitch_min, pitch_max) - call_margin, pitch_min, pitch_max)
    planned_pitch_max = clamp_int(max(profile.pitch_max, pitch_min) + call_margin, pitch_min, pitch_max)
    if planned_pitch_min > planned_pitch_max:
        planned_pitch_min, planned_pitch_max = pitch_min, pitch_max

    return ResponsePlan(
        target_seconds=target_seconds,
        target_notes=target_notes,
        pitch_min=planned_pitch_min,
        pitch_max=planned_pitch_max,
        same_pitch_limit=same_pitch_limit,
        dominant_pitch_max_share=dominant_pitch_max_share,
        resample_attempts=resample_attempts,
        duration_match_min_share=duration_match_min_share,
        stop_on_target_notes=stop_on_target_notes,
    )


def require_runtime() -> None:
    missing = []
    failed = []
    for module_name in ("mido", "torch", "transformers", "anticipation"):
        try:
            __import__(module_name)
        except ImportError:
            missing.append(module_name)
        except OSError as exc:
            failed.append((module_name, exc))
    if missing:
        raise SystemExit(
            "Missing runtime modules: "
            + ", ".join(missing)
            + ". Install project dependencies in .python_deps first."
        )
    if failed:
        details = "; ".join(f"{name}: {exc}" for name, exc in failed)
        raise SystemExit(
            "Runtime modules are present but failed to load native libraries. "
            f"Python executable: {sys.executable}. Details: {details}. "
            "Use a PyTorch-compatible Python interpreter, for example Python 3.12, "
            "or reinstall torch for the current interpreter."
        )


def model_cache_complete(model_id: str) -> bool:
    model_dir = HF_HUB_CACHE / f"models--{model_id.replace('/', '--')}" / "snapshots"
    if not model_dir.exists():
        return False
    for snapshot in model_dir.iterdir():
        if (snapshot / "config.json").exists() and (
            (snapshot / "pytorch_model.bin").exists()
            or (snapshot / "model.safetensors").exists()
        ):
            return True
    return False


def local_snapshot_dir(model_id: str) -> Optional[Path]:
    model_dir = HF_HUB_CACHE / f"models--{model_id.replace('/', '--')}" / "snapshots"
    if not model_dir.exists():
        return None
    for snapshot in model_dir.iterdir():
        if (snapshot / "config.json").exists() and (
            (snapshot / "pytorch_model.bin").exists()
            or (snapshot / "model.safetensors").exists()
        ):
            return snapshot
    return None


def load_transformer_model(model_id: str, offline: bool, endpoint: Optional[str]):
    if endpoint:
        os.environ["HF_ENDPOINT"] = endpoint

    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    else:
        os.environ.pop("HF_HUB_OFFLINE", None)
        os.environ.pop("TRANSFORMERS_OFFLINE", None)

    import torch
    from huggingface_hub import hf_hub_download
    from transformers import AutoModelForCausalLM

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    source = model_id

    if offline:
        snapshot = local_snapshot_dir(model_id)
        if snapshot is None:
            raise SystemExit(
                f"No complete local snapshot for {model_id}. "
                "Run once without --offline to download weights."
            )
        source = str(snapshot)
    else:
        print(f"[model] ensuring {model_id} weights are cached")
        try:
            hf_hub_download(
                repo_id=model_id,
                filename="pytorch_model.bin",
                cache_dir=str(HF_HUB_CACHE),
            )
        except Exception as exc:
            if model_cache_complete(model_id):
                print(f"[model] download check failed, using existing cache: {exc}")
            else:
                raise
        snapshot = local_snapshot_dir(model_id)
        if snapshot is not None:
            source = str(snapshot)

    print(f"[model] loading {source} on {device}")
    model = AutoModelForCausalLM.from_pretrained(
        source,
        cache_dir=str(HF_HUB_CACHE),
        local_files_only=offline,
        use_safetensors=False,
    )
    model.to(device)
    model.eval()
    return model


def model_device_name(model) -> str:
    try:
        return str(model.device)
    except AttributeError:
        try:
            return str(next(model.parameters()).device)
        except StopIteration:
            return "unknown"


def resolve_port(requested: str, available: List[str], direction: str) -> str:
    if requested in available:
        return requested

    matches = [name for name in available if name.lower().startswith(requested.lower())]
    if len(matches) == 1:
        print(f"[ports] resolved {direction} '{requested}' -> '{matches[0]}'")
        return matches[0]
    if len(matches) > 1:
        raise SystemExit(
            f"Multiple {direction} MIDI ports match '{requested}': {matches}. "
            "Pass the exact port name."
        )

    raise SystemExit(
        f"Cannot find {direction} MIDI port '{requested}'. Available ports: {available}. "
        "Run with --list-ports to inspect loopMIDI/VMPK port names."
    )


def notes_to_amt_events(notes: Iterable[CapturedNote]) -> List[int]:
    from anticipation import ops
    from anticipation.vocab import DUR_OFFSET, NOTE_OFFSET, TIME_OFFSET

    events: List[int] = []
    for note in notes:
        onset_tick = max(0, int(round(note.onset * TIME_RESOLUTION)))
        duration_tick = max(1, int(round((note.duration or 0.25) * TIME_RESOLUTION)))
        instrument = 0
        events.extend(
            [
                TIME_OFFSET + onset_tick,
                DUR_OFFSET + duration_tick,
                NOTE_OFFSET + instrument * 128 + note.pitch,
            ]
        )
    return ops.sort(events)


def build_rescue_response_events(
    notes: List[CapturedNote],
    profile: CallProfile,
    plan: ResponsePlan,
    controller: Optional[MusicalResponseController],
    default_note_duration: float,
    min_note_duration: float,
    max_note_duration: float,
) -> List[GeneratedEvent]:
    """Deterministic live fallback used only when the neural decoder emits no notes."""

    source = clean_prompt_phrase(notes, profile) or notes
    if not source:
        return []

    selected = source[-max(1, min(len(source), plan.target_notes)) :]
    source_start = min(note.onset for note in selected)
    source_end = max(
        note.onset + max(note.duration or default_note_duration, min_note_duration)
        for note in selected
    )
    source_span = max(source_end - source_start, 0.001)
    scale = max(0.2, plan.target_seconds / source_span)
    center = int(round(profile.mean_pitch))
    contour_shift = -5 if profile.contour == "descending" else 5

    events: List[GeneratedEvent] = []
    for idx, note in enumerate(selected):
        onset = max(0.0, (note.onset - source_start) * scale)
        tick = max(0, int(round(onset * TIME_RESOLUTION)))
        duration_seconds = max(note.duration or default_note_duration, min_note_duration) * scale
        duration_ticks = clamp_int(
            int(round(duration_seconds * TIME_RESOLUTION)),
            max(1, int(round(min_note_duration * TIME_RESOLUTION))),
            max(1, int(round(max_note_duration * TIME_RESOLUTION))),
        )
        inverted_pitch = center - (note.pitch - center)
        pitch = clamp_int(inverted_pitch + contour_shift, plan.pitch_min, plan.pitch_max)
        event = GeneratedEvent(
            tick=tick,
            duration_ticks=duration_ticks,
            pitch=pitch,
            instrument=0,
            velocity=clamp_int(note.velocity, 1, 127),
        )
        if controller is not None:
            reason = controller.rejection_reason(event)
            if reason is not None:
                controller.reject(reason)
                event = controller.fallback_event(event)
            event = controller.accept(event)
        events.append(event)
        if len(events) >= plan.target_notes:
            break

    return events


def partial_fallback_limit(
    amt_event_count: int,
    target_event_count: int,
    max_fallback_share: float,
) -> int:
    """Cap partial motif completion so AMT remains the primary response source.

    A non-empty live response may be completed after the AMT streaming window,
    but motif events never outnumber the AMT events that were actually heard.
    A zero-event response still uses the existing full fallback path.
    """

    if amt_event_count <= 0 or target_event_count <= amt_event_count:
        return 0
    share_cap = max(0, int(math.floor(target_event_count * clamp_float(max_fallback_share, 0.0, 1.0))))
    missing = target_event_count - amt_event_count
    return min(missing, share_cap, amt_event_count)


def build_partial_motif_completion(
    amt_events: List[GeneratedEvent],
    template_events: List[GeneratedEvent],
    fill_count: int,
    grid_ticks: int,
    controller: Optional[MusicalResponseController],
) -> List[GeneratedEvent]:
    """Append a bounded motif tail after already-streamed AMT events."""

    if not amt_events or not template_events or fill_count <= 0:
        return []
    last_tick = max(event.tick for event in amt_events)
    step = max(1, int(grid_ticks))
    completed: List[GeneratedEvent] = []
    for index in range(fill_count):
        source = template_events[(len(amt_events) + index) % len(template_events)]
        event = GeneratedEvent(
            tick=last_tick + step * (index + 1),
            duration_ticks=source.duration_ticks,
            pitch=source.pitch,
            instrument=source.instrument,
            velocity=source.velocity,
        )
        if controller is not None:
            reason = controller.rejection_reason(event)
            if reason is not None:
                controller.reject(reason)
                event = controller.fallback_event(event)
            event = controller.accept(event)
        completed.append(event)
    return completed


def response_duration_seconds(
    events: List[GeneratedEvent],
    min_note_duration: float,
    max_note_duration: float,
) -> float:
    if not events:
        return 0.0

    first_tick = min(event.tick for event in events)
    min_duration_ticks = max(1, int(round(min_note_duration * TIME_RESOLUTION)))
    max_duration_ticks = max(min_duration_ticks, int(round(max_note_duration * TIME_RESOLUTION)))
    last_tick = max(
        event.tick + clamp_int(event.duration_ticks, min_duration_ticks, max_duration_ticks)
        for event in events
    )
    return max(0.0, (last_tick - first_tick) / TIME_RESOLUTION)


def shape_response_duration(
    events: List[GeneratedEvent],
    plan: ResponsePlan,
    min_note_duration: float,
    max_note_duration: float,
    min_share: float,
    max_share: float,
) -> Tuple[List[GeneratedEvent], float, float, float]:
    if not events:
        return [], 0.0, 0.0, 1.0

    ordered = sorted(events, key=lambda item: (item.tick, item.pitch, item.duration_ticks))
    raw_duration = response_duration_seconds(ordered, min_note_duration, max_note_duration)
    target_seconds = max(plan.target_seconds, min_note_duration)
    min_allowed = target_seconds * min_share
    max_allowed = target_seconds * max_share
    min_duration_ticks = max(1, int(round(min_note_duration * TIME_RESOLUTION)))
    max_duration_ticks = max(min_duration_ticks, int(round(max_note_duration * TIME_RESOLUTION)))

    if min_allowed <= raw_duration <= max_allowed:
        normalized = [
            GeneratedEvent(
                tick=event.tick,
                duration_ticks=clamp_int(event.duration_ticks, min_duration_ticks, max_duration_ticks),
                pitch=event.pitch,
                instrument=event.instrument,
                velocity=event.velocity,
            )
            for event in ordered
        ]
        actual_duration = response_duration_seconds(normalized, min_note_duration, max_note_duration)
        return normalized, raw_duration, actual_duration, 1.0

    first_tick = min(event.tick for event in ordered)
    onset_span = max(1, max(event.tick for event in ordered) - first_tick)
    target_onset_span = max(
        1,
        int(round(max(target_seconds - min_note_duration, min_note_duration) * TIME_RESOLUTION)),
    )

    shaped: List[GeneratedEvent] = []
    if raw_duration < min_allowed:
        raw_onset_span = max(event.tick for event in ordered) - first_tick
        cluster_ratio = 1.0 - min(1.0, raw_onset_span / max(1, target_onset_span))
        sequence_mix = clamp_float(0.35 + 0.45 * cluster_ratio, 0.35, 0.80)
        duration_scale = min(2.0, target_seconds / max(raw_duration, 0.001))
        count = len(ordered)
        for index, event in enumerate(ordered):
            original_position = (event.tick - first_tick) / onset_span
            sequence_position = index / max(1, count - 1)
            position = (1.0 - sequence_mix) * original_position + sequence_mix * sequence_position
            tick = first_tick + int(round(position * target_onset_span))
            duration_ticks = clamp_int(
                int(round(event.duration_ticks * duration_scale)),
                min_duration_ticks,
                max_duration_ticks,
            )
            shaped.append(
                GeneratedEvent(
                    tick=tick,
                    duration_ticks=duration_ticks,
                    pitch=event.pitch,
                    instrument=event.instrument,
                    velocity=event.velocity,
                )
            )
    else:
        scale = target_seconds / max(raw_duration, 0.001)
        for event in ordered:
            tick = first_tick + int(round((event.tick - first_tick) * scale))
            duration_ticks = clamp_int(
                int(round(event.duration_ticks * scale)),
                min_duration_ticks,
                max_duration_ticks,
            )
            shaped.append(
                GeneratedEvent(
                    tick=tick,
                    duration_ticks=duration_ticks,
                    pitch=event.pitch,
                    instrument=event.instrument,
                    velocity=event.velocity,
                )
            )

    shaped = sorted(shaped, key=lambda item: (item.tick, item.pitch, item.duration_ticks))
    actual_duration = response_duration_seconds(shaped, min_note_duration, max_note_duration)
    stretch_factor = actual_duration / raw_duration if raw_duration > 0 else 1.0
    return shaped, raw_duration, actual_duration, stretch_factor


def project_live_event_to_call_scale(
    event: GeneratedEvent,
    profile: CallProfile,
    plan: ResponsePlan,
    previous_pitch: Optional[int],
    max_melodic_leap: int = 7,
) -> GeneratedEvent:
    """Project one streaming event into the inferred Call-relative pentatonic key."""

    intervals = (0, 3, 5, 7, 10) if profile.scale_mode == "minor" else (0, 2, 4, 7, 9)
    pitch_classes = {(profile.scale_root + interval) % 12 for interval in intervals}
    candidates = [
        pitch
        for pitch in range(plan.pitch_min, plan.pitch_max + 1)
        if pitch % 12 in pitch_classes
    ]
    if not candidates:
        return event
    allowed = candidates
    if previous_pitch is not None and max_melodic_leap > 0:
        local = [pitch for pitch in candidates if abs(pitch - previous_pitch) <= max_melodic_leap]
        if local:
            allowed = local
    pitch = min(allowed, key=lambda value: (abs(value - event.pitch), value))
    if previous_pitch is not None and pitch == previous_pitch and len(allowed) > 1:
        alternatives = [value for value in allowed if value != previous_pitch]
        pitch = min(
            alternatives,
            key=lambda value: (abs(value - event.pitch), abs(value - previous_pitch)),
        )
    return GeneratedEvent(
        tick=event.tick,
        duration_ticks=event.duration_ticks,
        pitch=pitch,
        instrument=event.instrument,
        velocity=event.velocity,
    )


def apply_live_pentatonic_style(
    events: List[GeneratedEvent],
    profile: CallProfile,
    plan: ResponsePlan,
    max_melodic_leap: int = 7,
) -> List[GeneratedEvent]:
    """Apply the paper's A5/A6 pentatonic and cadence controls to live output.

    The offline evaluation accepts an explicit style root.  In the live path we
    derive that root and major/minor mode from the CallProfile so the constraint
    follows the performer instead of being fixed to C.
    """

    if not events:
        return []
    intervals = (0, 3, 5, 7, 10) if profile.scale_mode == "minor" else (0, 2, 4, 7, 9)
    pitch_classes = {(profile.scale_root + interval) % 12 for interval in intervals}
    candidates = [
        pitch
        for pitch in range(plan.pitch_min, plan.pitch_max + 1)
        if pitch % 12 in pitch_classes
    ]
    if not candidates:
        return events

    styled: List[GeneratedEvent] = []
    previous_pitch: Optional[int] = None
    for event in sorted(events, key=lambda item: (item.tick, item.pitch)):
        projected = project_live_event_to_call_scale(
            event,
            profile,
            plan,
            previous_pitch,
            max_melodic_leap=max_melodic_leap,
        )
        styled.append(projected)
        previous_pitch = projected.pitch

    cadence_candidates = [pitch for pitch in candidates if pitch % 12 == profile.scale_root]
    if cadence_candidates:
        last = styled[-1]
        previous_to_cadence = styled[-2].pitch if len(styled) > 1 else None
        local_cadences = [
            pitch
            for pitch in cadence_candidates
            if (
                previous_to_cadence is None
                or (
                    pitch != previous_to_cadence
                    and abs(pitch - previous_to_cadence) <= max_melodic_leap
                )
            )
        ]
        if local_cadences:
            cadence_candidates = local_cadences
        cadence = min(cadence_candidates, key=lambda value: abs(value - last.pitch))
        if len(styled) > 1 and styled[-2].pitch == cadence:
            penultimate = styled[-2]
            earlier_pitch = styled[-3].pitch if len(styled) > 2 else None
            approach_candidates = [
                pitch
                for pitch in candidates
                if (
                    pitch != cadence
                    and abs(pitch - cadence) <= max_melodic_leap
                    and (
                        earlier_pitch is None
                        or abs(pitch - earlier_pitch) <= max_melodic_leap
                    )
                )
            ]
            if approach_candidates:
                approach = min(
                    approach_candidates,
                    key=lambda value: abs(value - penultimate.pitch),
                )
                styled[-2] = GeneratedEvent(
                    tick=penultimate.tick,
                    duration_ticks=penultimate.duration_ticks,
                    pitch=approach,
                    instrument=penultimate.instrument,
                    velocity=penultimate.velocity,
                )
        styled[-1] = GeneratedEvent(
            tick=last.tick,
            duration_ticks=last.duration_ticks,
            pitch=cadence,
            instrument=last.instrument,
            velocity=last.velocity,
        )
    return styled


def split_event(token_triplet: List[int]) -> GeneratedEvent:
    from anticipation.vocab import DUR_OFFSET, NOTE_OFFSET, TIME_OFFSET

    tick = token_triplet[0] - TIME_OFFSET
    duration_ticks = max(1, token_triplet[1] - DUR_OFFSET)
    note = token_triplet[2] - NOTE_OFFSET
    instrument = note // 128
    pitch = note - instrument * 128
    return GeneratedEvent(tick=tick, duration_ticks=duration_ticks, pitch=pitch, instrument=instrument)


def event_to_tokens(event: GeneratedEvent) -> List[int]:
    from anticipation.vocab import DUR_OFFSET, NOTE_OFFSET, TIME_OFFSET

    tick = max(0, int(event.tick))
    duration_ticks = max(1, int(event.duration_ticks))
    pitch = clamp_int(int(event.pitch), 0, 127)
    instrument = clamp_int(int(event.instrument), 0, 127)
    return [
        TIME_OFFSET + tick,
        DUR_OFFSET + duration_ticks,
        NOTE_OFFSET + instrument * 128 + pitch,
    ]


class MusicalResponseController:
    def __init__(self, profile: CallProfile, plan: ResponsePlan) -> None:
        self.profile = profile
        self.plan = plan
        self.stats = MusicalControlStats()
        self.accepted_events: List[GeneratedEvent] = []
        self.pitch_counts: Counter[int] = Counter()

    @property
    def accepted_count(self) -> int:
        return len(self.accepted_events)

    def should_stop(self, start_tick: int) -> bool:
        if not self.accepted_events:
            return False
        elapsed = (self.accepted_events[-1].tick - start_tick) / TIME_RESOLUTION
        if self.accepted_count >= self.plan.target_notes:
            if self.plan.stop_on_target_notes:
                return True
            return elapsed >= self.plan.target_seconds * self.plan.duration_match_min_share
        return elapsed >= self.plan.target_seconds

    def _same_pitch_run(self, pitch: int) -> int:
        run = 0
        for event in reversed(self.accepted_events):
            if event.pitch != pitch:
                break
            run += 1
        return run

    def rejection_reason(self, event: GeneratedEvent) -> Optional[str]:
        pitch = clamp_int(event.pitch, self.plan.pitch_min, self.plan.pitch_max)
        if self._same_pitch_run(pitch) >= self.plan.same_pitch_limit:
            return "repeat"

        projected_total = self.accepted_count + 1
        projected_count = self.pitch_counts[pitch] + 1
        enough_context = projected_total >= max(4, self.plan.same_pitch_limit + 2)
        if enough_context and projected_count / projected_total > self.plan.dominant_pitch_max_share:
            return "dominant"
        return None

    def normalize_event(self, event: GeneratedEvent) -> GeneratedEvent:
        return GeneratedEvent(
            tick=event.tick,
            duration_ticks=event.duration_ticks,
            pitch=clamp_int(event.pitch, self.plan.pitch_min, self.plan.pitch_max),
            instrument=event.instrument,
            velocity=clamp_int(event.velocity, 1, 127),
        )

    def accept(self, event: GeneratedEvent) -> GeneratedEvent:
        normalized = self.normalize_event(event)
        self.accepted_events.append(normalized)
        self.pitch_counts[normalized.pitch] += 1
        return normalized

    def reject(self, reason: str) -> None:
        if reason == "repeat":
            self.stats.rejected_repeat_count += 1
        elif reason == "dominant":
            self.stats.rejected_dominant_count += 1

    def fallback_event(self, event: GeneratedEvent) -> GeneratedEvent:
        base_pitch = int(round(self.profile.mean_pitch))
        offsets = [2, -2, 3, -3, 5, -5, 7, -7, 1, -1, 12, -12, 0]
        last_pitch = self.accepted_events[-1].pitch if self.accepted_events else None
        for offset in offsets:
            pitch = clamp_int(base_pitch + offset, self.plan.pitch_min, self.plan.pitch_max)
            if last_pitch is not None and pitch == last_pitch:
                continue
            candidate = GeneratedEvent(
                tick=event.tick,
                duration_ticks=event.duration_ticks,
                pitch=pitch,
                instrument=event.instrument,
                velocity=event.velocity,
            )
            if self.rejection_reason(candidate) is None:
                self.stats.fallback_count += 1
                return candidate

        self.stats.fallback_count += 1
        return self.normalize_event(event)


class StreamingAMTGenerator:
    """Small wrapper around AMT logits sampling that yields one event at a time."""

    def __init__(
        self,
        model,
        top_p: float,
        temperature: float,
        max_history_events: int = 339,
    ) -> None:
        self.model = model
        self.top_p = top_p
        self.temperature = temperature
        self.max_history_tokens = max_history_events * 3

    def _sample_event(self, z: List[int], tokens: List[int], current_time: int) -> List[int]:
        import torch
        from anticipation import ops
        from anticipation.sample import future_logits, instr_logits, nucleus, safe_logits
        from anticipation.vocab import DUR_OFFSET, NOTE_OFFSET, TIME_OFFSET
        from anticipation.config import MAX_DUR, MAX_NOTE, MAX_TIME

        history = tokens[-self.max_history_tokens :].copy()
        offset = ops.min_time(history, seconds=False)
        history[0::3] = [tok - offset for tok in history[0::3]]

        def fallback_token(part_idx: int) -> int:
            if part_idx == 0:
                relative_time = clamp_int(current_time - offset, 0, MAX_TIME - 1)
                return TIME_OFFSET + relative_time
            if part_idx == 1:
                return DUR_OFFSET + 25
            pitches = []
            for note_token in tokens[2::3][-8:]:
                note = note_token - NOTE_OFFSET
                if note >= 0:
                    pitches.append(note % 128)
            pitch = int(round(sum(pitches) / len(pitches))) if pitches else 60
            return NOTE_OFFSET + clamp_int(pitch, 0, 127)

        def sample_token(logits, fallback: int) -> int:
            import torch.nn.functional as F

            logits = torch.nan_to_num(
                logits.detach().clone(),
                nan=-float("inf"),
                posinf=1e4,
                neginf=-float("inf"),
            )
            finite_mask = torch.isfinite(logits)
            if not bool(finite_mask.any().item()):
                return fallback

            pre_nucleus = logits.clone()
            logits = nucleus(logits, self.top_p)
            finite_mask = torch.isfinite(logits)
            if not bool(finite_mask.any().item()):
                logits = pre_nucleus

            finite_mask = torch.isfinite(logits)
            if not bool(finite_mask.any().item()):
                return fallback

            finite_logits = logits[finite_mask].float().cpu()
            finite_indices = torch.nonzero(finite_mask, as_tuple=False).flatten().cpu()
            probs = F.softmax(finite_logits, dim=-1)
            if (
                probs.numel() == 0
                or not bool(torch.isfinite(probs).all().item())
                or float(probs.sum().item()) <= 0.0
            ):
                best = int(torch.argmax(finite_logits).item())
                return int(finite_indices[best].item())
            sampled = torch.multinomial(probs, 1)
            return int(finite_indices[int(sampled.item())].item())

        new_token: List[int] = []
        with torch.no_grad():
            for part_idx in range(3):
                input_tokens = torch.tensor(z + history + new_token).unsqueeze(0).to(self.model.device)
                logits = self.model(input_tokens).logits[0, -1]
                idx = input_tokens.shape[1] - 1
                logits = safe_logits(logits, idx)
                if part_idx == 0:
                    logits = future_logits(logits, current_time - offset)
                elif part_idx == 2:
                    logits = instr_logits(logits, tokens)
                if self.temperature > 0:
                    logits = logits / self.temperature
                fallback = fallback_token(part_idx)
                token = sample_token(logits, fallback)
                valid_ranges = (
                    (TIME_OFFSET, TIME_OFFSET + MAX_TIME),
                    (DUR_OFFSET, DUR_OFFSET + MAX_DUR),
                    (NOTE_OFFSET, NOTE_OFFSET + MAX_NOTE),
                )
                low, high = valid_ranges[part_idx]
                if not low <= token < high:
                    token = fallback
                new_token.append(token)

        new_token[0] += offset
        return new_token

    def generate_events(
        self,
        call_events: List[int],
        response_seconds: float,
        stop_event: threading.Event,
        controller: Optional[MusicalResponseController] = None,
    ):
        from anticipation import ops
        from anticipation.config import TIME_RESOLUTION
        from anticipation.vocab import AUTOREGRESS

        tokens = ops.sort(call_events.copy())
        start_tick = int(round(ops.max_time(tokens) * TIME_RESOLUTION))
        tokens = ops.pad(tokens, start_tick)
        end_tick = start_tick + int(round(response_seconds * TIME_RESOLUTION))
        current_time = start_tick
        z = [AUTOREGRESS]

        while not stop_event.is_set() and current_time < end_tick:
            if controller is not None and controller.should_stop(start_tick):
                break

            accepted_event: Optional[GeneratedEvent] = None
            raw_event: Optional[List[int]] = None
            attempts = controller.plan.resample_attempts if controller is not None else 1
            max_attempts = max(1, attempts)
            last_early_event: Optional[GeneratedEvent] = None
            for attempt_idx in range(max_attempts):
                candidate_raw = self._sample_event(z, tokens, max(start_tick, current_time))
                candidate_event = split_event(candidate_raw)
                if candidate_event.tick >= end_tick:
                    raw_event = candidate_raw
                    break
                if candidate_event.tick <= current_time:
                    last_early_event = candidate_event
                    if attempt_idx == max_attempts - 1:
                        repaired_tick = max(start_tick + 1, current_time + 1)
                        candidate_event = GeneratedEvent(
                            tick=repaired_tick,
                            duration_ticks=candidate_event.duration_ticks,
                            pitch=candidate_event.pitch,
                            instrument=candidate_event.instrument,
                            velocity=candidate_event.velocity,
                        )
                    else:
                        continue

                if controller is None:
                    accepted_event = candidate_event
                    raw_event = event_to_tokens(accepted_event)
                    break

                reason = controller.rejection_reason(candidate_event)
                if reason is None:
                    accepted_event = controller.accept(candidate_event)
                    raw_event = event_to_tokens(accepted_event)
                    break

                controller.reject(reason)
                if attempt_idx == attempts - 1:
                    accepted_event = controller.accept(controller.fallback_event(candidate_event))
                    raw_event = event_to_tokens(accepted_event)

            if raw_event is None or accepted_event is None:
                if last_early_event is None:
                    break
                repaired_event = GeneratedEvent(
                    tick=max(start_tick + 1, current_time + 1),
                    duration_ticks=last_early_event.duration_ticks,
                    pitch=last_early_event.pitch,
                    instrument=last_early_event.instrument,
                    velocity=last_early_event.velocity,
                )
                if controller is not None:
                    repaired_event = controller.accept(controller.fallback_event(repaired_event))
                accepted_event = repaired_event
                raw_event = event_to_tokens(repaired_event)
            if raw_event[0] >= end_tick:
                break

            tokens.extend(raw_event)
            tokens = ops.sort(tokens)
            yield accepted_event
            current_time = max(current_time, accepted_event.tick)


class AriaLiveGenerator:
    """Aria continuation backend adapted to the existing MIDI playback queue."""

    def __init__(
        self,
        model_id: str,
        offline: bool,
        endpoint: Optional[str],
        hf_token: Optional[str],
        top_p: float,
        temperature: float,
        max_new_tokens: int,
        prompt_tokens: int,
        response_bars: int,
        beats_per_bar: int,
        bpm: float,
        monophonic_response: bool,
        stream_chunk_tokens: int,
        stream_fallback_seconds: float,
    ) -> None:
        from aria_call_response_once import (
            load_model as load_aria_model,
            load_tokenizer as load_aria_tokenizer,
            patch_transformers_for_aria,
        )

        patch_transformers_for_aria()
        self.model, snapshot = load_aria_model(model_id, offline, endpoint, hf_token)
        if snapshot is None:
            raise SystemExit(f"Could not resolve local Aria tokenizer files for {model_id}.")
        self.tokenizer = load_aria_tokenizer(snapshot)
        self.top_p = top_p
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.prompt_tokens = prompt_tokens
        self.response_bars = response_bars
        self.beats_per_bar = beats_per_bar
        self.bpm = bpm
        self.monophonic_response = monophonic_response
        self.stream_chunk_tokens = stream_chunk_tokens
        self.stream_fallback_seconds = stream_fallback_seconds

    def _notes_to_midi_dict(self, notes: List[CapturedNote]):
        import mido
        from ariautils.midi import MidiDict

        ticks_per_beat = 480
        tempo = mido.bpm2tempo(self.bpm)
        midi = mido.MidiFile(ticks_per_beat=ticks_per_beat)
        track = mido.MidiTrack()
        midi.tracks.append(track)
        absolute_messages: List[Tuple[int, int, object]] = [
            (0, 0, mido.MetaMessage("set_tempo", tempo=tempo, time=0)),
            (
                0,
                1,
                mido.MetaMessage(
                    "time_signature",
                    numerator=self.beats_per_bar,
                    denominator=4,
                    time=0,
                ),
            ),
            (0, 2, mido.Message("program_change", program=0, channel=0, time=0)),
        ]
        prompt_end_tick = 0
        for note in notes:
            start_tick = max(0, int(round(mido.second2tick(note.onset, ticks_per_beat, tempo))))
            duration = max(0.03, note.duration or 0.25)
            end_tick = max(start_tick + 1, int(round(mido.second2tick(note.onset + duration, ticks_per_beat, tempo))))
            pitch = clamp_int(note.pitch, 0, 127)
            velocity = clamp_int(note.velocity, 1, 127)
            absolute_messages.append((start_tick, 3, mido.Message("note_on", note=pitch, velocity=velocity, channel=0, time=0)))
            absolute_messages.append((end_tick, 2, mido.Message("note_off", note=pitch, velocity=0, channel=0, time=0)))
            prompt_end_tick = max(prompt_end_tick, end_tick)

        last_tick = 0
        for tick, _, msg in sorted(absolute_messages, key=lambda item: (item[0], item[1])):
            msg.time = max(0, tick - last_tick)
            track.append(msg)
            last_tick = tick

        with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as handle:
            temp_path = Path(handle.name)
        try:
            midi.save(temp_path)
            midi_dict = MidiDict.from_midi(temp_path)
        finally:
            temp_path.unlink(missing_ok=True)
        return midi_dict, prompt_end_tick, ticks_per_beat, tempo

    def _input_ids(self, midi_dict):
        encoded = self.tokenizer(midi_dict, return_tensors="pt")
        input_ids = encoded["input_ids"]
        if self.prompt_tokens > 0 and input_ids.shape[-1] > self.prompt_tokens:
            input_ids = input_ids[..., : self.prompt_tokens]
        return input_ids.to(self.model.device)

    def _extract_response_events(
        self,
        decoded_midi_dict,
        prompt_end_tick: int,
        ticks_per_beat: int,
        tempo: int,
        response_seconds: float,
    ) -> List[GeneratedEvent]:
        import mido
        from aria_call_response_once import select_monophonic_notes

        msg_dict = decoded_midi_dict.get_msg_dict()
        source_ticks_per_beat = int(msg_dict.get("ticks_per_beat") or ticks_per_beat)
        bar_window_ticks = source_ticks_per_beat * self.response_bars * self.beats_per_bar
        seconds_window_ticks = max(
            1,
            int(round(mido.second2tick(response_seconds, source_ticks_per_beat, tempo))),
        )
        response_end_tick = prompt_end_tick + min(bar_window_ticks, seconds_window_ticks)
        response_notes = []
        for note in msg_dict["note_msgs"]:
            start = int(note["data"]["start"])
            end = int(note["data"]["end"])
            if start < prompt_end_tick or start >= response_end_tick:
                continue
            clipped_end = max(start + 1, min(end, response_end_tick))
            response_notes.append(
                {
                    **note,
                    "tick": max(0, start - prompt_end_tick),
                    "data": {
                        **note["data"],
                        "start": max(0, start - prompt_end_tick),
                        "end": max(1, clipped_end - prompt_end_tick),
                    },
                }
            )

        if self.monophonic_response:
            response_notes = select_monophonic_notes(response_notes)

        events: List[GeneratedEvent] = []
        for note in sorted(response_notes, key=lambda item: (int(item["data"]["start"]), int(item["data"]["pitch"]))):
            start_tick = int(note["data"]["start"])
            end_tick = int(note["data"]["end"])
            onset_seconds = mido.tick2second(start_tick, source_ticks_per_beat, tempo)
            duration_seconds = max(
                0.03,
                mido.tick2second(max(1, end_tick - start_tick), source_ticks_per_beat, tempo),
            )
            events.append(
                GeneratedEvent(
                    tick=max(0, int(round(onset_seconds * TIME_RESOLUTION))),
                    duration_ticks=max(1, int(round(duration_seconds * TIME_RESOLUTION))),
                    pitch=clamp_int(int(note["data"]["pitch"]), 0, 127),
                    instrument=0,
                    velocity=clamp_int(int(note["data"].get("velocity", 92)), 1, 127),
                )
            )
        return events

    def _controlled_events(
        self,
        events: List[GeneratedEvent],
        controller: Optional[MusicalResponseController],
    ) -> List[GeneratedEvent]:
        if controller is None:
            return events

        accepted: List[GeneratedEvent] = []
        for event in sorted(events, key=lambda item: item.tick):
            if controller.should_stop(0):
                break
            reason = controller.rejection_reason(event)
            if reason is None:
                accepted.append(controller.accept(event))
                continue
            controller.reject(reason)
            accepted.append(controller.accept(controller.fallback_event(event)))
        return accepted

    def _decode_events_from_ids(
        self,
        token_ids,
        prompt_end_tick: int,
        ticks_per_beat: int,
        tempo: int,
        response_seconds: float,
    ) -> List[GeneratedEvent]:
        decoded = self.tokenizer.decode(token_ids.detach().cpu().tolist())
        return self._extract_response_events(decoded, prompt_end_tick, ticks_per_beat, tempo, response_seconds)

    def _generate_batch_events(
        self,
        input_ids,
        prompt_end_tick: int,
        ticks_per_beat: int,
        tempo: int,
        response_seconds: float,
        controller: Optional[MusicalResponseController],
    ) -> List[GeneratedEvent]:
        import torch

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=input_ids,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        events = self._decode_events_from_ids(
            output_ids[0],
            prompt_end_tick,
            ticks_per_beat,
            tempo,
            response_seconds,
        )
        return self._controlled_events(events, controller)

    def _generate_stream_events(
        self,
        input_ids,
        prompt_end_tick: int,
        ticks_per_beat: int,
        tempo: int,
        response_seconds: float,
        stop_event: threading.Event,
        controller: Optional[MusicalResponseController],
    ):
        import torch

        current_ids = input_ids
        generated_tokens = 0
        emitted_keys: set[Tuple[int, int, int, int]] = set()
        emitted_count = 0
        last_emitted_tick = -1
        first_event_deadline = time.perf_counter() + self.stream_fallback_seconds

        while not stop_event.is_set() and generated_tokens < self.max_new_tokens:
            chunk_tokens = min(self.stream_chunk_tokens, self.max_new_tokens - generated_tokens)
            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids=current_ids,
                    max_new_tokens=chunk_tokens,
                    do_sample=True,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            new_tokens = int(output_ids.shape[-1] - current_ids.shape[-1])
            if new_tokens <= 0:
                break
            current_ids = output_ids
            generated_tokens += new_tokens

            try:
                events = self._decode_events_from_ids(
                    current_ids[0],
                    prompt_end_tick,
                    ticks_per_beat,
                    tempo,
                    response_seconds,
                )
            except Exception as exc:
                print(f"[aria-stream] partial decode skipped: {exc}")
                events = []

            if controller is not None:
                events = self._controlled_events(events, controller)

            emitted_this_chunk = False
            for event in sorted(events, key=lambda item: item.tick):
                key = (event.tick, event.duration_ticks, event.pitch, event.instrument)
                if key in emitted_keys or event.tick < last_emitted_tick:
                    continue
                emitted_keys.add(key)
                last_emitted_tick = event.tick
                emitted_count += 1
                emitted_this_chunk = True
                yield event

            if emitted_count == 0 and time.perf_counter() >= first_event_deadline:
                print("[aria-stream] no stable partial notes yet; falling back to batch generation")
                for event in self._generate_batch_events(
                    input_ids,
                    prompt_end_tick,
                    ticks_per_beat,
                    tempo,
                    response_seconds,
                    controller,
                ):
                    yield event
                return

            if not emitted_this_chunk:
                time.sleep(0.001)

        if emitted_count == 0 and not stop_event.is_set():
            print("[aria-stream] finished without notes; falling back to batch generation")
            for event in self._generate_batch_events(
                input_ids,
                prompt_end_tick,
                ticks_per_beat,
                tempo,
                response_seconds,
                controller,
            ):
                yield event

    def generate_events(
        self,
        notes: List[CapturedNote],
        response_seconds: float,
        stop_event: threading.Event,
        controller: Optional[MusicalResponseController],
        live_mode: str,
    ):
        midi_dict, prompt_end_tick, ticks_per_beat, tempo = self._notes_to_midi_dict(notes)
        input_ids = self._input_ids(midi_dict)
        print(
            f"[aria] prompt_notes={len(notes)} prompt_tokens={input_ids.shape[-1]} "
            f"prompt_end_tick={prompt_end_tick} mode={live_mode}"
        )
        if live_mode == "stream":
            yield from self._generate_stream_events(
                input_ids,
                prompt_end_tick,
                ticks_per_beat,
                tempo,
                response_seconds,
                stop_event,
                controller,
            )
            return

        for event in self._generate_batch_events(
            input_ids,
            prompt_end_tick,
            ticks_per_beat,
            tempo,
            response_seconds,
            controller,
        ):
            if stop_event.is_set():
                break
            yield event


class LockedOutputPort:
    def __init__(self, outport, lock: threading.Lock) -> None:
        self.outport = outport
        self.lock = lock

    def send(self, msg) -> None:
        with self.lock:
            self.outport.send(msg)


class ResponsePlayer:
    def __init__(
        self,
        outport,
        events_queue: "queue.Queue[Optional[GeneratedEvent]]",
        initial_buffer_events: int,
        initial_buffer_timeout: float,
        first_event_timeout: float,
        max_underrun_seconds: float,
        producer_done: Optional[threading.Event],
        pitch_min: int,
        pitch_max: int,
        min_note_duration: float,
        max_note_duration: float,
        rhythm_timing_provider: Optional[Callable[[], Optional[Tuple[float, float]]]] = None,
        rhythm_quantize_strength: float = 0.92,
        cancel_event: Optional[threading.Event] = None,
    ) -> None:
        self.outport = outport
        self.events_queue = events_queue
        self.initial_buffer_events = initial_buffer_events
        self.initial_buffer_timeout = initial_buffer_timeout
        self.first_event_timeout = first_event_timeout
        self.max_underrun_seconds = max_underrun_seconds
        self.producer_done = producer_done
        self.pitch_min = pitch_min
        self.pitch_max = pitch_max
        self.min_note_duration = min_note_duration
        self.max_note_duration = max_note_duration
        self.rhythm_timing_provider = rhythm_timing_provider
        self.rhythm_quantize_strength = clamp_float(rhythm_quantize_strength, 0.0, 1.0)
        self.cancel_event = cancel_event
        self.buffer_underruns = 0
        self.initial_buffer_reached_end = False

    def _send_due_note_offs(self, active_notes: List[Tuple[float, int, int]]) -> None:
        import mido

        now = time.monotonic()
        remaining: List[Tuple[float, int, int]] = []
        for due_time, pitch, channel in active_notes:
            if due_time <= now:
                self.outport.send(mido.Message("note_off", note=pitch, velocity=0, channel=channel))
                print(f"[ai_out] note_off pitch={pitch:3d} velocity=  0", flush=True)
            else:
                remaining.append((due_time, pitch, channel))
        active_notes[:] = remaining

    def _release_active_notes(self, active_notes: List[Tuple[float, int, int]]) -> None:
        import mido

        for _, pitch, channel in active_notes:
            self.outport.send(mido.Message("note_off", note=pitch, velocity=0, channel=channel))
            print(f"[ai_out] note_off pitch={pitch:3d} velocity=  0", flush=True)
        active_notes.clear()

    def _sleep_until(
        self,
        target_time: float,
        active_notes: List[Tuple[float, int, int]],
    ) -> bool:
        while True:
            self._send_due_note_offs(active_notes)
            if self.cancel_event is not None and self.cancel_event.is_set():
                self._release_active_notes(active_notes)
                return False
            now = time.monotonic()
            if now >= target_time:
                return True
            next_due = min((item[0] for item in active_notes), default=target_time)
            sleep_until = min(target_time, next_due)
            time.sleep(max(0.001, min(0.05, sleep_until - now)))

    def _collect_initial_buffer(self) -> Tuple[List[GeneratedEvent], float]:
        buffer_start = time.monotonic()
        buffered: List[GeneratedEvent] = []
        soft_deadline = time.monotonic() + self.initial_buffer_timeout
        first_event_deadline = time.monotonic() + max(self.initial_buffer_timeout, self.first_event_timeout)
        while len(buffered) < self.initial_buffer_events:
            if self.cancel_event is not None and self.cancel_event.is_set():
                break
            now = time.monotonic()
            if buffered and now >= soft_deadline:
                break
            if not buffered and now >= first_event_deadline:
                print(
                    f"[playback] no first event after {self.first_event_timeout:.1f}s; "
                    "declaring empty response"
                )
                break
            deadline = soft_deadline if buffered else first_event_deadline
            timeout = max(0.01, min(0.10, deadline - now))
            try:
                item = self.events_queue.get(timeout=timeout)
            except queue.Empty:
                continue
            if item is None:
                self.initial_buffer_reached_end = True
                break
            buffered.append(item)
        return buffered, time.monotonic() - buffer_start

    def play(self) -> PlaybackStats:
        import mido

        buffered, initial_buffer_wait = self._collect_initial_buffer()
        if not buffered:
            print("[playback] no generated events; nothing to play")
            now = time.monotonic()
            return PlaybackStats(
                playback_start_time=None,
                playback_end_time=now,
                initial_buffer_wait=initial_buffer_wait,
                initial_buffer_count=0,
                played_events=0,
                buffer_underruns=self.buffer_underruns,
            )

        first_tick = buffered[0].tick
        playback_start = time.monotonic()
        rhythm_bpm: Optional[float] = None
        if self.rhythm_timing_provider is not None:
            timing = self.rhythm_timing_provider()
            if timing is not None:
                rhythm_bpm, phase = timing
                beat_seconds = 60.0 / rhythm_bpm
                # Stay locked to the learnt groove without holding a finished
                # AI answer for an entire beat.  Starting on the next 1/16
                # keeps it musical and caps the added wait at one subdivision.
                subdivision_seconds = beat_seconds / 4.0
                steps_elapsed = max(0.0, (playback_start - phase) / subdivision_seconds)
                playback_start = phase + math.ceil(steps_elapsed) * subdivision_seconds
                if playback_start - time.monotonic() < 0.015:
                    playback_start += subdivision_seconds
                print(
                    f"[playback] rhythm_sync bpm={rhythm_bpm:.1f} grid=1/16 "
                    f"wait={max(0.0, playback_start - time.monotonic()):.3f}s",
                    flush=True,
                )
        initial_buffer_count = len(buffered)
        pending = buffered
        active_notes: List[Tuple[float, int, int]] = []
        played_events = 0
        underrun_started_at: Optional[float] = None
        print(f"[playback] starting with {len(buffered)} buffered events")

        while True:
            self._send_due_note_offs(active_notes)
            if self.cancel_event is not None and self.cancel_event.is_set():
                self._release_active_notes(active_notes)
                print("[playback] cancelled by panic", flush=True)
                break
            if pending:
                event = pending.pop(0)
            else:
                if self.initial_buffer_reached_end:
                    break
                try:
                    item = self.events_queue.get(timeout=0.05)
                except queue.Empty:
                    self.buffer_underruns += 1
                    now = time.monotonic()
                    if underrun_started_at is None:
                        underrun_started_at = now
                    if self.buffer_underruns == 1 or self.buffer_underruns % 25 == 0:
                        print(f"[playback] buffer underrun #{self.buffer_underruns}")
                    if now - underrun_started_at >= self.max_underrun_seconds:
                        print(
                            f"[playback] underrun lasted {now - underrun_started_at:.2f}s; "
                            "ending this response"
                        )
                        break
                    continue
                if item is None:
                    break
                underrun_started_at = None
                event = item

            onset_offset = max(0.0, (event.tick - first_tick) / TIME_RESOLUTION)
            if rhythm_bpm is not None:
                onset_offset = quantize_onset_to_rhythm_grid(
                    onset_offset,
                    rhythm_bpm,
                    self.rhythm_quantize_strength,
                )
            target_time = playback_start + onset_offset
            if not self._sleep_until(target_time, active_notes):
                break

            velocity = clamp_int(getattr(event, "velocity", 92), 1, 127)
            channel = 0 if event.instrument != 128 else 9
            pitch = min(max(event.pitch, self.pitch_min), self.pitch_max)
            self.outport.send(mido.Message("note_on", note=pitch, velocity=velocity, channel=channel))
            print(f"[ai_out] note_on pitch={pitch:3d} velocity={velocity:3d}", flush=True)
            duration = min(
                max(self.min_note_duration, event.duration_ticks / TIME_RESOLUTION),
                self.max_note_duration,
            )
            active_notes.append((time.monotonic() + duration, pitch, channel))
            played_events += 1

        while active_notes:
            if not self._sleep_until(min(item[0] for item in active_notes), active_notes):
                break
        playback_end = time.monotonic()
        print(f"[playback] done; buffer_underruns={self.buffer_underruns}")
        return PlaybackStats(
            playback_start_time=playback_start,
            playback_end_time=playback_end,
            initial_buffer_wait=initial_buffer_wait,
            initial_buffer_count=initial_buffer_count,
            played_events=played_events,
            buffer_underruns=self.buffer_underruns,
        )


class LiveCallResponseApp:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.recorder = PhraseRecorder(default_duration=args.default_note_duration)
        self.stop_event = threading.Event()
        self.busy = threading.Event()
        # A performer must be able to keep playing while the previous AI
        # answer is sounding.  Keep one complete, most-recent phrase ready
        # instead of dropping it just because a response thread is busy.
        self.pending_lock = threading.Lock()
        self.pending_response: Optional[Tuple[List[CapturedNote], EndpointDecision]] = None
        self.playback_cancel_event = threading.Event()
        self.output_lock = threading.Lock()
        self._melody_output = None
        self.speculative_lock = threading.Lock()
        self.speculative_preload: Optional[SpeculativePreloadState] = None
        self.speculative_preload_id = 0
        self.speculative_cancel_count = 0
        self.rhythm_guide = RhythmGuideEngine(drum_channel=args.drum_channel)
        self.loop_bank = ResponseLoopBank()
        self.loop_save_mode = "response"
        self._last_response_lock = threading.Lock()
        self._last_response: List[GeneratedEvent] = []
        self._last_call: List[CapturedNote] = []
        self._monitor_active_notes: set[tuple[int, int]] = set()
        self._control_offset = 0
        self._last_rhythm_status: Optional[Tuple[object, ...]] = None
        self._last_loop_status: Optional[Tuple[object, ...]] = None
        self.model = None
        if args.response_strategy == "motif":
            self.generator = None
        elif args.backend == "aria":
            self.generator = AriaLiveGenerator(
                model_id=args.aria_model_id,
                offline=args.offline,
                endpoint=args.endpoint,
                hf_token=args.hf_token,
                top_p=args.top_p,
                temperature=args.temperature,
                max_new_tokens=args.aria_max_new_tokens,
                prompt_tokens=args.aria_prompt_tokens,
                response_bars=args.aria_response_bars,
                beats_per_bar=args.aria_beats_per_bar,
                bpm=args.aria_bpm,
                monophonic_response=args.aria_monophonic_response,
                stream_chunk_tokens=args.aria_stream_chunk_tokens,
                stream_fallback_seconds=args.aria_stream_fallback_seconds,
            )
            self.model = self.generator.model
        else:
            self.model = load_transformer_model(args.model_id, args.offline, args.endpoint)
            self.generator = StreamingAMTGenerator(
                self.model,
                top_p=args.top_p,
                temperature=args.temperature,
            )
        self.device = "rule-based" if self.model is None else model_device_name(self.model)
        self.metrics = MetricsRecorder(
            path=None if args.no_metrics else Path(args.metrics_path),
            enabled=not args.no_metrics,
        )
        self.round_id = 0
        self.round_lock = threading.Lock()
        self.vad = MidiEndpointVAD(
            theta=args.theta,
            window_size=args.window_size,
            min_intensity=args.min_intensity,
            min_cutoff=args.min_cutoff,
            max_cutoff=args.max_cutoff,
            chord_cluster_window=args.chord_cluster_window,
            endpoint_confirm_delay=args.endpoint_confirm_delay,
            on_candidate_endpoint=self.on_candidate_endpoint,
            on_candidate_cancel=self.on_candidate_cancel,
            on_endpoint=self.on_endpoint,
        )
        print(
            "[startup] backend={} strategy={} model_id={} device={} top_p={} temperature={} "
            "response_seconds={} latency_mode={} initial_buffer={}/{}s".format(
                args.backend,
                args.response_strategy,
                args.aria_model_id if args.backend == "aria" else args.model_id,
                self.device,
                args.top_p,
                args.temperature,
                args.response_seconds,
                args.latency_mode,
                args.initial_buffer_events,
                args.initial_buffer_timeout,
            )
        )
        if args.backend == "aria":
            print(
                "[startup] aria_live_mode={} response_bars={} bpm={} "
                "prompt_tokens={} max_new_tokens={} monophonic={}".format(
                    args.aria_live_mode,
                    args.aria_response_bars,
                    args.aria_bpm,
                    args.aria_prompt_tokens,
                    args.aria_max_new_tokens,
                    "on" if args.aria_monophonic_response else "off",
                )
            )
        print(
            "[startup] pitch_range={}..{} note_duration={}..{}s".format(
                args.pitch_min,
                args.pitch_max,
                args.min_note_duration,
                args.max_note_duration,
            )
        )
        if args.monitor_input:
            print("[startup] monitor_input=on; human Call notes will be forwarded to output")
        print(
            "[startup] musical_control={} same_pitch_limit={} dominant_max_share={} "
            "length_ratio={} note_ratio={}".format(
                "on" if args.musical_control else "off",
                args.same_pitch_limit,
                args.dominant_pitch_max_share,
                args.response_length_ratio,
                args.response_note_ratio,
            )
        )
        if args.speculative_preload:
            print("[startup] speculative_preload=on; candidate endpoints will pre-generate AMT responses")
        print(
            "[startup] drum_pad_range={}..{} drum_channel={} control_file={}".format(
                args.drum_pad_min,
                args.drum_pad_max,
                args.drum_channel + 1,
                args.control_file or "off",
            )
        )

    def _rhythm_status(self) -> Tuple[object, ...]:
        status = self.rhythm_guide.status()
        return (
            status.state,
            status.tap_count,
            status.bpm,
            status.confidence,
            status.bars,
            status.pattern,
            round(status.loop_seconds, 3),
            status.learning,
            status.playing,
            status.replacing,
            status.stopping,
            status.saved_slots,
            status.current_slot,
            status.queued_slot,
        )

    def _report_rhythm_status(self, force: bool = False) -> None:
        current = self._rhythm_status()
        if not force and current == self._last_rhythm_status:
            return
        self._last_rhythm_status = current
        status = self.rhythm_guide.status()
        print(
            "[rhythm] "
            + json.dumps(
                {
                    "state": status.state,
                    "tap_count": status.tap_count,
                    "bpm": status.bpm,
                    "confidence": status.confidence,
                    "bars": status.bars,
                    "steps_per_bar": status.steps_per_bar,
                    "pattern": list(status.pattern),
                    "learning": status.learning,
                    "playing": status.playing,
                    "replacing": status.replacing,
                    "stopping": status.stopping,
                    "loop_seconds": status.loop_seconds,
                    "saved_slots": list(status.saved_slots),
                    "current_slot": status.current_slot,
                    "queued_slot": status.queued_slot,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    def _loop_status(self) -> Tuple[object, ...]:
        return (
            self.loop_save_mode,
            bool(self._last_response),
            tuple(
                (
                    slot.name,
                    slot.has_content,
                    slot.event_count,
                    round(slot.loop_seconds, 3),
                    slot.playing,
                    slot.stopping,
                )
                for slot in self.loop_bank.status()
            ),
        )

    def _report_loop_status(self, force: bool = False) -> None:
        current = self._loop_status()
        if not force and current == self._last_loop_status:
            return
        self._last_loop_status = current
        slots = [
            {
                "name": slot.name,
                "has_content": slot.has_content,
                "event_count": slot.event_count,
                "loop_seconds": round(slot.loop_seconds, 3),
                "playing": slot.playing,
                "stopping": slot.stopping,
            }
            for slot in self.loop_bank.status()
        ]
        print(
            "[loop] "
            + json.dumps(
                {
                    "mode": self.loop_save_mode,
                    "latest_ready": bool(self._last_response),
                    "slots": slots,
                },
                ensure_ascii=False,
            )
        )

    def _response_loop_events(self, events: List[GeneratedEvent]) -> Tuple[List[LoopEvent], float]:
        melodic = [event for event in events if event.instrument != 128]
        if not melodic:
            return [], 0.0
        first_tick = min(event.tick for event in melodic)
        loop_events: List[LoopEvent] = []
        end_time = 0.0
        for event in melodic:
            onset = max(0.0, (event.tick - first_tick) / TIME_RESOLUTION)
            duration = min(
                max(self.args.min_note_duration, event.duration_ticks / TIME_RESOLUTION),
                self.args.max_note_duration,
            )
            pitch = min(max(event.pitch, self.args.pitch_min), self.args.pitch_max)
            velocity = clamp_int(getattr(event, "velocity", 92), 1, 127)
            loop_events.extend(
                [
                    LoopEvent(onset, "note_on", pitch, velocity, 0),
                    LoopEvent(onset + duration, "note_off", pitch, 0, 0),
                ]
            )
            end_time = max(end_time, onset + duration)
        return loop_events, max(0.5, end_time + 0.12)

    def _call_response_loop_events(
        self,
        phrase: List[CapturedNote],
        response: List[GeneratedEvent],
    ) -> Tuple[List[LoopEvent], float]:
        response_events, response_seconds = self._response_loop_events(response)
        if not phrase:
            return response_events, response_seconds
        first_onset = min(note.onset for note in phrase)
        call_events: List[LoopEvent] = []
        call_end = 0.0
        for note in phrase:
            onset = max(0.0, note.onset - first_onset)
            duration = note.duration if note.duration is not None else self.args.default_note_duration
            duration = min(max(self.args.min_note_duration, duration), self.args.max_note_duration)
            pitch = min(max(note.pitch, self.args.pitch_min), self.args.pitch_max)
            call_events.extend(
                [
                    LoopEvent(onset, "note_on", pitch, clamp_int(note.velocity, 1, 127), 0),
                    LoopEvent(onset + duration, "note_off", pitch, 0, 0),
                ]
            )
            call_end = max(call_end, onset + duration)
        response_start = call_end + 0.22
        shifted = [
            LoopEvent(event.offset + response_start, event.kind, event.note, event.velocity, 0)
            for event in response_events
        ]
        return call_events + shifted, max(0.5, response_start + response_seconds)

    def _save_last_response_to_slot(self, name: str) -> bool:
        with self._last_response_lock:
            response = list(self._last_response)
            phrase = list(self._last_call)
        if self.loop_save_mode == "call_response":
            events, seconds = self._call_response_loop_events(phrase, response)
        else:
            events, seconds = self._response_loop_events(response)
        return self.loop_bank.save(name, events, seconds)

    def _read_live_controls(self) -> List[dict]:
        path_text = self.args.control_file
        if not path_text:
            return []
        path = Path(path_text)
        if not path.exists():
            return []
        try:
            size = path.stat().st_size
            if size < self._control_offset:
                self._control_offset = 0
            with path.open("r", encoding="utf-8") as handle:
                handle.seek(self._control_offset)
                lines = handle.readlines()
                self._control_offset = handle.tell()
        except OSError as exc:
            print(f"[drum] control read failed: {exc}")
            return []

        controls: List[dict] = []
        for line in lines:
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                print("[drum] ignored an invalid live control message")
                continue
            if isinstance(payload, dict):
                controls.append(payload)
        return controls

    def _send_drum_events(self, outport, events: Iterable[DrumEvent]) -> None:
        import mido

        if outport is None:
            return
        with self.output_lock:
            for event in events:
                outport.send(
                    mido.Message(
                        event.kind,
                        note=event.note,
                        velocity=event.velocity,
                        channel=self.args.drum_channel,
                    )
                )
                print(
                    f"[drum_out] {event.kind} pitch={event.note:3d} velocity={event.velocity:3d}",
                    flush=True,
                )

    def _send_loop_events(self, outport, events: Iterable[LoopEvent]) -> None:
        import mido

        with self.output_lock:
            for event in events:
                outport.send(
                    mido.Message(
                        event.kind,
                        note=event.note,
                        velocity=event.velocity,
                        channel=event.channel,
                    )
                )
                print(
                    f"[loop_out] {event.kind} pitch={event.note:3d} velocity={event.velocity:3d}",
                    flush=True,
                )

    def _forward_drum_message(self, outport, msg) -> None:
        import mido

        with self.output_lock:
            outport.send(
                msg.copy(
                    time=0,
                    channel=self.args.drum_channel,
                    velocity=0 if msg.type == "note_off" else int(msg.velocity),
                )
            )

    def _panic_outputs(self, melody_output, drum_output) -> None:
        import mido

        ports = [melody_output]
        if drum_output is not melody_output:
            ports.append(drum_output)
        with self.output_lock:
            for outport in ports:
                if outport is None:
                    continue
                for channel in range(16):
                    outport.send(mido.Message("control_change", channel=channel, control=64, value=0))
                    outport.send(mido.Message("control_change", channel=channel, control=123, value=0))
                    outport.send(mido.Message("control_change", channel=channel, control=120, value=0))
        self._monitor_active_notes.clear()
        print("[panic] sustain_off all_notes_off all_sound_off", flush=True)

    def _apply_drum_controls(self, now: float, outport) -> None:
        for control in self._read_live_controls():
            action = control.get("action")
            if action in {"rhythm_learn_start", "drum_record_start"}:
                self.rhythm_guide.start_learning(now)
                self._report_rhythm_status(force=True)
            elif action in {"rhythm_learn_finish", "drum_record_finish"}:
                if not self.rhythm_guide.finish_learning(now):
                    print("[rhythm] at least three taps are required before finishing")
                self._report_rhythm_status(force=True)
            elif action in {"rhythm_stop", "drum_stop"}:
                self.rhythm_guide.request_stop()
                self._report_rhythm_status(force=True)
            elif action in {"rhythm_stop_now", "drum_stop_now"}:
                self._send_drum_events(outport, self.rhythm_guide.emergency_stop())
                self._report_rhythm_status(force=True)
            elif action == "rhythm_tap":
                tap_time = float(control.get("timestamp", now))
                velocity = int(control.get("velocity", 96))
                if self.rhythm_guide.capture_tap(velocity, tap_time):
                    print(f"[rhythm_tap] velocity={velocity:3d}", flush=True)
                    self._report_rhythm_status(force=True)
            elif isinstance(action, str) and action.startswith("rhythm_save_"):
                slot = action.rsplit("_", 1)[-1].upper()
                try:
                    if not self.rhythm_guide.save_slot(slot):
                        print("[rhythm] no active rhythm available to save")
                except ValueError:
                    print(f"[rhythm] ignored unknown save slot: {slot}")
                self._report_rhythm_status(force=True)
            elif isinstance(action, str) and action.startswith("rhythm_load_"):
                slot = action.rsplit("_", 1)[-1].upper()
                try:
                    if not self.rhythm_guide.load_slot(slot, now):
                        print(f"[rhythm] slot {slot} is empty")
                except ValueError:
                    print(f"[rhythm] ignored unknown load slot: {slot}")
                self._report_rhythm_status(force=True)
            elif action == "rhythm_variation":
                if not self.rhythm_guide.create_variation(now):
                    print("[rhythm] no active rhythm available to vary")
                self._report_rhythm_status(force=True)
            elif action in {"loop_set_mode_response", "loop_set_mode_call_response"}:
                self.loop_save_mode = "call_response" if action.endswith("call_response") else "response"
                self._report_loop_status(force=True)
            elif isinstance(action, str) and action.startswith("loop_save_"):
                slot = action.rsplit("_", 1)[-1].upper()
                try:
                    if self._save_last_response_to_slot(slot):
                        print(f"[loop] saved slot={slot} mode={self.loop_save_mode}")
                    else:
                        print("[loop] no completed AI response available to save yet")
                except ValueError:
                    print(f"[loop] ignored unknown save slot: {slot}")
                self._report_loop_status(force=True)
            elif isinstance(action, str) and action.startswith("loop_toggle_"):
                slot = action.rsplit("_", 1)[-1].upper()
                try:
                    if not self.loop_bank.toggle(slot, now):
                        print(f"[loop] slot {slot} is empty; save a response first")
                except ValueError:
                    print(f"[loop] ignored unknown toggle slot: {slot}")
                self._report_loop_status(force=True)
            elif isinstance(action, str) and action.startswith("loop_stop_") and action != "loop_stop_all":
                slot = action.rsplit("_", 1)[-1].upper()
                try:
                    self.loop_bank.request_stop(slot)
                except ValueError:
                    print(f"[loop] ignored unknown stop slot: {slot}")
                self._report_loop_status(force=True)
            elif action == "loop_stop_all":
                self._send_loop_events(outport, self.loop_bank.stop_all_now())
                self._report_loop_status(force=True)
            elif action == "panic_all":
                self.playback_cancel_event.set()
                self._send_loop_events(self._melody_output, self.loop_bank.stop_all_now())
                self._send_drum_events(outport, self.rhythm_guide.emergency_stop())
                self._panic_outputs(self._melody_output, outport)
                self._report_rhythm_status(force=True)
                self._report_loop_status(force=True)

    def _can_speculatively_preload(self) -> bool:
        return (
            self.args.speculative_preload
            and self.args.backend == "amt"
            and self.args.response_strategy == "controlled_amt"
            and not self.busy.is_set()
        )

    def _generate_amt_events_for_phrase(
        self,
        phrase: List[CapturedNote],
        stop_event: threading.Event,
    ) -> Tuple[List[GeneratedEvent], MusicalControlStats, Optional[float]]:
        profile = analyze_call_phrase(phrase)
        prompt_phrase = clean_prompt_phrase(phrase, profile) if self.args.musical_control else phrase
        plan = build_response_plan(
            profile=profile,
            response_seconds=self.args.response_seconds,
            max_events=self.args.max_events,
            pitch_min=self.args.pitch_min,
            pitch_max=self.args.pitch_max,
            response_length_ratio=self.args.response_length_ratio,
            response_note_ratio=self.args.response_note_ratio,
            same_pitch_limit=self.args.same_pitch_limit,
            dominant_pitch_max_share=self.args.dominant_pitch_max_share,
            resample_attempts=self.args.resample_attempts,
            duration_match_min_share=self.args.duration_match_min_share,
            stop_on_target_notes=self.args.live_stop_on_target_notes,
        )
        controller = MusicalResponseController(profile, plan) if self.args.musical_control else None
        call_events = notes_to_amt_events(prompt_phrase)
        if self.args.latency_mode == "fast":
            events = self._generate_budgeted_amt_events(
                call_events,
                prompt_phrase,
                profile,
                plan,
                controller,
                plan.target_seconds if self.args.musical_control else self.args.response_seconds,
            )
            return events, controller.stats if controller is not None else MusicalControlStats(), (
                time.monotonic() if events else None
            )

        events: List[GeneratedEvent] = []
        first_event_time: Optional[float] = None
        for event in self.generator.generate_events(
            call_events,
            response_seconds=plan.target_seconds if self.args.musical_control else self.args.response_seconds,
            stop_event=stop_event,
            controller=controller,
        ):
            if first_event_time is None:
                first_event_time = time.monotonic()
            events.append(event)
            if len(events) >= self.args.max_events:
                break
        stats = controller.stats if controller is not None else MusicalControlStats()
        return events, stats, first_event_time

    def _generate_budgeted_amt_events(
        self,
        call_events: List[int],
        prompt_phrase: List[CapturedNote],
        profile: CallProfile,
        plan: ResponsePlan,
        controller: Optional[MusicalResponseController],
        response_seconds: float,
    ) -> List[GeneratedEvent]:
        """Keep live AMT inference inside a fixed latency budget.

        AMT event sampling is slower than the musical spacing between events on
        Apple Silicon. Collecting the whole response therefore makes the player
        miss its first-event deadline. We sample as much neural material as the
        live budget permits, then place those pitches onto the call-derived
        rhythmic template so playback receives a complete buffer at once.
        """

        model_queue: "queue.Queue[Optional[GeneratedEvent]]" = queue.Queue()
        local_stop = threading.Event()
        target_count = min(self.args.max_events, plan.target_notes)

        def producer() -> None:
            try:
                count = 0
                for event in self.generator.generate_events(
                    call_events,
                    response_seconds=response_seconds,
                    stop_event=local_stop,
                    controller=None,
                ):
                    model_queue.put(event)
                    count += 1
                    if count >= target_count:
                        break
            except Exception as exc:
                print(f"[amt-live] model sampling failed: {exc!r}")
            finally:
                model_queue.put(None)

        threading.Thread(target=producer, daemon=True).start()
        deadline = time.monotonic() + self.args.amt_generation_budget
        model_events: List[GeneratedEvent] = []
        producer_done = False

        while len(model_events) < target_count and not producer_done:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                item = model_queue.get(timeout=min(0.05, remaining))
            except queue.Empty:
                continue
            if item is None:
                producer_done = True
            else:
                model_events.append(item)

        local_stop.set()
        while len(model_events) < target_count:
            try:
                item = model_queue.get_nowait()
            except queue.Empty:
                break
            if item is None:
                break
            model_events.append(item)

        if not model_events:
            print(
                f"[amt-live] budget={self.args.amt_generation_budget:.2f}s "
                "model_events=0; using immediate musical fallback"
            )
            return []

        template = build_rescue_response_events(
            prompt_phrase,
            profile,
            plan,
            controller=None,
            default_note_duration=self.args.default_note_duration,
            min_note_duration=self.args.min_note_duration,
            max_note_duration=self.args.max_note_duration,
        )
        if not template:
            return model_events

        blended: List[GeneratedEvent] = []
        for index, template_event in enumerate(template[:target_count]):
            if index < len(model_events):
                model_event = model_events[index]
                melodic_deviation = clamp_int(
                    model_event.pitch - template_event.pitch,
                    -2,
                    2,
                )
                event = GeneratedEvent(
                    tick=template_event.tick,
                    duration_ticks=template_event.duration_ticks,
                    pitch=template_event.pitch + melodic_deviation,
                    instrument=model_event.instrument,
                    velocity=clamp_int(
                        round(template_event.velocity * 0.72 + model_event.velocity * 0.28),
                        52,
                        108,
                    ),
                )
            else:
                event = template_event
                if controller is not None:
                    controller.stats.fallback_count += 1

            if controller is not None:
                reason = controller.rejection_reason(event)
                if reason is not None:
                    controller.reject(reason)
                    event = controller.fallback_event(event)
                event = controller.accept(event)
            blended.append(event)

        print(
            f"[amt-live] budget={self.args.amt_generation_budget:.2f}s "
            f"model_events={len(model_events)} completed_events={len(blended)} "
            f"key={profile.scale_root}:{profile.scale_mode}"
        )
        return blended

    def _stream_budgeted_amt_events(
        self,
        call_events: List[int],
        response_seconds: float,
        target_count: int,
    ) -> Iterator[GeneratedEvent]:
        """Yield AMT events as they arrive while enforcing a live wall-time budget."""

        model_queue: "queue.Queue[Optional[GeneratedEvent]]" = queue.Queue()
        local_stop = threading.Event()

        def producer() -> None:
            try:
                for event in self.generator.generate_events(
                    call_events,
                    response_seconds=response_seconds,
                    stop_event=local_stop,
                    controller=None,
                ):
                    model_queue.put(event)
                    if local_stop.is_set():
                        break
            except Exception as exc:
                print(f"[amt-stream] model sampling failed: {exc!r}")
            finally:
                model_queue.put(None)

        threading.Thread(target=producer, daemon=True).start()
        deadline = time.monotonic() + self.args.amt_generation_budget
        emitted = 0
        try:
            while emitted < target_count and not self.stop_event.is_set():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    print(
                        f"[amt-stream] budget={self.args.amt_generation_budget:.2f}s "
                        f"model_events={emitted}; closing neural stream"
                    )
                    break
                try:
                    item = model_queue.get(timeout=min(0.05, remaining))
                except queue.Empty:
                    continue
                if item is None:
                    break
                emitted += 1
                yield item
        finally:
            local_stop.set()

    def on_candidate_endpoint(self, candidate: EndpointCandidate) -> None:
        if not self._can_speculatively_preload():
            return

        phrase = self.recorder.snapshot_phrase(candidate.candidate_time)
        if not phrase:
            return

        with self.speculative_lock:
            if self.speculative_preload is not None and not self.speculative_preload.done_event.is_set():
                self.speculative_preload.cancelled = True
                self.speculative_preload.stop_event.set()
            self.speculative_preload_id += 1
            preload = SpeculativePreloadState(
                preload_id=self.speculative_preload_id,
                candidate_time=candidate.candidate_time,
                phrase_note_count=len(phrase),
                started_at=time.monotonic(),
                stop_event=threading.Event(),
                done_event=threading.Event(),
                events=[],
                stats=MusicalControlStats(),
            )
            self.speculative_preload = preload

        print(
            f"[candidate] speculative_start id={preload.preload_id} "
            f"notes={len(phrase)} clusters={candidate.onset_cluster_count} "
            f"cutoff={candidate.tau_cutoff:.3f}s"
        )

        def worker() -> None:
            try:
                events, stats, first_event_time = self._generate_amt_events_for_phrase(
                    phrase,
                    preload.stop_event,
                )
                preload.events = events
                preload.stats = stats
                preload.first_event_time = first_event_time
            except Exception as exc:
                preload.error = repr(exc)
            finally:
                preload.ended_at = time.monotonic()
                preload.done_event.set()
                print(
                    f"[candidate] speculative_done id={preload.preload_id} "
                    f"events={len(preload.events)} cancelled={preload.cancelled} "
                    f"error={preload.error or 'none'}"
                )

        threading.Thread(target=worker, daemon=True).start()

    def on_candidate_cancel(self, cancel: EndpointCancel) -> None:
        with self.speculative_lock:
            preload = self.speculative_preload
            if preload is None:
                return
            if abs(preload.candidate_time - cancel.candidate_time) > 0.001:
                return
            preload.cancelled = True
            preload.stop_event.set()
            self.speculative_cancel_count += 1
        print(
            f"[candidate] cancel id={preload.preload_id} reason={cancel.reason} "
            f"new_pitch={cancel.new_event.pitch} phrase_len={len(cancel.phrase)}"
        )

    def _take_matching_preload(
        self,
        decision: EndpointDecision,
        phrase: List[CapturedNote],
    ) -> Optional[SpeculativePreloadState]:
        if decision.candidate_time is None:
            return None
        with self.speculative_lock:
            preload = self.speculative_preload
            if preload is None:
                return None
            if preload.cancelled:
                self.speculative_preload = None
                return None
            if abs(preload.candidate_time - decision.candidate_time) > 0.001:
                preload.cancelled = True
                preload.stop_event.set()
                self.speculative_preload = None
                return None
            if preload.phrase_note_count != len(phrase):
                preload.cancelled = True
                preload.stop_event.set()
                self.speculative_preload = None
                return None
            self.speculative_preload = None
            return preload

    def on_endpoint(self, decision: EndpointDecision) -> None:
        phrase = self.recorder.consume_phrase(decision.cut_time)
        if not phrase:
            print("[endpoint] no completed notes captured; ignoring")
            return

        if self.busy.is_set():
            # Do not start overlapping generated phrases: that creates a
            # muddy, uncontrollable result in Logic.  The latest complete
            # performer phrase replaces an older queued phrase and begins as
            # soon as the current response has finished.
            with self.pending_lock:
                replaced = self.pending_response is not None
                self.pending_response = (phrase, decision)
            print(
                f"[endpoint] queued phrase_len={len(phrase)} while AI response is playing "
                f"replaced={'yes' if replaced else 'no'}"
            )
            return

        preload = self._take_matching_preload(decision, phrase)

        self._start_response_cycle(phrase, decision, preload)

    def _next_round_id(self) -> int:
        with self.round_lock:
            self.round_id += 1
            return self.round_id

    def _start_response_cycle(
        self,
        phrase: List[CapturedNote],
        decision: EndpointDecision,
        preload: Optional[SpeculativePreloadState] = None,
        *,
        queued: bool = False,
    ) -> None:
        round_id = self._next_round_id()

        print(
            "[endpoint]{} phrase_len={} mu={:.3f}/s cutoff={:.3f}s silence={:.3f}s "
            "confirm={:.3f}s preload={}".format(
                " queued->" if queued else "",
                len(phrase),
                decision.mu_tempo,
                decision.tau_cutoff,
                decision.silence,
                decision.confirmation_delay,
                "yes" if preload is not None else "no",
            )
        )
        print(f"[round {round_id}] endpoint -> generating")
        thread = threading.Thread(
            target=self.run_response_cycle,
            args=(round_id, phrase, decision, preload),
            daemon=True,
        )
        thread.start()

    def _start_pending_response_or_mark_idle(self) -> None:
        """Atomically hand the response lane to the newest queued phrase."""

        with self.pending_lock:
            pending = self.pending_response
            self.pending_response = None
            if pending is None:
                self.busy.clear()
                return

        # Keep busy set during the handoff.  A third phrase arriving in this
        # tiny interval is queued behind this one instead of racing into an
        # overlapping response thread.
        phrase, decision = pending
        print(f"[endpoint] starting queued phrase_len={len(phrase)}")
        self._start_response_cycle(phrase, decision, queued=True)

    def run_response_cycle(
        self,
        round_id: int,
        phrase: List[CapturedNote],
        decision: EndpointDecision,
        preload: Optional[SpeculativePreloadState] = None,
    ) -> None:
        import mido

        self.busy.set()
        self.playback_cancel_event.clear()
        events_queue: "queue.Queue[Optional[GeneratedEvent]]" = queue.Queue()
        profile = analyze_call_phrase(phrase)
        prompt_phrase = clean_prompt_phrase(phrase, profile) if self.args.musical_control else phrase
        plan = build_response_plan(
            profile=profile,
            response_seconds=self.args.response_seconds,
            max_events=self.args.max_events,
            pitch_min=self.args.pitch_min,
            pitch_max=self.args.pitch_max,
            response_length_ratio=self.args.response_length_ratio,
            response_note_ratio=self.args.response_note_ratio,
            same_pitch_limit=self.args.same_pitch_limit,
            dominant_pitch_max_share=self.args.dominant_pitch_max_share,
            resample_attempts=self.args.resample_attempts,
            duration_match_min_share=self.args.duration_match_min_share,
            stop_on_target_notes=self.args.live_stop_on_target_notes,
        )
        controller = MusicalResponseController(profile, plan) if self.args.musical_control else None
        call_events = notes_to_amt_events(prompt_phrase) if self.args.backend == "amt" else []
        call_start_time = decision.phrase[0].timestamp if decision.phrase else None
        generation_start_time = time.monotonic()
        metrics = RoundMetrics(
            round_id=round_id,
            call_notes=len(phrase),
            model_id=self.args.aria_model_id if self.args.backend == "aria" else self.args.model_id,
            device=self.device,
            call_start_time=call_start_time,
            endpoint_time=decision.cut_time,
            generation_start_time=generation_start_time,
            endpoint_silence=decision.silence,
            endpoint_cutoff=decision.tau_cutoff,
            mu_tempo=decision.mu_tempo,
            call_duration_seconds=profile.duration_seconds,
            dominant_pitch=profile.dominant_pitch,
            dominant_pitch_share=profile.dominant_pitch_share,
            tail_repeat_pitch=profile.tail_repeat_pitch,
            tail_repeat_count=profile.tail_repeat_count,
            target_response_seconds=plan.target_seconds,
            target_response_notes=plan.target_notes,
            candidate_time=decision.candidate_time,
            candidate_confirm_delay=decision.confirmation_delay,
            speculative_cancel_count=self.speculative_cancel_count,
        )

        print(
            f"[round {round_id}] [analysis] call_duration={profile.duration_seconds:.3f}s "
            f"call_notes={profile.note_count} density={profile.note_density:.2f}/s "
            f"contour={profile.contour}"
        )
        print(
            f"[round {round_id}] [analysis] dominant_pitch={profile.dominant_pitch} "
            f"share={profile.dominant_pitch_share:.2f} "
            f"tail_repeat_pitch={profile.tail_repeat_pitch} "
            f"tail_repeat_count={profile.tail_repeat_count}"
        )
        if self.args.musical_control:
            print(
                f"[round {round_id}] [plan] target_seconds={plan.target_seconds:.3f}s "
                f"target_notes={plan.target_notes} pitch_range={plan.pitch_min}..{plan.pitch_max} "
                f"prompt_notes={len(prompt_phrase)}/{len(phrase)}"
            )
        else:
            print(f"[round {round_id}] [plan] musical_control=off")
        streaming_live = (
            self.args.backend == "amt"
            and self.args.response_strategy == "streaming_amt"
        )
        generator_label = (
            "PUBLISHED MOTIF"
            if self.args.response_strategy == "motif"
            else "STREAMING AMT"
            if streaming_live
            else self.args.backend.upper()
        )
        print(f"[round {round_id}] [generating] {generator_label} response thread started")
        t0 = time.perf_counter()
        inference_done = threading.Event()

        def inference_worker() -> None:
            first_latency_reported = False
            count = 0
            amt_generated_count = 0
            previous_live_pitch: Optional[int] = None
            generated_events: List[GeneratedEvent] = []
            # If speculative AMT has already consumed its full budget and
            # produced no events, retrying exactly the same request here used
            # to add another full budget before the legitimate empty-output
            # fallback could play.  Keep that fact so this round goes straight
            # to the existing Controlled-AMT fallback instead.
            empty_preload_exhausted = False
            try:
                response_seconds = plan.target_seconds if self.args.musical_control else self.args.response_seconds
                event_iter = None
                if preload is not None:
                    print(f"[round {round_id}] [preload] waiting for speculative id={preload.preload_id}")
                    preload.done_event.wait()
                    metrics.preload_latency = (
                        None
                        if preload.ended_at is None
                        else max(0.0, preload.ended_at - preload.started_at)
                    )
                    if preload.events and not preload.cancelled and not preload.error:
                        metrics.speculative_preload_used = True
                        metrics.first_event_time = preload.first_event_time
                        metrics.rejected_repeat_count = preload.stats.rejected_repeat_count
                        metrics.rejected_dominant_count = preload.stats.rejected_dominant_count
                        metrics.fallback_count = preload.stats.fallback_count
                        event_iter = iter(preload.events)
                        print(
                            f"[round {round_id}] [preload] using id={preload.preload_id} "
                            f"events={len(preload.events)} latency={metrics.preload_latency:.3f}s"
                        )
                    else:
                        print(
                            f"[round {round_id}] [preload] fallback id={preload.preload_id} "
                            f"events={len(preload.events)} cancelled={preload.cancelled} "
                            f"error={preload.error or 'none'}"
                        )
                        empty_preload_exhausted = (
                            not preload.cancelled
                            and not preload.error
                            and not preload.events
                        )

                if event_iter is None:
                    if empty_preload_exhausted:
                        # AMT has already been given its configured generation
                        # window for this exact phrase.  Leave event_iter empty
                        # so the normal, controlled empty-AMT rescue below can
                        # respond immediately rather than running a duplicate
                        # inference pass.
                        event_iter = iter(())
                        print(
                            f"[round {round_id}] [preload] AMT empty after budget; "
                            "using controlled fallback without duplicate inference"
                        )
                    elif self.args.response_strategy == "motif":
                        motif_events = build_rescue_response_events(
                            prompt_phrase,
                            profile,
                            plan,
                            controller,
                            default_note_duration=self.args.default_note_duration,
                            min_note_duration=self.args.min_note_duration,
                            max_note_duration=self.args.max_note_duration,
                        )
                        event_iter = iter(motif_events)
                        print(
                            f"[round {round_id}] [strategy] published live motif "
                            f"events={len(motif_events)}"
                        )
                    elif self.args.backend == "aria":
                        event_iter = self.generator.generate_events(
                            prompt_phrase,
                            response_seconds=response_seconds,
                            stop_event=self.stop_event,
                            controller=controller,
                            live_mode=self.args.aria_live_mode,
                        )
                    elif streaming_live:
                        event_iter = self._stream_budgeted_amt_events(
                            call_events,
                            response_seconds=response_seconds,
                            target_count=min(self.args.max_events, plan.target_notes),
                        )
                    elif self.args.latency_mode == "fast":
                        # Sample the paper AMT inside a fixed real-time budget,
                        # then let the existing phrase controller complete and
                        # time-shape that AMT material for playback.
                        event_iter = iter(
                            self._generate_budgeted_amt_events(
                                call_events,
                                prompt_phrase,
                                profile,
                                plan,
                                controller,
                                response_seconds,
                            )
                        )
                    else:
                        event_iter = self.generator.generate_events(
                            call_events,
                            response_seconds=response_seconds,
                            stop_event=self.stop_event,
                            controller=controller,
                        )
                for event in event_iter:
                    if streaming_live:
                        if controller is not None:
                            reason = controller.rejection_reason(event)
                            if reason is not None:
                                controller.reject(reason)
                                event = controller.fallback_event(event)
                        if self.args.live_style == "pentatonic":
                            event = project_live_event_to_call_scale(
                                event,
                                profile,
                                plan,
                                previous_live_pitch,
                            )
                            previous_live_pitch = event.pitch
                        if controller is not None:
                            event = controller.accept(event)
                    if not first_latency_reported:
                        if metrics.first_event_time is None:
                            metrics.first_event_time = time.monotonic()
                        print(
                            f"[round {round_id}] [generating] "
                            f"first_event_latency={time.perf_counter() - t0:.3f}s"
                        )
                        first_latency_reported = True
                    if self.args.duration_match and not streaming_live:
                        generated_events.append(event)
                    else:
                        events_queue.put(event)
                        generated_events.append(event)
                    count += 1
                    print(
                        f"[round {round_id}] [generating] sampled event #{count}: "
                        f"tick={event.tick} dur={event.duration_ticks} pitch={event.pitch}"
                    )
                    if count >= self.args.max_events:
                        break
                amt_generated_count = count if streaming_live else 0
                if should_use_motif_fallback(
                    generated_count=count,
                    fallback_on_empty=self.args.fallback_on_empty,
                    backend=self.args.backend,
                    response_strategy=self.args.response_strategy,
                    stopped=self.stop_event.is_set(),
                ):
                    rescue_events = build_rescue_response_events(
                        prompt_phrase,
                        profile,
                        plan,
                        controller,
                        default_note_duration=self.args.default_note_duration,
                        min_note_duration=self.args.min_note_duration,
                        max_note_duration=self.args.max_note_duration,
                    )
                    if rescue_events:
                        print(
                            f"[round {round_id}] [rescue] AMT produced no playable notes; "
                            f"using motif fallback events={len(rescue_events)}"
                        )
                    for event in rescue_events:
                        if streaming_live and self.args.live_style == "pentatonic":
                            event = project_live_event_to_call_scale(
                                event,
                                profile,
                                plan,
                                previous_live_pitch,
                            )
                            previous_live_pitch = event.pitch
                        if not first_latency_reported:
                            if metrics.first_event_time is None:
                                metrics.first_event_time = time.monotonic()
                            print(
                                f"[round {round_id}] [generating] "
                                f"first_event_latency={time.perf_counter() - t0:.3f}s"
                            )
                            first_latency_reported = True
                        if self.args.duration_match and not streaming_live:
                            generated_events.append(event)
                        else:
                            events_queue.put(event)
                            generated_events.append(event)
                        count += 1
                        print(
                            f"[round {round_id}] [generating] sampled rescue event #{count}: "
                            f"tick={event.tick} dur={event.duration_ticks} pitch={event.pitch}"
                        )
                        if count >= self.args.max_events:
                            break

                if streaming_live and amt_generated_count > 0 and count < self.args.max_events:
                    target_event_count = min(plan.target_notes, self.args.max_events)
                    fill_count = partial_fallback_limit(
                        amt_generated_count,
                        target_event_count,
                        self.args.partial_fallback_max_share,
                    )
                    template_events = build_rescue_response_events(
                        prompt_phrase,
                        profile,
                        plan,
                        controller=None,
                        default_note_duration=self.args.default_note_duration,
                        min_note_duration=self.args.min_note_duration,
                        max_note_duration=self.args.max_note_duration,
                    )
                    completion_events = build_partial_motif_completion(
                        generated_events,
                        template_events,
                        fill_count,
                        grid_ticks=max(1, round(estimate_phrase_grid(prompt_phrase) * TIME_RESOLUTION)),
                        controller=controller,
                    )
                    for event in completion_events:
                        if self.args.live_style == "pentatonic":
                            event = project_live_event_to_call_scale(
                                event,
                                profile,
                                plan,
                                previous_live_pitch,
                            )
                            previous_live_pitch = event.pitch
                        events_queue.put(event)
                        generated_events.append(event)
                        count += 1
                        print(
                            f"[round {round_id}] [partial_fallback] event #{count}: "
                            f"tick={event.tick} dur={event.duration_ticks} pitch={event.pitch}"
                        )
                    if completion_events:
                        print(
                            f"[round {round_id}] [partial_fallback] "
                            f"amt={amt_generated_count} motif={len(completion_events)} "
                            f"target={target_event_count}"
                        )

                if generated_events and self.args.duration_match and not streaming_live:
                    shaped_events, raw_duration, actual_duration, stretch_factor = shape_response_duration(
                        generated_events,
                        plan,
                        min_note_duration=self.args.min_note_duration,
                        max_note_duration=self.args.max_note_duration,
                        min_share=self.args.duration_match_min_share,
                        max_share=self.args.duration_match_max_share,
                    )
                    generated_events = shaped_events
                    if self.args.live_style == "pentatonic":
                        generated_events = apply_live_pentatonic_style(
                            generated_events,
                            profile,
                            plan,
                        )
                        print(
                            f"[round {round_id}] [style] pentatonic root={profile.scale_root} "
                            f"mode={profile.scale_mode} events={len(generated_events)}"
                        )
                    count = len(generated_events)
                    metrics.raw_response_seconds = raw_duration
                    metrics.actual_response_seconds = actual_duration
                    metrics.duration_stretch_factor = stretch_factor
                    metrics.duration_match_ratio = (
                        actual_duration / plan.target_seconds
                        if plan.target_seconds and plan.target_seconds > 0
                        else None
                    )
                    print(
                        f"[round {round_id}] [duration] raw={raw_duration:.3f}s "
                        f"actual={actual_duration:.3f}s target={plan.target_seconds:.3f}s "
                        f"ratio={metrics.duration_match_ratio or 0.0:.2f} "
                        f"stretch={stretch_factor:.2f}"
                    )
                    for index, event in enumerate(generated_events, start=1):
                        events_queue.put(event)
                        print(
                            f"[round {round_id}] [generating] queued event #{index}: "
                            f"tick={event.tick} dur={event.duration_ticks} pitch={event.pitch}"
                        )
                elif generated_events:
                    raw_duration = response_duration_seconds(
                        generated_events,
                        self.args.min_note_duration,
                        self.args.max_note_duration,
                    )
                    metrics.raw_response_seconds = raw_duration
                    metrics.actual_response_seconds = raw_duration
                    metrics.duration_stretch_factor = 1.0
                    metrics.duration_match_ratio = (
                        raw_duration / plan.target_seconds
                        if plan.target_seconds and plan.target_seconds > 0
                        else None
                    )
            finally:
                if generated_events:
                    with self._last_response_lock:
                        self._last_response = list(generated_events)
                        self._last_call = list(phrase)
                    print(f"[loop] latest response ready events={len(generated_events)}")
                    self._report_loop_status(force=True)
                metrics.generated_events = count
                if controller is not None and not metrics.speculative_preload_used:
                    metrics.rejected_repeat_count = controller.stats.rejected_repeat_count
                    metrics.rejected_dominant_count = controller.stats.rejected_dominant_count
                    metrics.fallback_count = controller.stats.fallback_count
                events_queue.put(None)
                inference_done.set()
                print(
                    f"[round {round_id}] [generating] done; events={count} "
                    f"rejected_repeat={metrics.rejected_repeat_count} "
                    f"rejected_dominant={metrics.rejected_dominant_count} "
                    f"fallback={metrics.fallback_count}"
                )

        inference_thread = threading.Thread(target=inference_worker, daemon=True)
        inference_thread.start()

        try:
            if self._melody_output is None:
                raise RuntimeError("Logic melody MIDI output is not available")
            initial_buffer_events = self.args.initial_buffer_events
            initial_buffer_timeout = self.args.initial_buffer_timeout
            if self.args.backend == "aria" and self.args.aria_live_mode == "batch":
                initial_buffer_events = 1
                initial_buffer_timeout = max(initial_buffer_timeout, self.args.aria_batch_timeout)
            player = ResponsePlayer(
                LockedOutputPort(self._melody_output, self.output_lock),
                events_queue,
                initial_buffer_events=initial_buffer_events,
                initial_buffer_timeout=initial_buffer_timeout,
                first_event_timeout=self.args.first_event_timeout,
                max_underrun_seconds=self.args.max_underrun_seconds,
                producer_done=inference_done,
                pitch_min=self.args.pitch_min,
                pitch_max=self.args.pitch_max,
                min_note_duration=self.args.min_note_duration,
                max_note_duration=self.args.max_note_duration,
                rhythm_timing_provider=self.rhythm_guide.timing,
                cancel_event=self.playback_cancel_event,
            )
            print(f"[round {round_id}] [buffering] waiting for initial response buffer")
            buffer_start = time.perf_counter()
            playback_stats = player.play()
            metrics.playback_start_time = playback_stats.playback_start_time
            metrics.playback_end_time = playback_stats.playback_end_time
            metrics.initial_buffer_wait = playback_stats.initial_buffer_wait
            metrics.initial_buffer_count = playback_stats.initial_buffer_count
            metrics.played_events = playback_stats.played_events
            metrics.buffer_underruns = playback_stats.buffer_underruns
            print(
                f"[round {round_id}] [buffering] "
                f"total_response_cycle={time.perf_counter() - buffer_start:.3f}s"
            )
        except Exception as exc:
            metrics.status = "error"
            metrics.error = repr(exc)
            print(f"[round {round_id}] [error] {exc}")
        finally:
            inference_thread.join(timeout=1.0)
            self.metrics.record(metrics)
            self._start_pending_response_or_mark_idle()

    def run(self) -> None:
        import mido

        input_port = resolve_port(self.args.input_port, mido.get_input_names(), "input")
        output_names = mido.get_output_names()
        drum_output_port = resolve_port(self.args.output_port, output_names, "drum output")
        melody_output_port = resolve_port(self.args.melody_output_port, output_names, "melody output")
        if melody_output_port is None:
            melody_output_port = drum_output_port
        print(f"[ports] input={input_port}")
        print(f"[ports] melody_output={melody_output_port}")
        print(f"[ports] drum_output={drum_output_port}")
        if self.args.startup_test_note:
            send_test_output(melody_output_port)
        print("[listening] play a MIDI Call on the selected input, then stop")

        start = time.monotonic()
        from contextlib import ExitStack

        with mido.open_input(input_port) as inport, ExitStack() as stack:
            melody_output = stack.enter_context(mido.open_output(melody_output_port))
            self._melody_output = melody_output
            drum_output = (
                melody_output
                if drum_output_port == melody_output_port
                else stack.enter_context(mido.open_output(drum_output_port))
            )
            monitor_out = melody_output if self.args.monitor_input else None
            if monitor_out is not None:
                print("[monitor] forwarding human input to output")
            self._report_rhythm_status(force=True)
            self._report_loop_status(force=True)
            try:
                while not self.stop_event.is_set():
                    now = time.monotonic()
                    self._apply_drum_controls(now, drum_output)
                    if self.rhythm_guide.maybe_finish_learning(now):
                        print("[rhythm] silence detected; learned pattern is now active", flush=True)
                        self._report_rhythm_status(force=True)
                    self._send_drum_events(drum_output, self.rhythm_guide.tick(now))
                    self._send_loop_events(melody_output, self.loop_bank.tick(now))
                    for msg in inport.iter_pending():
                        now = time.monotonic()
                        # MIDI note_on with velocity 0 is a note_off. Normalize it
                        # before any busy-state filtering so no note can hang.
                        if msg.type == "note_on" and msg.velocity == 0:
                            msg = msg.copy(type="note_off", velocity=0)
                        if should_capture_rhythm_pad(
                            msg,
                            self.rhythm_guide.learning,
                            self.args.drum_pad_min,
                            self.args.drum_pad_max,
                            self.args.drum_channel,
                        ):
                            if self.rhythm_guide.capture(msg, now):
                                print(
                                    f"[rhythm_tap] velocity={int(getattr(msg, 'velocity', 0)):3d}",
                                    flush=True,
                                )
                                self._report_rhythm_status(force=True)
                            continue
                        if is_drum_pad_note(
                            msg,
                            self.args.drum_pad_min,
                            self.args.drum_pad_max,
                            self.args.drum_channel,
                        ):
                            print(
                                f"[pad] {msg.type} pitch={int(msg.note):3d} "
                                f"velocity={int(getattr(msg, 'velocity', 0)):3d} "
                                f"channel={int(getattr(msg, 'channel', 0)) + 1}"
                            )
                            self._forward_drum_message(drum_output, msg)
                            continue
                        if msg.type == "note_on" and msg.velocity > 0:
                            if monitor_out is not None:
                                self.forward_input_message(monitor_out, msg)
                            # Continue recording during an AI answer.  Endpoint
                            # handling queues the newest completed phrase, so
                            # the performer can play continuously without
                            # creating overlapping generated responses.
                            self.recorder.note_on(msg.note, msg.velocity, getattr(msg, "channel", 0), now)
                            self.vad.observe_note_on(msg.note, msg.velocity, now)
                            print(
                                f"[note_on] pitch={msg.note:3d} velocity={msg.velocity:3d} "
                                f"mu={self.vad.mu_tempo():.3f}/s cutoff={self.vad.tau_cutoff():.3f}s"
                            )
                        elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                            if monitor_out is not None:
                                self.forward_input_message(monitor_out, msg)
                                self._monitor_active_notes.discard(
                                    (getattr(msg, "note", 0), getattr(msg, "channel", 0))
                                )
                            # Releases must be recorded during AI playback as
                            # well; otherwise the queued phrase can contain
                            # stuck notes or never reach its endpoint.
                            self.recorder.note_off(msg.note, getattr(msg, "channel", 0), now)
                            print(f"[note_off] pitch={msg.note:3d} velocity=  0")

                    self.vad.tick()
                    self._report_rhythm_status()
                    self._report_loop_status()
                    if self.args.duration and time.monotonic() - start >= self.args.duration:
                        self.stop_event.set()
                        break
                    time.sleep(self.args.poll_interval)
            finally:
                if monitor_out is not None:
                    with self.output_lock:
                        for pitch, channel in sorted(self._monitor_active_notes):
                            monitor_out.send(
                                mido.Message("note_off", note=pitch, velocity=0, channel=channel)
                            )
                    self._monitor_active_notes.clear()
                self._send_loop_events(melody_output, self.loop_bank.stop_all_now())
                self._send_drum_events(drum_output, self.rhythm_guide.emergency_stop())
                self._panic_outputs(melody_output, drum_output)
                self._melody_output = None

    def forward_input_message(self, outport, msg) -> None:
        with self.output_lock:
            outport.send(msg.copy(time=0))
        if msg.type == "note_on" and msg.velocity > 0:
            self._monitor_active_notes.add(
                (getattr(msg, "note", 0), getattr(msg, "channel", 0))
            )


def list_ports() -> None:
    try:
        import mido
    except ImportError as exc:
        raise SystemExit("mido is not installed in this Python environment.") from exc

    print("Available MIDI input ports:")
    try:
        input_names = mido.get_input_names()
        output_names = mido.get_output_names()
    except Exception as exc:
        raise SystemExit(
            "Could not query MIDI ports. This usually means python-rtmidi is missing "
            "or incompatible with the current Python interpreter. "
            f"Python executable: {sys.executable}. Original error: {exc}"
        ) from exc

    for index, name in enumerate(input_names):
        print(f"  [{index}] {name}")
    print("Available MIDI output ports:")
    for index, name in enumerate(output_names):
        print(f"  [{index}] {name}")


def send_test_output(
    output_port_request: str,
    note: int = 60,
    velocity: int = 96,
    duration: float = 0.35,
) -> None:
    import mido

    output_port = resolve_port(output_port_request, mido.get_output_names(), "output")
    print(f"[test-output] output={output_port} note={note} duration={duration:.2f}s")
    with mido.open_output(output_port) as outport:
        outport.send(mido.Message("note_on", note=note, velocity=velocity, channel=0))
        time.sleep(duration)
        outport.send(mido.Message("note_off", note=note, velocity=0, channel=0))
    print("[test-output] done")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Live AMT MIDI Call-and-Response prototype")
    parser.add_argument("--list-ports", action="store_true", help="list MIDI ports and exit")
    parser.add_argument("--test-output", action="store_true", help="send one test note to output and exit")
    parser.add_argument("--startup-test-note", action="store_true", help="send one test note before live listening")
    parser.add_argument("--input-port", default=DEFAULT_INPUT_PORT)
    parser.add_argument("--output-port", default=DEFAULT_OUTPUT_PORT)
    parser.add_argument(
        "--melody-output-port",
        default=DEFAULT_MELODY_OUTPUT_PORT,
        help="MIDI output port for AI melody and saved loops; drum output uses --output-port",
    )
    parser.add_argument(
        "--backend",
        choices=["amt", "aria"],
        default="amt",
        help="generation backend; use aria for aria-medium-gen live testing",
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--aria-model-id", default=DEFAULT_ARIA_MODEL_ID)
    parser.add_argument(
        "--aria-live-mode",
        choices=["batch", "stream"],
        default="batch",
        help="batch is stable; stream is experimental partial decoding with batch fallback",
    )
    parser.add_argument("--aria-max-new-tokens", type=int, default=1024)
    parser.add_argument("--aria-prompt-tokens", type=int, default=512)
    parser.add_argument("--aria-response-bars", type=int, default=2)
    parser.add_argument("--aria-beats-per-bar", type=int, default=4)
    parser.add_argument("--aria-bpm", type=float, default=60.0)
    parser.add_argument("--aria-monophonic-response", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--aria-stream-chunk-tokens", type=int, default=96)
    parser.add_argument("--aria-stream-fallback-seconds", type=float, default=8.0)
    parser.add_argument(
        "--aria-batch-timeout",
        type=float,
        default=90.0,
        help="maximum seconds to wait for the first Aria batch response events",
    )
    parser.add_argument("--offline", action="store_true", help="load model only from local cache")
    parser.add_argument("--endpoint", default=None, help="optional HuggingFace endpoint")
    parser.add_argument("--hf-token", default=None, help="optional Hugging Face token for gated/private checkpoints")
    parser.add_argument("--response-seconds", type=float, default=8.0)
    parser.add_argument("--max-events", type=int, default=64)
    parser.add_argument(
        "--amt-generation-budget",
        type=float,
        default=1.8,
        help="maximum live seconds spent sampling AMT before any bounded completion",
    )
    parser.add_argument("--top-p", type=float, default=0.98)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument(
        "--response-strategy",
        choices=["controlled_amt", "streaming_amt", "motif"],
        default="controlled_amt",
        help=(
            "controlled_amt keeps the paper-compatible AMT path; streaming_amt plays AMT "
            "events while generation continues and permits only bounded motif completion; "
            "motif uses the deterministic transform by itself"
        ),
    )
    parser.add_argument(
        "--partial-fallback-max-share",
        type=float,
        default=0.50,
        help="maximum target-note share supplied by motif after a non-empty AMT stream",
    )
    parser.add_argument(
        "--musical-control",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="enable phrase-level musical controls; use --no-musical-control to disable",
    )
    parser.add_argument(
        "--speculative-preload",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="pre-generate AMT responses during the revocable endpoint confirmation window",
    )
    parser.add_argument(
        "--fallback-on-empty",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="use a deterministic motif fallback if AMT emits no playable response events",
    )
    parser.add_argument(
        "--duration-match",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="reshape response timing so the heard response duration tracks the human Call duration",
    )
    parser.add_argument(
        "--live-stop-on-target-notes",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="stop live AMT generation as soon as the planned note count is reached",
    )
    parser.add_argument("--duration-match-min-share", type=float, default=0.80)
    parser.add_argument("--duration-match-max-share", type=float, default=1.25)
    parser.add_argument("--same-pitch-limit", type=int, default=2)
    parser.add_argument("--dominant-pitch-max-share", type=float, default=0.35)
    parser.add_argument("--resample-attempts", type=int, default=8)
    parser.add_argument("--response-length-ratio", type=float, default=1.0)
    parser.add_argument("--response-note-ratio", type=float, default=1.0)
    parser.add_argument(
        "--live-style",
        choices=["chromatic", "pentatonic"],
        default="chromatic",
        help="optional live A5/A6 style projection; repository-default chromatic remains available",
    )
    parser.add_argument(
        "--latency-mode",
        choices=sorted(LATENCY_PRESETS),
        default="classic",
        help="buffering preset: fast starts earliest, smooth waits for more events",
    )
    parser.add_argument(
        "--initial-buffer-events",
        type=int,
        default=None,
        help="override latency preset event count",
    )
    parser.add_argument(
        "--initial-buffer-timeout",
        type=float,
        default=None,
        help="override latency preset wait time in seconds",
    )
    parser.add_argument(
        "--first-event-timeout",
        type=float,
        default=8.0,
        help="maximum seconds to wait for the first generated event before treating the response as empty",
    )
    parser.add_argument(
        "--max-underrun-seconds",
        type=float,
        default=2.0,
        help="maximum continuous playback underrun time before ending the current response",
    )
    parser.add_argument("--metrics-path", default=str(default_metrics_path()))
    parser.add_argument("--no-metrics", action="store_true", help="disable per-round CSV metrics")
    parser.add_argument(
        "--monitor-input",
        action="store_true",
        help="forward human MIDI input to the output port so the Call is audible",
    )
    parser.add_argument(
        "--control-file",
        default=None,
        help="newline-delimited live control file used by the performance console",
    )
    parser.add_argument(
        "--drum-pad-min",
        type=int,
        default=36,
        help="lowest MIDI note treated as a drum pad",
    )
    parser.add_argument(
        "--drum-pad-max",
        type=int,
        default=51,
        help="highest MIDI note treated as a drum pad (MiniLab 3 commonly uses 36..51)",
    )
    parser.add_argument(
        "--drum-channel",
        type=int,
        default=9,
        help="zero-based MIDI channel used for Logic's drum track",
    )
    parser.add_argument("--pitch-min", type=int, default=36, help="minimum AI playback MIDI pitch")
    parser.add_argument("--pitch-max", type=int, default=96, help="maximum AI playback MIDI pitch")
    parser.add_argument("--min-note-duration", type=float, default=0.04)
    parser.add_argument("--max-note-duration", type=float, default=2.5)
    parser.add_argument("--theta", type=float, default=0.05)
    parser.add_argument("--window-size", type=int, default=8)
    parser.add_argument("--min-intensity", type=float, default=0.25)
    parser.add_argument("--min-cutoff", type=float, default=None)
    parser.add_argument("--max-cutoff", type=float, default=None)
    parser.add_argument("--chord-cluster-window", type=float, default=0.08)
    parser.add_argument("--endpoint-confirm-delay", type=float, default=0.15)
    parser.add_argument("--poll-interval", type=float, default=0.02)
    parser.add_argument("--duration", type=float, default=None, help="optional auto-stop seconds")
    parser.add_argument("--default-note-duration", type=float, default=0.25)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.pitch_min < 0 or args.pitch_max > 127 or args.pitch_min > args.pitch_max:
        raise SystemExit("--pitch-min/--pitch-max must define a valid MIDI range from 0 to 127.")
    if args.min_note_duration <= 0 or args.max_note_duration <= 0:
        raise SystemExit("--min-note-duration and --max-note-duration must be positive.")
    if args.min_note_duration > args.max_note_duration:
        raise SystemExit("--min-note-duration cannot be larger than --max-note-duration.")
    if args.initial_buffer_events < 1:
        raise SystemExit("--initial-buffer-events must be at least 1.")
    if args.initial_buffer_timeout < 0:
        raise SystemExit("--initial-buffer-timeout cannot be negative.")
    if args.first_event_timeout <= 0:
        raise SystemExit("--first-event-timeout must be positive.")
    if args.max_underrun_seconds <= 0:
        raise SystemExit("--max-underrun-seconds must be positive.")
    if args.response_seconds <= 0:
        raise SystemExit("--response-seconds must be positive.")
    if args.max_events < 1:
        raise SystemExit("--max-events must be at least 1.")
    if args.amt_generation_budget <= 0:
        raise SystemExit("--amt-generation-budget must be positive.")
    if not 0.0 <= args.partial_fallback_max_share <= 1.0:
        raise SystemExit("--partial-fallback-max-share must be in [0, 1].")
    if args.same_pitch_limit < 1:
        raise SystemExit("--same-pitch-limit must be at least 1.")
    if not 0.0 < args.duration_match_min_share <= 1.0:
        raise SystemExit("--duration-match-min-share must be in (0, 1].")
    if args.duration_match_max_share < 1.0:
        raise SystemExit("--duration-match-max-share must be at least 1.0.")
    if args.duration_match_min_share > args.duration_match_max_share:
        raise SystemExit("--duration-match-min-share cannot exceed --duration-match-max-share.")
    if not 0.0 < args.dominant_pitch_max_share <= 1.0:
        raise SystemExit("--dominant-pitch-max-share must be in (0, 1].")
    if args.resample_attempts < 1:
        raise SystemExit("--resample-attempts must be at least 1.")
    if args.response_length_ratio <= 0:
        raise SystemExit("--response-length-ratio must be positive.")
    if args.response_note_ratio <= 0:
        raise SystemExit("--response-note-ratio must be positive.")
    if args.chord_cluster_window < 0:
        raise SystemExit("--chord-cluster-window cannot be negative.")
    if args.endpoint_confirm_delay < 0:
        raise SystemExit("--endpoint-confirm-delay cannot be negative.")
    if args.aria_max_new_tokens < 1:
        raise SystemExit("--aria-max-new-tokens must be at least 1.")
    if args.aria_prompt_tokens < 1:
        raise SystemExit("--aria-prompt-tokens must be at least 1.")
    if args.aria_response_bars < 1:
        raise SystemExit("--aria-response-bars must be at least 1.")
    if args.aria_beats_per_bar < 1:
        raise SystemExit("--aria-beats-per-bar must be at least 1.")
    if args.aria_bpm <= 0:
        raise SystemExit("--aria-bpm must be positive.")
    if args.aria_stream_chunk_tokens < 1:
        raise SystemExit("--aria-stream-chunk-tokens must be at least 1.")
    if args.aria_stream_fallback_seconds <= 0:
        raise SystemExit("--aria-stream-fallback-seconds must be positive.")
    if args.aria_batch_timeout <= 0:
        raise SystemExit("--aria-batch-timeout must be positive.")
    if not 0 <= args.drum_pad_min <= args.drum_pad_max <= 127:
        raise SystemExit("--drum-pad-min/--drum-pad-max must define a MIDI range from 0 to 127.")
    if not 0 <= args.drum_channel <= 15:
        raise SystemExit("--drum-channel must be a MIDI channel from 0 to 15.")


def apply_latency_preset(args: argparse.Namespace) -> None:
    preset_events, preset_timeout = LATENCY_PRESETS[args.latency_mode]
    if args.initial_buffer_events is None:
        args.initial_buffer_events = preset_events
    if args.initial_buffer_timeout is None:
        args.initial_buffer_timeout = preset_timeout


def main() -> None:
    args = build_parser().parse_args()
    apply_latency_preset(args)
    validate_args(args)
    if args.list_ports:
        list_ports()
        return
    if args.test_output:
        require_runtime()
        send_test_output(args.output_port)
        return
    require_runtime()
    app = LiveCallResponseApp(args)
    try:
        app.run()
    except KeyboardInterrupt:
        app.stop_event.set()
        print("\n[shutdown] stopped")
    finally:
        app.metrics.print_summary()


if __name__ == "__main__":
    main()
