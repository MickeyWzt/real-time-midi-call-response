"""Tap-learned rhythm guide for live MIDI performance."""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from statistics import mean
from typing import Any

from drum_loop import DrumEvent


@dataclass(frozen=True)
class RhythmHit:
    step: int
    velocity: int
    micro_offset: float = 0.0


@dataclass(frozen=True)
class RhythmPattern:
    bpm: float
    bars: int
    steps_per_bar: int
    hits: tuple[RhythmHit, ...]
    events: tuple[DrumEvent, ...]
    confidence: float

    @property
    def beat_seconds(self) -> float:
        return 60.0 / self.bpm

    @property
    def loop_seconds(self) -> float:
        return self.beat_seconds * 4.0 * self.bars


@dataclass(frozen=True)
class RhythmGuideStatus:
    state: str
    tap_count: int
    bpm: float
    confidence: float
    bars: int
    steps_per_bar: int
    pattern: tuple[int, ...]
    learning: bool
    playing: bool
    replacing: bool
    stopping: bool
    loop_seconds: float
    saved_slots: tuple[str, ...]
    current_slot: str | None
    queued_slot: str | None


class RhythmGuideEngine:
    """Infer a stable groove from taps and emit an automatic drum guide."""

    def __init__(
        self,
        default_bpm: float = 100.0,
        minimum_bpm: float = 55.0,
        maximum_bpm: float = 180.0,
        auto_finish_seconds: float = 1.25,
        minimum_taps: int = 3,
        maximum_bars: int = 1,
        drum_channel: int = 9,
    ) -> None:
        self.default_bpm = default_bpm
        self.minimum_bpm = minimum_bpm
        self.maximum_bpm = maximum_bpm
        self.auto_finish_seconds = auto_finish_seconds
        self.minimum_taps = max(2, minimum_taps)
        self.maximum_bars = max(1, min(4, maximum_bars))
        self.drum_channel = drum_channel
        self._lock = threading.RLock()
        self._learning_started_at: float | None = None
        self._last_tap_at: float | None = None
        self._taps: list[tuple[float, int]] = []
        self._active: RhythmPattern | None = None
        self._queued: RhythmPattern | None = None
        self._loop_started_at: float | None = None
        self._next_event_index = 0
        self._stop_at_boundary = False
        self._active_notes: set[tuple[int, int]] = set()
        self._slots: dict[str, RhythmPattern] = {}
        self._current_slot: str | None = None
        self._queued_slot: str | None = None

    @property
    def learning(self) -> bool:
        return self._learning_started_at is not None

    @property
    def playing(self) -> bool:
        return self._active is not None and self._loop_started_at is not None

    def start_learning(self, now: float) -> bool:
        with self._lock:
            if self._learning_started_at is not None:
                return False
            self._learning_started_at = now
            self._last_tap_at = None
            self._taps = []
            return True

    def capture(self, message: Any, now: float) -> bool:
        if message.type != "note_on" or int(getattr(message, "velocity", 0)) <= 0:
            return False
        return self.capture_tap(int(getattr(message, "velocity", 96)), now)

    def capture_tap(self, velocity: int, now: float) -> bool:
        with self._lock:
            if self._learning_started_at is None:
                return False
            if self._last_tap_at is not None and now - self._last_tap_at < 0.055:
                return False
            self._taps.append((now, max(1, min(127, velocity))))
            self._last_tap_at = now
            return True

    def maybe_finish_learning(self, now: float) -> bool:
        with self._lock:
            if self._last_tap_at is None or len(self._taps) < self.minimum_taps:
                return False
            if now - self._last_tap_at < self.auto_finish_seconds:
                return False
        return self.finish_learning(now)

    def finish_learning(self, now: float) -> bool:
        with self._lock:
            if self._learning_started_at is None or len(self._taps) < self.minimum_taps:
                return False
            pattern = self._analyze_taps(self._taps)
            self._learning_started_at = None
            self._last_tap_at = None
            self._taps = []
            self._stop_at_boundary = False
            if self.playing:
                self._queued = pattern
                self._queued_slot = None
            else:
                self._active = pattern
                self._current_slot = None
                self._loop_started_at = now + 0.08
                self._next_event_index = 0
            return True

    def request_stop(self) -> bool:
        with self._lock:
            if not self.playing:
                return False
            self._stop_at_boundary = True
            self._queued = None
            self._queued_slot = None
            return True

    def save_slot(self, name: str) -> bool:
        with self._lock:
            slot = self._slot_name(name)
            pattern = self._queued or self._active
            if pattern is None:
                return False
            self._slots[slot] = pattern
            return True

    def load_slot(self, name: str, now: float) -> bool:
        with self._lock:
            slot = self._slot_name(name)
            pattern = self._slots.get(slot)
            if pattern is None:
                return False
            self._stop_at_boundary = False
            if self.playing:
                self._queued = pattern
                self._queued_slot = slot
            else:
                self._active = pattern
                self._current_slot = slot
                self._queued = None
                self._queued_slot = None
                self._loop_started_at = now + 0.08
                self._next_event_index = 0
            return True

    def create_variation(self, now: float) -> bool:
        with self._lock:
            source = self._queued or self._active
            if source is None:
                return False
            occupied = {hit.step for hit in source.hits}
            varied: list[RhythmHit] = []
            total_steps = source.bars * source.steps_per_bar
            for index, hit in enumerate(source.hits):
                shift = 1 if index % 3 == 1 else -1 if index % 4 == 3 else 0
                candidate = (hit.step + shift) % total_steps
                step = hit.step if candidate in occupied and candidate != hit.step else candidate
                varied.append(
                    RhythmHit(
                        step=step,
                        velocity=max(42, min(127, hit.velocity + (8 if index % 2 == 0 else -7))),
                        micro_offset=hit.micro_offset * 0.6,
                    )
                )
            offbeat = 10 % total_steps
            if offbeat not in {hit.step for hit in varied}:
                varied.append(RhythmHit(step=offbeat, velocity=68))
            hits = tuple(sorted(varied, key=lambda item: item.step))
            pattern = RhythmPattern(
                bpm=source.bpm,
                bars=source.bars,
                steps_per_bar=source.steps_per_bar,
                hits=hits,
                events=self._build_accompaniment(hits, source.bpm, source.bars),
                confidence=source.confidence,
            )
            if self.playing:
                self._queued = pattern
                self._queued_slot = None
            else:
                self._active = pattern
                self._current_slot = None
                self._loop_started_at = now + 0.08
                self._next_event_index = 0
            self._stop_at_boundary = False
            return True

    def emergency_stop(self) -> list[DrumEvent]:
        with self._lock:
            offs = self._all_notes_off()
            self._active = None
            self._queued = None
            self._loop_started_at = None
            self._next_event_index = 0
            self._stop_at_boundary = False
            self._learning_started_at = None
            self._last_tap_at = None
            self._taps = []
            self._active_notes.clear()
            self._current_slot = None
            self._queued_slot = None
            return offs

    def status(self) -> RhythmGuideStatus:
        with self._lock:
            pattern = self._queued or self._active
            if self.learning:
                state = "learning"
            elif self.playing:
                state = "replacing" if self._queued else "stopping" if self._stop_at_boundary else "running"
            else:
                state = "idle"
            return RhythmGuideStatus(
                state=state,
                tap_count=len(self._taps),
                bpm=round(pattern.bpm if pattern else self.default_bpm, 1),
                confidence=round(pattern.confidence if pattern else 0.0, 3),
                bars=pattern.bars if pattern else 1,
                steps_per_bar=pattern.steps_per_bar if pattern else 16,
                pattern=tuple(hit.step for hit in pattern.hits) if pattern else (),
                learning=self.learning,
                playing=self.playing,
                replacing=self._queued is not None,
                stopping=self._stop_at_boundary,
                loop_seconds=round(pattern.loop_seconds if pattern else 0.0, 3),
                saved_slots=tuple(sorted(self._slots)),
                current_slot=self._current_slot,
                queued_slot=self._queued_slot,
            )

    def timing(self) -> tuple[float, float] | None:
        with self._lock:
            if not self.playing or self._active is None or self._loop_started_at is None:
                return None
            return self._active.bpm, self._loop_started_at

    def tick(self, now: float) -> list[DrumEvent]:
        with self._lock:
            if not self.playing or self._active is None or self._loop_started_at is None:
                return []

            due: list[DrumEvent] = []
            while self.playing and self._active is not None and self._loop_started_at is not None:
                boundary = self._loop_started_at + self._active.loop_seconds
                if now < boundary:
                    break
                if self._stop_at_boundary:
                    due.extend(self._all_notes_off())
                    self._active = None
                    self._loop_started_at = None
                    self._next_event_index = 0
                    self._stop_at_boundary = False
                    break
                if self._queued is not None:
                    self._active = self._queued
                    self._queued = None
                    self._current_slot = self._queued_slot
                    self._queued_slot = None
                self._loop_started_at = boundary
                self._next_event_index = 0

            if not self.playing or self._active is None or self._loop_started_at is None:
                return due

            elapsed = max(0.0, now - self._loop_started_at)
            while self._next_event_index < len(self._active.events):
                event = self._active.events[self._next_event_index]
                if event.offset > elapsed:
                    break
                due.append(event)
                self._register_event(event)
                self._next_event_index += 1
            return due

    def _analyze_taps(self, taps: list[tuple[float, int]]) -> RhythmPattern:
        origin = taps[0][0]
        times = [timestamp - origin for timestamp, _ in taps]
        intervals = [b - a for a, b in zip(times, times[1:]) if b - a >= 0.055]
        preferred_bpm = self._active.bpm if self._active is not None else self.default_bpm
        candidates = {preferred_bpm}
        for interval in intervals:
            for factor in (0.5, 2.0 / 3.0, 1.0, 4.0 / 3.0, 2.0, 3.0, 4.0):
                beat = interval * factor
                if beat <= 0:
                    continue
                bpm = 60.0 / beat
                if self.minimum_bpm <= bpm <= self.maximum_bpm:
                    candidates.add(round(bpm, 4))

        def candidate_score(bpm: float) -> float:
            grid = (60.0 / bpm) / 4.0
            errors = [abs(value - round(value / grid) * grid) for value in times]
            interval_errors = [
                abs(value - round(value / grid) * grid) for value in intervals
            ]
            interval_steps = [max(1, int(round(value / grid))) for value in intervals]
            subdivision_penalty = mean(
                {4: 0.0, 2: 0.012, 1: 0.025, 8: 0.025}.get(steps, 0.045)
                for steps in interval_steps
            ) if interval_steps else 0.0
            tempo_bias = abs(math.log(max(bpm, 1.0) / preferred_bpm)) * 0.035
            interval_consistency = mean(interval_errors) * 1.5 if interval_errors else 0.0
            return mean(errors) + interval_consistency + subdivision_penalty + tempo_bias

        bpm = min(candidates, key=candidate_score)
        grid = (60.0 / bpm) / 4.0
        raw_steps = [max(0, int(round(value / grid))) for value in times]
        max_step = max(raw_steps, default=0)
        closing_tap = len(raw_steps) >= 3 and max_step >= 16 and max_step % 16 == 0
        if closing_tap:
            bars = max(1, min(self.maximum_bars, max_step // 16))
            usable = list(zip(times[:-1], taps[:-1], raw_steps[:-1]))
        else:
            bars = max(1, min(self.maximum_bars, math.ceil((max_step + 1) / 16)))
            usable = list(zip(times, taps, raw_steps))
        total_steps = bars * 16

        grouped: dict[int, list[tuple[int, float]]] = {}
        for relative, (_, velocity), raw_step in usable:
            step = raw_step % total_steps
            residual = relative - raw_step * grid
            grouped.setdefault(step, []).append((velocity, residual))
        if not grouped:
            grouped[0] = [(96, 0.0)]

        hits = tuple(
            RhythmHit(
                step=step,
                velocity=round(mean(item[0] for item in values)),
                micro_offset=max(-grid * 0.18, min(grid * 0.18, mean(item[1] for item in values) * 0.22)),
            )
            for step, values in sorted(grouped.items())
        )
        confidence = max(0.15, min(1.0, 1.0 - candidate_score(bpm) * 1.8))
        events = self._build_accompaniment(hits, bpm, bars)
        return RhythmPattern(
            bpm=bpm,
            bars=bars,
            steps_per_bar=16,
            hits=hits,
            events=events,
            confidence=confidence,
        )

    def _build_accompaniment(
        self,
        hits: tuple[RhythmHit, ...],
        bpm: float,
        bars: int,
    ) -> tuple[DrumEvent, ...]:
        grid = (60.0 / bpm) / 4.0
        note_length = min(0.09, grid * 0.48)
        hit_map = {hit.step: hit for hit in hits}
        scheduled: dict[tuple[int, int], tuple[int, float]] = {}

        for bar in range(bars):
            base = bar * 16
            for local_step in range(0, 16, 2):
                step = base + local_step
                velocity = 76 if local_step in (0, 8) else 58
                scheduled[(step, 42)] = (velocity, 0.0)
            for local_step in (4, 12):
                scheduled[(base + local_step, 38)] = (94, 0.0)

        for hit in hits:
            scheduled[(hit.step, 36)] = (max(62, hit.velocity), hit.micro_offset)
        if not any(note == 36 and step % 16 == 0 for step, note in scheduled):
            scheduled[(0, 36)] = (102, 0.0)

        events: list[DrumEvent] = []
        for (step, note), (velocity, micro_offset) in sorted(scheduled.items()):
            onset = max(0.0, step * grid + micro_offset)
            events.append(DrumEvent(onset, "note_on", note, velocity, self.drum_channel))
            events.append(DrumEvent(onset + note_length, "note_off", note, 0, self.drum_channel))
        return tuple(sorted(events, key=lambda item: (item.offset, item.kind != "note_off", item.note)))

    def _register_event(self, event: DrumEvent) -> None:
        key = (event.note, event.channel)
        if event.kind == "note_on" and event.velocity > 0:
            self._active_notes.add(key)
        else:
            self._active_notes.discard(key)

    def _all_notes_off(self) -> list[DrumEvent]:
        return [
            DrumEvent(0.0, "note_off", note, 0, channel)
            for note, channel in sorted(self._active_notes)
        ]

    @staticmethod
    def _slot_name(name: str) -> str:
        slot = str(name).upper()
        if slot not in {"A", "B", "C", "D"}:
            raise ValueError(f"Unknown rhythm slot: {slot}")
        return slot
