"""Live-captured MIDI drum loop playback for the performance console.

The loop deliberately keeps the player's timing and velocity.  It is not a
quantiser or a pattern generator: it remembers exactly what was played and
repeats it until the performer asks it to stop or replace it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class DrumEvent:
    offset: float
    kind: str
    note: int
    velocity: int
    channel: int


@dataclass(frozen=True)
class DrumLoopStatus:
    state: str
    event_count: int
    loop_seconds: float
    recording: bool
    playing: bool
    replacing: bool
    stopping: bool


class DrumLoopEngine:
    """Record incoming pad MIDI and replay it at musical loop boundaries."""

    def __init__(self, minimum_loop_seconds: float = 0.35) -> None:
        self.minimum_loop_seconds = minimum_loop_seconds
        self._record_started_at: float | None = None
        self._recorded: list[DrumEvent] = []
        self._active: list[DrumEvent] = []
        self._active_seconds = 0.0
        self._loop_started_at: float | None = None
        self._next_event_index = 0
        self._queued: tuple[list[DrumEvent], float] | None = None
        self._stop_at_boundary = False
        self._active_notes: set[tuple[int, int]] = set()

    @property
    def recording(self) -> bool:
        return self._record_started_at is not None

    @property
    def playing(self) -> bool:
        return bool(self._active) and self._loop_started_at is not None

    def status(self) -> DrumLoopStatus:
        if self.recording:
            state = "recording"
        elif self.playing:
            state = "replacing" if self._queued else "stopping" if self._stop_at_boundary else "looping"
        else:
            state = "idle"
        return DrumLoopStatus(
            state=state,
            event_count=len(self._active),
            loop_seconds=self._active_seconds,
            recording=self.recording,
            playing=self.playing,
            replacing=self._queued is not None,
            stopping=self._stop_at_boundary,
        )

    def start_recording(self, now: float) -> None:
        self._record_started_at = now
        self._recorded = []

    def capture(self, message: Any, now: float) -> bool:
        """Remember a note message only while the performer is recording."""
        if self._record_started_at is None:
            return False
        if message.type not in {"note_on", "note_off"}:
            return False
        kind = "note_off" if message.type == "note_off" or message.velocity == 0 else "note_on"
        self._recorded.append(
            DrumEvent(
                offset=max(0.0, now - self._record_started_at),
                kind=kind,
                note=int(message.note),
                velocity=0 if kind == "note_off" else int(message.velocity),
                channel=int(getattr(message, "channel", 9)),
            )
        )
        return True

    def finish_recording(self, now: float) -> bool:
        if self._record_started_at is None:
            return False
        started_at = self._record_started_at
        self._record_started_at = None
        events = sorted(self._recorded, key=lambda item: (item.offset, item.kind != "note_off"))
        self._recorded = []
        if not any(event.kind == "note_on" for event in events):
            return False
        loop_seconds = max(self.minimum_loop_seconds, now - started_at)
        if self.playing:
            self._queued = (events, loop_seconds)
        else:
            self._active = events
            self._active_seconds = loop_seconds
            self._loop_started_at = started_at
            self._next_event_index = 0
        return True

    def request_stop(self) -> bool:
        if not self.playing:
            return False
        self._stop_at_boundary = True
        self._queued = None
        return True

    def emergency_stop(self) -> list[DrumEvent]:
        offs = self._all_notes_off()
        self._clear_loop()
        self._record_started_at = None
        self._recorded = []
        return offs

    def tick(self, now: float) -> list[DrumEvent]:
        """Return the drum MIDI messages due at *now*.

        A replacement and a normal stop both happen at the end of the current
        loop, so changing a beat does not create an abrupt cut.
        """
        if not self.playing or self._loop_started_at is None:
            return []

        due: list[DrumEvent] = []
        while self.playing and now >= self._loop_started_at + self._active_seconds:
            boundary = self._loop_started_at + self._active_seconds
            if self._stop_at_boundary:
                due.extend(self._all_notes_off())
                self._clear_loop()
                break
            if self._queued is not None:
                self._active, self._active_seconds = self._queued
                self._queued = None
            self._loop_started_at = boundary
            self._next_event_index = 0

        if not self.playing or self._loop_started_at is None:
            return due

        elapsed = max(0.0, now - self._loop_started_at)
        while self._next_event_index < len(self._active):
            event = self._active[self._next_event_index]
            if event.offset > elapsed:
                break
            due.append(event)
            self._register_event(event)
            self._next_event_index += 1
        return due

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

    def _clear_loop(self) -> None:
        self._active = []
        self._active_seconds = 0.0
        self._loop_started_at = None
        self._next_event_index = 0
        self._queued = None
        self._stop_at_boundary = False
        self._active_notes.clear()


def is_drum_pad_note(
    message: Any,
    note_min: int,
    note_max: int,
    drum_channel: int | None = None,
) -> bool:
    """Return whether *message* came from the MiniLab pad area.

    MiniLab 3 pads are configured on MIDI channel 10 in the Logic setup. The
    keyboard can play the same pitches as the pad note range, so pitch alone
    must not steal low keyboard notes from the melodic input. The range is
    retained only as a fallback for test or adapter messages without channel
    information.
    """
    if message.type not in {"note_on", "note_off"}:
        return False
    note = int(message.note)
    channel = getattr(message, "channel", None)
    if drum_channel is not None and channel is not None:
        return int(channel) == int(drum_channel)
    return note_min <= note <= note_max
