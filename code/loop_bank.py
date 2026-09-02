"""MIDI response loop bank used by the live performance console.

The bank intentionally stores MIDI events, not audio.  A performer can keep
changing sounds in Logic and the recalled phrase will always use the current
instrument on the AI track.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


SLOT_NAMES = ("A", "B", "C", "D")


@dataclass(frozen=True)
class LoopEvent:
    offset: float
    kind: str
    note: int
    velocity: int
    channel: int = 0


@dataclass(frozen=True)
class LoopSlotStatus:
    name: str
    has_content: bool
    event_count: int
    loop_seconds: float
    playing: bool
    stopping: bool


class ResponseLoopBank:
    """Four independently playable MIDI loops with musical boundary stops."""

    def __init__(self, minimum_loop_seconds: float = 0.50) -> None:
        self.minimum_loop_seconds = minimum_loop_seconds
        self._events: dict[str, list[LoopEvent]] = {name: [] for name in SLOT_NAMES}
        self._seconds: dict[str, float] = {name: 0.0 for name in SLOT_NAMES}
        self._started_at: dict[str, float | None] = {name: None for name in SLOT_NAMES}
        self._next_index: dict[str, int] = {name: 0 for name in SLOT_NAMES}
        self._stopping: set[str] = set()
        self._pending_events: list[LoopEvent] = []
        # Counts are needed when the same pitch overlaps itself in a phrase.
        self._active_notes: dict[str, dict[tuple[int, int], int]] = {
            name: {} for name in SLOT_NAMES
        }

    def save(self, name: str, events: Iterable[LoopEvent], loop_seconds: float) -> bool:
        name = self._name(name)
        prepared = sorted(
            (self._normalise(event) for event in events),
            key=lambda item: (item.offset, item.kind != "note_off"),
        )
        if not any(event.kind == "note_on" and event.velocity > 0 for event in prepared):
            return False
        # Release a currently playing version before replacing its event list.
        # Otherwise its active notes would be tracked against the new phrase.
        self._pending_events.extend(self.stop_now(name))
        self._events[name] = prepared
        self._seconds[name] = max(self.minimum_loop_seconds, float(loop_seconds))
        return True

    def toggle(self, name: str, now: float) -> bool:
        name = self._name(name)
        if not self._events[name]:
            return False
        if self._started_at[name] is None:
            self._started_at[name] = now
            self._next_index[name] = 0
            self._stopping.discard(name)
        else:
            self._stopping.add(name)
        return True

    def request_stop(self, name: str) -> bool:
        name = self._name(name)
        if self._started_at[name] is None:
            return False
        self._stopping.add(name)
        return True

    def stop_now(self, name: str) -> list[LoopEvent]:
        name = self._name(name)
        offs = self._all_notes_off(name)
        self._started_at[name] = None
        self._next_index[name] = 0
        self._stopping.discard(name)
        return offs

    def stop_all_now(self) -> list[LoopEvent]:
        events: list[LoopEvent] = []
        for name in SLOT_NAMES:
            events.extend(self.stop_now(name))
        return events

    def tick(self, now: float) -> list[LoopEvent]:
        due = self._pending_events
        self._pending_events = []
        for name in SLOT_NAMES:
            started_at = self._started_at[name]
            if started_at is None:
                continue
            while started_at is not None and now >= started_at + self._seconds[name]:
                if name in self._stopping:
                    due.extend(self.stop_now(name))
                    started_at = None
                    break
                started_at += self._seconds[name]
                self._started_at[name] = started_at
                self._next_index[name] = 0
            if started_at is None:
                continue
            elapsed = max(0.0, now - started_at)
            events = self._events[name]
            while self._next_index[name] < len(events):
                event = events[self._next_index[name]]
                if event.offset > elapsed + 0.001:
                    break
                due.append(event)
                self._register(name, event)
                self._next_index[name] += 1
        return due

    def status(self) -> list[LoopSlotStatus]:
        return [
            LoopSlotStatus(
                name=name,
                has_content=bool(self._events[name]),
                event_count=len(self._events[name]),
                loop_seconds=self._seconds[name],
                playing=self._started_at[name] is not None,
                stopping=name in self._stopping,
            )
            for name in SLOT_NAMES
        ]

    def _all_notes_off(self, name: str) -> list[LoopEvent]:
        notes = self._active_notes[name]
        events = [
            LoopEvent(0.0, "note_off", note, 0, channel)
            for note, channel in sorted(notes)
        ]
        notes.clear()
        return events

    def _register(self, name: str, event: LoopEvent) -> None:
        key = (event.note, event.channel)
        if event.kind == "note_on" and event.velocity > 0:
            self._active_notes[name][key] = self._active_notes[name].get(key, 0) + 1
        else:
            count = self._active_notes[name].get(key, 0)
            if count <= 1:
                self._active_notes[name].pop(key, None)
            else:
                self._active_notes[name][key] = count - 1

    @staticmethod
    def _normalise(event: LoopEvent) -> LoopEvent:
        kind = "note_off" if event.kind == "note_off" or event.velocity == 0 else "note_on"
        return LoopEvent(
            offset=max(0.0, float(event.offset)),
            kind=kind,
            note=max(0, min(127, int(event.note))),
            velocity=0 if kind == "note_off" else max(1, min(127, int(event.velocity))),
            channel=max(0, min(15, int(event.channel))),
        )

    @staticmethod
    def _name(name: str) -> str:
        name = str(name).upper()
        if name not in SLOT_NAMES:
            raise ValueError(f"Unknown loop slot: {name}")
        return name
