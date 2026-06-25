"""
切脉大师: MIDI VAD endpoint detector.

Implements the endpoint rule described in 论文内容.pdf:

    S(tau) = exp(-mu_tempo * tau)
    S(tau_cutoff) = theta
    tau_cutoff = ln(1 / theta) / mu_tempo

The local inhomogeneous Poisson intensity lambda(t) is estimated with a
sliding-window onset density, mu_tempo. When the silence since the last
Note-On exceeds tau_cutoff, the current human "Call" phrase is cut and
returned for downstream Transformer inference.

This file has no hard dependency on mido unless you run live MIDI mode.
Use simulation mode first:

    python code/midi_vad_endpoint.py --simulate
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Deque, Iterable, List, Optional, Sequence


PROJECT_DEPS = Path(__file__).resolve().parents[1] / ".python_deps"
if PROJECT_DEPS.exists():
    sys.path.insert(0, str(PROJECT_DEPS))


@dataclass(frozen=True)
class MidiNoteEvent:
    """A minimal MIDI note event used by the endpoint detector."""

    timestamp: float
    pitch: int
    velocity: int = 64
    kind: str = "note_on"


@dataclass(frozen=True)
class EndpointCandidate:
    """Diagnostic record emitted when a revocable endpoint candidate appears."""

    phrase: List[MidiNoteEvent]
    candidate_time: float
    last_event_time: float
    silence: float
    mu_tempo: float
    tau_cutoff: float
    survival: float
    theta: float
    onset_cluster_count: int = 0


@dataclass(frozen=True)
class EndpointCancel:
    """Diagnostic record emitted when new input revokes a candidate endpoint."""

    phrase: List[MidiNoteEvent]
    candidate_time: float
    cancel_time: float
    new_event: MidiNoteEvent
    reason: str
    onset_cluster_count: int = 0


@dataclass(frozen=True)
class EndpointDecision:
    """Diagnostic record emitted when a Call phrase is cut."""

    phrase: List[MidiNoteEvent]
    cut_time: float
    last_event_time: float
    silence: float
    mu_tempo: float
    tau_cutoff: float
    survival: float
    theta: float
    candidate_time: Optional[float] = None
    confirmation_delay: float = 0.0
    onset_cluster_count: int = 0


class MidiEndpointVAD:
    """
    Dynamic MIDI endpoint detector based on survival analysis.

    The paper models the Note-On stream as an inhomogeneous Poisson process.
    In implementation, the local intensity lambda(t) is approximated by
    recent onset density:

        mu_tempo = (K - 1) / (t_K - t_1)

    over the current sliding window. The phrase endpoint is detected when:

        current_time - T_last_event > ln(1/theta) / mu_tempo
    """

    def __init__(
        self,
        theta: float = 0.05,
        window_size: int = 8,
        min_intensity: float = 0.25,
        min_cutoff: Optional[float] = None,
        max_cutoff: Optional[float] = None,
        chord_cluster_window: float = 0.08,
        endpoint_confirm_delay: float = 0.15,
        on_candidate_endpoint: Optional[Callable[[EndpointCandidate], None]] = None,
        on_candidate_cancel: Optional[Callable[[EndpointCancel], None]] = None,
        on_endpoint: Optional[Callable[[EndpointDecision], None]] = None,
    ) -> None:
        if not 0.0 < theta < 1.0:
            raise ValueError("theta must be between 0 and 1, for example 0.05.")
        if window_size < 2:
            raise ValueError("window_size must be at least 2.")
        if min_intensity <= 0:
            raise ValueError("min_intensity must be positive.")
        if min_cutoff is not None and min_cutoff <= 0:
            raise ValueError("min_cutoff must be positive when provided.")
        if max_cutoff is not None and max_cutoff <= 0:
            raise ValueError("max_cutoff must be positive when provided.")
        if chord_cluster_window < 0:
            raise ValueError("chord_cluster_window cannot be negative.")
        if endpoint_confirm_delay < 0:
            raise ValueError("endpoint_confirm_delay cannot be negative.")
        if (
            min_cutoff is not None
            and max_cutoff is not None
            and min_cutoff > max_cutoff
        ):
            raise ValueError("min_cutoff cannot be larger than max_cutoff.")

        self.theta = theta
        self.window_size = window_size
        self.min_intensity = min_intensity
        self.min_cutoff = min_cutoff
        self.max_cutoff = max_cutoff
        self.chord_cluster_window = chord_cluster_window
        self.endpoint_confirm_delay = endpoint_confirm_delay
        self.on_candidate_endpoint = on_candidate_endpoint
        self.on_candidate_cancel = on_candidate_cancel
        self.on_endpoint = on_endpoint

        self._onsets: Deque[float] = deque(maxlen=window_size)
        self._phrase: List[MidiNoteEvent] = []
        self._active = False
        self._last_event_time: Optional[float] = None
        self._current_cluster_start: Optional[float] = None
        self._candidate_time: Optional[float] = None

    @property
    def active(self) -> bool:
        return self._active

    @property
    def phrase(self) -> Sequence[MidiNoteEvent]:
        return tuple(self._phrase)

    def observe_note_on(
        self,
        pitch: int,
        velocity: int = 64,
        timestamp: Optional[float] = None,
    ) -> None:
        """Register one human Note-On event."""

        now = time.monotonic() if timestamp is None else timestamp
        event = MidiNoteEvent(timestamp=now, pitch=pitch, velocity=velocity)
        cancelled_candidate_time = self._candidate_time

        self._candidate_time = None
        if (
            self._current_cluster_start is None
            or not self._active
            or now - self._current_cluster_start > self.chord_cluster_window
        ):
            self._onsets.append(now)
            self._current_cluster_start = now
        self._phrase.append(event)
        self._last_event_time = now
        self._active = True

        if cancelled_candidate_time is not None and self.on_candidate_cancel is not None:
            self.on_candidate_cancel(
                EndpointCancel(
                    phrase=list(self._phrase),
                    candidate_time=cancelled_candidate_time,
                    cancel_time=now,
                    new_event=event,
                    reason="new_note_on",
                    onset_cluster_count=len(self._onsets),
                )
            )

    def mu_tempo(self) -> float:
        """Estimate local Poisson intensity from recent onset density."""

        if len(self._onsets) < 2:
            return self.min_intensity

        duration = self._onsets[-1] - self._onsets[0]
        if duration <= 0:
            return self.min_intensity

        density = (len(self._onsets) - 1) / duration
        return max(density, self.min_intensity)

    def tau_cutoff(self) -> float:
        """Compute tau_cutoff = ln(1/theta) / mu_tempo."""

        cutoff = math.log(1.0 / self.theta) / self.mu_tempo()
        if self.min_cutoff is not None:
            cutoff = max(cutoff, self.min_cutoff)
        if self.max_cutoff is not None:
            cutoff = min(cutoff, self.max_cutoff)
        return cutoff

    def survival(self, silence: float) -> float:
        """Compute S(tau)=exp(-mu_tempo*tau) for current local intensity."""

        return math.exp(-self.mu_tempo() * max(silence, 0.0))

    def tick(self, timestamp: Optional[float] = None) -> Optional[EndpointDecision]:
        """
        Check whether the current phrase has ended.

        Call this periodically from a polling loop. Returns an EndpointDecision
        exactly once per phrase when the silence boundary is crossed.
        """

        if not self._active or self._last_event_time is None or not self._phrase:
            return None

        now = time.monotonic() if timestamp is None else timestamp
        silence = now - self._last_event_time
        cutoff = self.tau_cutoff()

        if silence <= cutoff:
            self._candidate_time = None
            return None

        if self._candidate_time is None:
            self._candidate_time = now
            if self.on_candidate_endpoint is not None:
                self.on_candidate_endpoint(
                    EndpointCandidate(
                        phrase=list(self._phrase),
                        candidate_time=now,
                        last_event_time=self._last_event_time,
                        silence=silence,
                        mu_tempo=self.mu_tempo(),
                        tau_cutoff=cutoff,
                        survival=self.survival(silence),
                        theta=self.theta,
                        onset_cluster_count=len(self._onsets),
                    )
                )
            if self.endpoint_confirm_delay > 0:
                return None
        elif now - self._candidate_time < self.endpoint_confirm_delay:
            return None

        decision = EndpointDecision(
            phrase=list(self._phrase),
            cut_time=now,
            last_event_time=self._last_event_time,
            silence=silence,
            mu_tempo=self.mu_tempo(),
            tau_cutoff=cutoff,
            survival=self.survival(silence),
            theta=self.theta,
            candidate_time=self._candidate_time,
            confirmation_delay=0.0 if self._candidate_time is None else now - self._candidate_time,
            onset_cluster_count=len(self._onsets),
        )

        self._phrase.clear()
        self._onsets.clear()
        self._active = False
        self._last_event_time = None
        self._current_cluster_start = None
        self._candidate_time = None

        if self.on_endpoint is not None:
            self.on_endpoint(decision)

        return decision


def print_decision(decision: EndpointDecision) -> None:
    pitches = [event.pitch for event in decision.phrase]
    print("\n[切脉大师] Endpoint detected")
    print(f"  phrase_len     : {len(decision.phrase)} notes")
    print(f"  onset_clusters : {decision.onset_cluster_count}")
    print(f"  pitches        : {pitches}")
    print(f"  mu_tempo       : {decision.mu_tempo:.3f} note_on/sec")
    print(f"  tau_cutoff     : {decision.tau_cutoff:.3f} sec")
    print(f"  silence        : {decision.silence:.3f} sec")
    print(f"  confirm_delay  : {decision.confirmation_delay:.3f} sec")
    print(f"  survival S(tau): {decision.survival:.4f}")
    print(f"  theta          : {decision.theta:.4f}")
    print("  action         : close Call window -> trigger Transformer thread")


def run_simulation(args: argparse.Namespace) -> None:
    """
    Deterministic demo: three short phrases with different onset densities.

    Timestamps are simulated, so this runs instantly and does not require MIDI.
    """

    vad = MidiEndpointVAD(
        theta=args.theta,
        window_size=args.window_size,
        min_intensity=args.min_intensity,
        min_cutoff=args.min_cutoff,
        max_cutoff=args.max_cutoff,
        chord_cluster_window=args.chord_cluster_window,
        endpoint_confirm_delay=args.endpoint_confirm_delay,
        on_endpoint=print_decision,
    )

    simulated_events = [
        # Phrase 1: medium tempo
        MidiNoteEvent(0.00, 60),
        MidiNoteEvent(0.42, 64),
        MidiNoteEvent(0.82, 67),
        MidiNoteEvent(1.24, 72),
        # Phrase 2: faster run
        MidiNoteEvent(4.00, 72),
        MidiNoteEvent(4.16, 74),
        MidiNoteEvent(4.31, 75),
        MidiNoteEvent(4.47, 79),
        MidiNoteEvent(4.63, 81),
        # Phrase 3: slower gesture
        MidiNoteEvent(8.00, 55),
        MidiNoteEvent(8.82, 59),
        MidiNoteEvent(9.64, 62),
    ]

    print("[切脉大师] simulation started")
    print(
        "[formula] S(tau)=exp(-mu_tempo*tau), "
        "tau_cutoff=ln(1/theta)/mu_tempo"
    )

    last_checked = 0.0
    for event in simulated_events:
        probe_time = last_checked
        while probe_time < event.timestamp:
            vad.tick(probe_time)
            probe_time += args.poll_interval

        print(f"[note_on] t={event.timestamp:5.2f}s pitch={event.pitch}")
        vad.observe_note_on(event.pitch, event.velocity, event.timestamp)
        last_checked = event.timestamp

    # Let the final phrase become silent long enough to cut.
    end_time = simulated_events[-1].timestamp + 8.0
    probe_time = last_checked
    while probe_time <= end_time:
        vad.tick(probe_time)
        probe_time += args.poll_interval

    print("\n[切脉大师] simulation finished")


def list_midi_ports() -> None:
    try:
        import mido
    except ImportError:
        print("mido is not installed. Install mido + python-rtmidi for live MIDI mode.")
        return

    names = mido.get_input_names()
    if not names:
        print("No MIDI input ports found.")
        return

    print("Available MIDI input ports:")
    for index, name in enumerate(names):
        print(f"  [{index}] {name}")


def run_live(args: argparse.Namespace) -> None:
    try:
        import mido
    except ImportError as exc:
        raise SystemExit(
            "Live MIDI mode requires mido and a MIDI backend such as python-rtmidi. "
            "Simulation mode works without them: python code/midi_vad_endpoint.py --simulate"
        ) from exc

    port_name = args.port
    if port_name is None:
        input_names = mido.get_input_names()
        if not input_names:
            raise SystemExit("No MIDI input ports found. Connect a keyboard or virtual MIDI port.")
        port_name = input_names[0]

    vad = MidiEndpointVAD(
        theta=args.theta,
        window_size=args.window_size,
        min_intensity=args.min_intensity,
        min_cutoff=args.min_cutoff,
        max_cutoff=args.max_cutoff,
        chord_cluster_window=args.chord_cluster_window,
        endpoint_confirm_delay=args.endpoint_confirm_delay,
        on_endpoint=print_decision,
    )

    print(f"[切脉大师] listening on MIDI input: {port_name}")
    if args.duration:
        print(f"Auto-stop after {args.duration:.1f} seconds.")
    else:
        print("Press Ctrl+C to stop.")

    with mido.open_input(port_name) as inport:
        start = time.monotonic()
        while True:
            for msg in inport.iter_pending():
                if msg.type == "note_on" and msg.velocity > 0:
                    now = time.monotonic()
                    vad.observe_note_on(msg.note, msg.velocity, now)
                    print(
                        f"[note_on] pitch={msg.note:3d} velocity={msg.velocity:3d} "
                        f"mu={vad.mu_tempo():.3f}/s cutoff={vad.tau_cutoff():.3f}s"
                    )
            vad.tick()
            if args.duration and time.monotonic() - start >= args.duration:
                break
            time.sleep(args.poll_interval)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="切脉大师: nonhomogeneous-Poisson MIDI VAD endpoint detector"
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--simulate", action="store_true", help="run built-in MIDI event simulation")
    mode.add_argument("--list-ports", action="store_true", help="list live MIDI input ports")
    mode.add_argument("--live", action="store_true", help="listen to a real MIDI input port")

    parser.add_argument("--port", help="MIDI input port name for --live")
    parser.add_argument("--theta", type=float, default=0.05, help="survival confidence boundary")
    parser.add_argument("--window-size", type=int, default=8, help="recent onsets used for mu_tempo")
    parser.add_argument(
        "--min-intensity",
        type=float,
        default=0.25,
        help="fallback lower bound for mu_tempo in note_on/sec",
    )
    parser.add_argument(
        "--min-cutoff",
        type=float,
        default=None,
        help="optional lower clamp for tau_cutoff in seconds",
    )
    parser.add_argument(
        "--max-cutoff",
        type=float,
        default=None,
        help="optional upper clamp for tau_cutoff in seconds",
    )
    parser.add_argument(
        "--chord-cluster-window",
        type=float,
        default=0.08,
        help="seconds in which near-simultaneous note-ons count as one onset cluster",
    )
    parser.add_argument(
        "--endpoint-confirm-delay",
        type=float,
        default=0.15,
        help="seconds to wait after a candidate endpoint before triggering AI",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=0.02,
        help="endpoint polling interval in seconds",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="optional live-mode auto-stop duration in seconds",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.list_ports:
        list_midi_ports()
    elif args.live:
        run_live(args)
    else:
        run_simulation(args)


if __name__ == "__main__":
    main()
