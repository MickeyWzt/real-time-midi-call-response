from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Optional


ROOT = Path(__file__).resolve().parents[1]
PROJECT_DEPS = ROOT / ".python_deps"
if PROJECT_DEPS.exists():
    sys.path.insert(0, str(PROJECT_DEPS))

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from vst_host_manager import get_piano_host_manager


STATIC_DIR = ROOT / "code" / "static"
LIVE_SCRIPT = ROOT / "code" / "live_call_response.py"
MIDI_HELPER_SCRIPT = ROOT / "code" / "midi_io_helper.py"
LIVE_LOG_PATH = ROOT / "logs" / "mfp_live_studio_live.log"
LIVE_CONTROL_PATH = ROOT / "logs" / "mfp_live_controls.jsonl"
DEFAULT_OUTPUT_PORT = "Python_OUT"
DEFAULT_MELODY_OUTPUT_PORT = "Logic Pro Virtual In"
LIVE_BACKEND = "amt"
LIVE_RESPONSE_STRATEGY = "streaming_amt"
DEFAULT_MODEL_ID = "stanford-crfm/music-small-800k"
DEFAULT_ARIA_MODEL_ID = str(ROOT / "model_weights" / "aria-medium-gen")
DEVICE_POLL_SECONDS = 1.0
MIDI_ERROR_RETRY_SECONDS = 5.0

IGNORED_INPUT_TERMS = (
    "python_in",
    "python_out",
    "loopmidi",
    "microsoft gs",
    "wavetable",
)
PHYSICAL_KEYBOARD_HINTS = (
    "minilab",
    "arturia",
    "keylab",
    "launchkey",
    "keystation",
    "oxygen",
    "mpk",
    "novation",
    "m-audio",
    "alesis",
    "roland",
    "yamaha",
    "korg",
    "akai",
    "usb midi",
    "midi keyboard",
)


def _norm(name: str) -> str:
    return name.casefold()


def is_virtual_or_system_port(name: str) -> bool:
    lowered = _norm(name)
    return any(term in lowered for term in IGNORED_INPUT_TERMS)


def is_likely_physical_keyboard(name: str) -> bool:
    lowered = _norm(name)
    return not is_virtual_or_system_port(name) and any(
        hint in lowered for hint in PHYSICAL_KEYBOARD_HINTS
    )


def resolve_port(requested: str, names: list[str]) -> Optional[str]:
    if not names:
        return None
    if requested in names:
        return requested
    lowered = _norm(requested)
    for name in names:
        if lowered in _norm(name):
            return name
    return None


def choose_default_input(inputs: list[str]) -> tuple[Optional[str], bool]:
    physical = [name for name in inputs if is_likely_physical_keyboard(name)]
    if physical:
        return physical[0], False

    fallback = resolve_port("Python_IN", inputs)
    if fallback:
        return fallback, True
    return (inputs[0], False) if inputs else (None, True)


def choose_virtual_keyboard_output(outputs: list[str]) -> Optional[str]:
    return resolve_port("Python_IN", outputs)


def choose_audio_output(outputs: list[str], piano_host_available: bool = True) -> Optional[str]:
    loopback_output = resolve_port(DEFAULT_OUTPUT_PORT, outputs)
    if loopback_output:
        return loopback_output

    return (
        resolve_port("Microsoft GS", outputs)
        or resolve_port("Wavetable", outputs)
        or (outputs[0] if outputs else None)
    )


def needs_piano_host(output_port: str) -> bool:
    return os.name == "nt" and DEFAULT_OUTPUT_PORT.casefold() in output_port.casefold()


@dataclass
class StudioConfig:
    backend: str = LIVE_BACKEND
    response_strategy: str = LIVE_RESPONSE_STRATEGY
    model_id: str = DEFAULT_MODEL_ID
    aria_model_id: str = DEFAULT_ARIA_MODEL_ID
    response_seconds: float = 3.0
    max_events: int = 12
    top_p: float = 0.95
    temperature: float = 0.75
    latency_mode: str = "fast"
    max_underrun_seconds: float = 4.5
    min_cutoff: Optional[float] = None
    max_cutoff: Optional[float] = None
    chord_cluster_window: float = 0.08
    endpoint_confirm_delay: float = 0.15
    amt_generation_budget: float = 4.0
    partial_fallback_max_share: float = 0.50
    output_port: str = DEFAULT_OUTPUT_PORT
    melody_output_port: str = DEFAULT_MELODY_OUTPUT_PORT


@dataclass
class SessionState:
    running: bool = False
    pid: Optional[int] = None
    input_port: Optional[str] = None
    output_port: str = DEFAULT_OUTPUT_PORT
    virtual_mode: bool = True
    status: str = "idle"
    round_id: Optional[int] = None
    last_error: Optional[str] = None
    model_status: str = "not loaded"
    started_at: Optional[float] = None


@dataclass
class DeviceState:
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    selected_input: Optional[str] = None
    selected_output: Optional[str] = None
    virtual_keyboard_output: Optional[str] = None
    virtual_mode: bool = True
    midi_error: Optional[str] = None


@dataclass
class RhythmState:
    state: str = "idle"
    tap_count: int = 0
    bpm: float = 100.0
    confidence: float = 0.0
    bars: int = 1
    steps_per_bar: int = 16
    pattern: list[int] = field(default_factory=list)
    loop_seconds: float = 0.0
    learning: bool = False
    playing: bool = False
    replacing: bool = False
    stopping: bool = False
    saved_slots: list[str] = field(default_factory=list)
    current_slot: Optional[str] = None
    queued_slot: Optional[str] = None


@dataclass
class LoopSlotState:
    name: str
    has_content: bool = False
    event_count: int = 0
    loop_seconds: float = 0.0
    playing: bool = False
    stopping: bool = False


@dataclass
class LoopBankState:
    mode: str = "response"
    latest_ready: bool = False
    slots: list[LoopSlotState] = field(
        default_factory=lambda: [LoopSlotState(name=name) for name in ("A", "B", "C", "D")]
    )


class ConnectionManager:
    def __init__(self) -> None:
        self.active: set[WebSocket] = set()

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.active.add(websocket)

    def disconnect(self, websocket: WebSocket) -> None:
        self.active.discard(websocket)

    async def broadcast(self, payload: dict[str, Any]) -> None:
        stale: list[WebSocket] = []
        message = json.dumps(payload, ensure_ascii=False)
        for websocket in list(self.active):
            try:
                await websocket.send_text(message)
            except Exception:
                stale.append(websocket)
        for websocket in stale:
            self.disconnect(websocket)


class LiveStudioController:
    def __init__(self) -> None:
        self.config = StudioConfig()
        self.session = SessionState()
        self.devices = DeviceState()
        self.rhythm = RhythmState()
        self.loop_bank = LoopBankState()
        self.process: Optional[subprocess.Popen[str]] = None
        self.manager = ConnectionManager()
        self.piano_host = get_piano_host_manager()
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self._last_device_signature: Optional[
            tuple[tuple[str, ...], tuple[str, ...], Optional[str]]
        ] = None
        self._round_data: dict[str, Any] = {}
        self._next_midi_probe_at = 0.0

    def attach_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self.loop = loop

    def _threadsafe_broadcast(self, payload: dict[str, Any]) -> None:
        if self.loop is None:
            return
        self.loop.call_soon_threadsafe(
            lambda: asyncio.create_task(self.manager.broadcast(payload))
        )

    def _run_midi_helper(
        self,
        *args: str,
        timeout: float = 3.0,
    ) -> tuple[Optional[dict[str, Any]], Optional[str]]:
        cmd = [sys.executable, str(MIDI_HELPER_SCRIPT), *args]
        try:
            completed = subprocess.run(
                cmd,
                cwd=str(ROOT),
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return None, "CoreMIDI helper timed out."
        except OSError as exc:
            return None, f"Could not start CoreMIDI helper: {exc}"

        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError:
            payload = None
        if completed.returncode != 0:
            if isinstance(payload, dict) and payload.get("error"):
                return None, str(payload["error"])
            detail = (completed.stderr or completed.stdout).strip().splitlines()
            message = detail[-1] if detail else "No diagnostic output."
            return None, f"CoreMIDI helper exited {completed.returncode}: {message}"
        if payload is None:
            return None, "CoreMIDI helper returned invalid output."
        if not payload.get("ok"):
            return None, str(payload.get("error") or "CoreMIDI helper failed.")
        return payload, None

    def refresh_devices(self, keep_manual: bool = False, force: bool = False) -> DeviceState:
        now = time.monotonic()
        if not force and now < self._next_midi_probe_at:
            return self.devices

        payload, midi_error = self._run_midi_helper("--list-ports")
        if midi_error:
            self._next_midi_probe_at = now + MIDI_ERROR_RETRY_SECONDS
            self.devices = replace(self.devices, midi_error=midi_error)
            return self.devices

        self._next_midi_probe_at = now + DEVICE_POLL_SECONDS
        inputs = list(payload.get("inputs", [])) if payload else []
        outputs = list(payload.get("outputs", [])) if payload else []
        selected_input, virtual_mode = choose_default_input(inputs)
        selected_output = choose_audio_output(outputs, self.piano_host.status().available)
        virtual_output = choose_virtual_keyboard_output(outputs)

        if keep_manual and self.devices.selected_input in inputs:
            selected_input = self.devices.selected_input
            virtual_mode = selected_input == virtual_output

        self.devices = DeviceState(
            inputs=inputs,
            outputs=outputs,
            selected_input=selected_input,
            selected_output=selected_output,
            virtual_keyboard_output=virtual_output,
            virtual_mode=virtual_mode,
            midi_error=None,
        )
        return self.devices

    async def poll_devices_forever(self) -> None:
        while True:
            try:
                old_input = self.devices.selected_input
                devices = self.refresh_devices(keep_manual=self.session.running)
                signature = (tuple(devices.inputs), tuple(devices.outputs), devices.midi_error)
                if signature != self._last_device_signature:
                    self._last_device_signature = signature
                    await self.broadcast_devices()
                if (
                    self.session.running
                    and old_input
                    and devices.selected_input
                    and old_input != devices.selected_input
                ):
                    await self.manager.broadcast(
                        {
                            "type": "log",
                            "level": "info",
                            "message": (
                                "New MIDI input detected. It will be used after "
                                "you stop and restart the live session."
                            ),
                        }
                    )
            except Exception as exc:
                await self.manager.broadcast(
                    {"type": "error", "message": f"MIDI device poll failed: {exc}"}
                )
            await asyncio.sleep(DEVICE_POLL_SECONDS)

    async def broadcast_devices(self) -> None:
        await self.manager.broadcast({"type": "devices", **asdict(self.devices)})

    async def broadcast_session(self) -> None:
        await self.manager.broadcast({"type": "session_status", **asdict(self.session)})

    async def broadcast_piano_host(self) -> None:
        await self.manager.broadcast(
            {"type": "piano_host_status", **asdict(self.piano_host.status())}
        )

    async def broadcast_rhythm(self) -> None:
        await self.manager.broadcast({"type": "rhythm_status", **asdict(self.rhythm)})

    async def broadcast_loop_bank(self) -> None:
        await self.manager.broadcast({"type": "loop_bank_status", **asdict(self.loop_bank)})

    def send_live_control(self, action: str, **values: Any) -> bool:
        if self.process is None or self.process.poll() is not None:
            self._threadsafe_broadcast(
                {"type": "error", "message": "Start the AI engine before using live controls."}
            )
            return False
        try:
            LIVE_CONTROL_PATH.parent.mkdir(parents=True, exist_ok=True)
            with LIVE_CONTROL_PATH.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"action": action, **values}) + "\n")
            return True
        except OSError as exc:
            self._threadsafe_broadcast(
                {"type": "error", "message": f"Could not send drum control: {exc}"}
            )
            return False

    def _send_note(self, port_name: str, kind: str, pitch: int, velocity: int) -> bool:
        _, error = self._run_midi_helper(
            "--send-note",
            "--output-port",
            port_name,
            "--message",
            kind,
            "--pitch",
            str(int(pitch)),
            "--velocity",
            str(int(velocity)),
        )
        if error:
            self._threadsafe_broadcast({"type": "error", "message": error})
            return False
        return True

    def panic_all_outputs(self) -> bool:
        ports = {self.config.melody_output_port, self.config.output_port}
        ok = True
        for port_name in sorted(port for port in ports if port):
            _, error = self._run_midi_helper("--panic", "--output-port", port_name, timeout=5.0)
            if error:
                ok = False
                self._threadsafe_broadcast({"type": "error", "message": error})
        self._threadsafe_broadcast(
            {
                "type": "panic_status",
                "ok": ok,
                "message": "所有旋律、鼓和延音已释放" if ok else "部分 MIDI 输出未能释放",
            }
        )
        return ok

    def send_virtual_note(self, kind: str, pitch: int, velocity: int = 100) -> None:
        self.refresh_devices(keep_manual=True)
        port_name = self.devices.virtual_keyboard_output
        if port_name is None:
            self._threadsafe_broadcast(
                {
                    "type": "error",
                    "message": self.devices.midi_error
                    or "Virtual keyboard needs an IAC Python_IN output port.",
                }
            )
            return
        if not self._send_note(port_name, kind, pitch, velocity):
            return
        self._threadsafe_broadcast(
            {
                "type": "visual_note",
                "source": "human",
                "pitch": int(pitch),
                "velocity": int(velocity),
                "event": kind,
                "time": time.time(),
            }
        )

    def send_test_note(self, pitch: int = 60, velocity: int = 92, duration: float = 0.45) -> None:
        self.refresh_devices(keep_manual=True)
        port_name = resolve_port(self.config.melody_output_port, self.devices.outputs)
        if port_name is None:
            self._threadsafe_broadcast(
                {
                    "type": "error",
                    "message": self.devices.midi_error
                    or "No Logic melody MIDI output found. Enable Logic Pro Virtual In.",
                }
            )
            return

        if not self._send_note(port_name, "note_on", pitch, velocity):
            return
        self._threadsafe_broadcast(
            {
                "type": "visual_note",
                "source": "test",
                "pitch": pitch,
                "velocity": velocity,
                "event": "note_on",
                "time": time.time(),
            }
        )

        def note_off() -> None:
            try:
                if not self._send_note(port_name, "note_off", pitch, 0):
                    return
                self._threadsafe_broadcast(
                    {
                        "type": "visual_note",
                        "source": "test",
                        "pitch": pitch,
                        "velocity": 0,
                        "event": "note_off",
                        "time": time.time(),
                    }
                )
            except Exception:
                pass

        threading.Timer(duration, note_off).start()

    def build_live_command(self, input_port: str) -> list[str]:
        cfg = self.config
        # Live Studio uses AMT as the primary generator and plays each accepted
        # event immediately. Motif logic is only an empty-output rescue or a
        # bounded tail completion after AMT has already established the reply.
        cfg.backend = LIVE_BACKEND
        cfg.response_strategy = LIVE_RESPONSE_STRATEGY
        cmd = [
            sys.executable,
            "-u",
            str(LIVE_SCRIPT),
            "--backend",
            cfg.backend,
            "--response-strategy",
            cfg.response_strategy,
            "--model-id",
            cfg.model_id,
            "--aria-model-id",
            cfg.aria_model_id,
            "--offline",
            "--input-port",
            input_port,
            "--output-port",
            cfg.output_port,
            "--melody-output-port",
            cfg.melody_output_port,
            "--startup-test-note",
            "--response-seconds",
            str(cfg.response_seconds),
            "--max-events",
            str(cfg.max_events),
            "--amt-generation-budget",
            str(cfg.amt_generation_budget),
            "--partial-fallback-max-share",
            str(cfg.partial_fallback_max_share),
            "--top-p",
            str(cfg.top_p),
            "--temperature",
            str(cfg.temperature),
            "--latency-mode",
            cfg.latency_mode,
            "--max-underrun-seconds",
            str(cfg.max_underrun_seconds),
            "--musical-control",
            "--fallback-on-empty",
            "--no-duration-match",
            "--live-stop-on-target-notes",
            "--duration-match-min-share",
            "0.80",
            "--duration-match-max-share",
            "1.25",
            "--same-pitch-limit",
            "2",
            "--dominant-pitch-max-share",
            "0.35",
            "--response-length-ratio",
            "1.0",
            "--response-note-ratio",
            "1.0",
            "--live-style",
            "pentatonic",
            "--chord-cluster-window",
            str(cfg.chord_cluster_window),
            "--endpoint-confirm-delay",
            str(cfg.endpoint_confirm_delay),
            "--control-file",
            str(LIVE_CONTROL_PATH),
        ]
        if cfg.min_cutoff is not None:
            cmd.extend(["--min-cutoff", str(cfg.min_cutoff)])
        if cfg.max_cutoff is not None:
            cmd.extend(["--max-cutoff", str(cfg.max_cutoff)])
        # The human Call and the AI response share Logic's melody track. Keep
        # monitoring enabled on macOS too so changing that Logic instrument
        # changes both what the performer plays and what the loop replays.
        if cfg.melody_output_port:
            cmd.append("--monitor-input")
        return cmd

    async def start_session(self, requested_input: Optional[str] = None) -> None:
        if self.process is not None and self.process.poll() is None:
            await self.manager.broadcast(
                {"type": "log", "level": "info", "message": "Live session is already running."}
            )
            return

        self.refresh_devices()
        self.panic_all_outputs()
        input_port = requested_input or self.devices.selected_input
        if input_port is None:
            self.session.last_error = "No MIDI input found. Create Python_IN or connect a keyboard."
            await self.broadcast_session()
            await self.manager.broadcast({"type": "error", "message": self.session.last_error})
            return

        resolved_input = resolve_port(input_port, self.devices.inputs)
        if resolved_input is None:
            self.session.last_error = f"MIDI input not found: {input_port}"
            await self.broadcast_session()
            await self.manager.broadcast({"type": "error", "message": self.session.last_error})
            return

        piano_status = self.piano_host.status()
        resolved_output = choose_audio_output(self.devices.outputs, piano_status.available)
        resolved_melody_output = resolve_port(
            self.config.melody_output_port,
            self.devices.outputs,
        ) or resolved_output
        if resolved_output is None or resolved_melody_output is None:
            self.session.last_error = "No MIDI output found. Start loopMIDI or enable Microsoft GS Wavetable Synth."
            await self.broadcast_session()
            await self.manager.broadcast({"type": "error", "message": self.session.last_error})
            return

        self.config.output_port = resolved_output
        self.config.melody_output_port = resolved_melody_output
        if needs_piano_host(resolved_output):
            piano_status = self.piano_host.launch()
        await self.broadcast_piano_host()
        if needs_piano_host(resolved_output) and not piano_status.available:
            await self.manager.broadcast({"type": "error", "message": piano_status.message})

        cmd = self.build_live_command(resolved_input)
        LIVE_CONTROL_PATH.parent.mkdir(parents=True, exist_ok=True)
        LIVE_CONTROL_PATH.write_text("", encoding="utf-8")
        self.rhythm = RhythmState()
        self.loop_bank = LoopBankState()
        env = os.environ.copy()
        deps = str(PROJECT_DEPS)
        if PROJECT_DEPS.exists():
            env["PYTHONPATH"] = deps + os.pathsep + env.get("PYTHONPATH", "")

        platform_process_options: dict[str, Any]
        if os.name == "nt":
            platform_process_options = {
                "creationflags": subprocess.CREATE_NEW_PROCESS_GROUP,
            }
        else:
            platform_process_options = {"start_new_session": True}

        self.process = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=env,
            **platform_process_options,
        )
        self.session = SessionState(
            running=True,
            pid=self.process.pid,
            input_port=resolved_input,
            output_port=resolved_output,
            virtual_mode=resolved_input == self.devices.virtual_keyboard_output,
            status="starting",
            model_status="loading",
            started_at=time.time(),
        )
        await self.broadcast_session()
        await self.manager.broadcast(
            {
                "type": "log",
                "level": "info",
                "message": f"Started live session pid={self.process.pid} input={resolved_input}",
            }
        )
        threading.Thread(target=self._read_process_output, daemon=True).start()

    async def stop_session(self) -> None:
        process = self.process
        self.process = None
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=4)
            except subprocess.TimeoutExpired:
                process.kill()
        self.panic_all_outputs()
        self.session.running = False
        self.session.pid = None
        self.session.status = "stopped"
        self.session.model_status = "not loaded"
        self.rhythm = RhythmState()
        self.loop_bank = LoopBankState()
        await self.broadcast_session()
        await self.broadcast_rhythm()
        await self.broadcast_loop_bank()

    def _read_process_output(self) -> None:
        assert self.process is not None and self.process.stdout is not None
        LIVE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        log_file = LIVE_LOG_PATH.open("a", encoding="utf-8")
        for raw_line in iter(self.process.stdout.readline, ""):
            line = raw_line.strip()
            if not line:
                continue
            try:
                log_file.write(line + "\n")
                log_file.flush()
                self._parse_live_log(line)
                self._threadsafe_broadcast({"type": "log", "level": "live", "message": line})
            except Exception as exc:
                self._threadsafe_broadcast(
                    {"type": "error", "message": f"Failed to parse live log: {exc}"}
                )
        log_file.close()

        code = self.process.poll() if self.process is not None else None
        self.session.running = False
        self.session.pid = None
        self.session.status = "stopped" if code in (0, None) else "error"
        if code not in (0, None):
            self.session.last_error = f"live_call_response.py exited with code {code}"
        self._threadsafe_broadcast({"type": "session_status", **asdict(self.session)})

    def _parse_live_log(self, line: str) -> None:
        playback_note = re.search(
            r"\[(ai|drum|loop)_out\]\s+(note_on|note_off)\s+pitch=\s*(\d+)\s+velocity=\s*(\d+)",
            line,
        )
        if playback_note:
            bus, event, pitch, velocity = playback_note.groups()
            self._threadsafe_broadcast(
                {
                    "type": "playback_note",
                    "bus": bus,
                    "event": event,
                    "pitch": int(pitch),
                    "velocity": int(velocity),
                    "time": time.time(),
                }
            )
            return

        if line.startswith("[rhythm] "):
            try:
                payload = json.loads(line[len("[rhythm] "):])
                pattern = payload.get("pattern", [])
                self.rhythm = RhythmState(
                    state=str(payload.get("state", "idle")),
                    tap_count=int(payload.get("tap_count", 0)),
                    bpm=float(payload.get("bpm", 100.0)),
                    confidence=float(payload.get("confidence", 0.0)),
                    bars=max(1, int(payload.get("bars", 1))),
                    steps_per_bar=max(1, int(payload.get("steps_per_bar", 16))),
                    pattern=[int(step) for step in pattern] if isinstance(pattern, list) else [],
                    loop_seconds=float(payload.get("loop_seconds", 0.0)),
                    learning=bool(payload.get("learning", False)),
                    playing=bool(payload.get("playing", False)),
                    replacing=bool(payload.get("replacing", False)),
                    stopping=bool(payload.get("stopping", False)),
                    saved_slots=[str(slot) for slot in payload.get("saved_slots", [])],
                    current_slot=payload.get("current_slot"),
                    queued_slot=payload.get("queued_slot"),
                )
                self._threadsafe_broadcast(
                    {"type": "rhythm_status", **asdict(self.rhythm)}
                )
            except (TypeError, ValueError, json.JSONDecodeError):
                pass
            return
        if line.startswith("[loop] "):
            try:
                payload = json.loads(line[len("[loop] "):])
                slots = [LoopSlotState(**slot) for slot in payload.get("slots", [])]
                if len(slots) == 4:
                    self.loop_bank = LoopBankState(
                        mode=str(payload.get("mode", "response")),
                        latest_ready=bool(payload.get("latest_ready", False)),
                        slots=slots,
                    )
                    self._threadsafe_broadcast(
                        {"type": "loop_bank_status", **asdict(self.loop_bank)}
                    )
            except (TypeError, ValueError, json.JSONDecodeError):
                pass
            return
        note_match = re.search(r"\[note_on\]\s+pitch=\s*(\d+)\s+velocity=\s*(\d+).*cutoff=([0-9.]+)s", line)
        if note_match:
            pitch, velocity, cutoff = note_match.groups()
            self.session.status = "listening"
            self._round_data["endpoint_cutoff"] = float(cutoff)
            self._threadsafe_broadcast({"type": "session_status", **asdict(self.session)})
            self._threadsafe_broadcast(
                {
                    "type": "visual_note",
                    "source": "human",
                    "pitch": int(pitch),
                    "velocity": int(velocity),
                    "event": "note_on",
                    "time": time.time(),
                }
            )
            return

        note_off_match = re.search(r"\[note_off\]\s+pitch=\s*(\d+)\s+velocity=\s*(\d+)", line)
        if note_off_match:
            pitch, velocity = note_off_match.groups()
            self._threadsafe_broadcast(
                {
                    "type": "visual_note",
                    "source": "human",
                    "pitch": int(pitch),
                    "velocity": int(velocity),
                    "event": "note_off",
                    "time": time.time(),
                }
            )
            return

        if line.startswith("[candidate]"):
            self.session.status = "candidate"
            self._threadsafe_broadcast({"type": "session_status", **asdict(self.session)})
            self._broadcast_round_state("candidate", {"message": line})
            return

        endpoint_match = re.search(r"\[endpoint\].*phrase_len=(\d+).*cutoff=([0-9.]+)s", line)
        if endpoint_match:
            notes, cutoff = endpoint_match.groups()
            self.session.status = "endpoint"
            self._round_data.update({"call_notes": int(notes), "endpoint_cutoff": float(cutoff)})
            self._threadsafe_broadcast({"type": "session_status", **asdict(self.session)})
            self._broadcast_round_state("endpoint", self._round_data)
            return

        round_match = re.search(r"\[round\s+(\d+)\]", line)
        if round_match:
            self.session.round_id = int(round_match.group(1))

        if "loading model" in line.lower() or "loading" in line.lower() and "model" in line.lower():
            self.session.model_status = "loading"
            self._threadsafe_broadcast({"type": "session_status", **asdict(self.session)})

        model_ready = False
        if "device=cuda" in line.lower() or "cuda" in line.lower():
            self.session.model_status = "cuda ready"
            model_ready = True
        elif "device=mps" in line.lower():
            self.session.model_status = "mps ready"
            model_ready = True
        elif "[startup]" in line and "device=cpu" in line.lower():
            self.session.model_status = "cpu ready"
            model_ready = True

        if model_ready:
            self._threadsafe_broadcast({"type": "session_status", **asdict(self.session)})

        if line.startswith("[listening]"):
            self.session.status = "listening"
            self._threadsafe_broadcast({"type": "session_status", **asdict(self.session)})
            return

        if "endpoint -> generating" in line:
            self.session.status = "generating"
            self._threadsafe_broadcast({"type": "session_status", **asdict(self.session)})
            self._broadcast_round_state("generating", {"round_id": self.session.round_id})
            return

        analysis = re.search(r"call_duration=([0-9.]+)s call_notes=(\d+).*contour=([A-Za-z_]+)", line)
        if analysis:
            duration, notes, contour = analysis.groups()
            self._round_data.update(
                {"call_duration": float(duration), "call_notes": int(notes), "contour": contour}
            )
            self._broadcast_round_state("analysis", self._round_data)
            return

        plan = re.search(r"target_seconds=([0-9.]+)s target_notes=(\d+).*prompt_notes=(\d+)/(\d+)", line)
        if plan:
            seconds, notes, prompt, total = plan.groups()
            self._round_data.update(
                {
                    "target_seconds": float(seconds),
                    "target_notes": int(notes),
                    "prompt_notes": int(prompt),
                    "total_notes": int(total),
                }
            )
            self._broadcast_round_state("plan", self._round_data)
            return

        first_latency = re.search(r"first_event_latency=([0-9.]+)s", line)
        if first_latency:
            self._round_data["first_event_latency"] = float(first_latency.group(1))
            self._broadcast_round_state("first_event", self._round_data)
            return

        generated_event = re.search(
            r"(?:queued|sampled) event #\s*(\d+):\s*tick=(-?\d+)\s+dur=(-?\d+)\s+pitch=(-?\d+)",
            line,
            re.IGNORECASE,
        )
        if generated_event:
            number, tick, duration, pitch = generated_event.groups()
            pitch_value = max(0, min(127, int(pitch)))
            self.session.status = "playback_pending"
            self._round_data["generated_events"] = int(number)
            self._threadsafe_broadcast({"type": "session_status", **asdict(self.session)})
            self._threadsafe_broadcast(
                {
                    "type": "visual_note",
                    "source": "ai",
                    "pitch": pitch_value,
                    "velocity": 86,
                    "event": "note_on",
                    "tick": int(tick),
                    "duration_ticks": int(duration),
                    "time": time.time(),
                }
            )
            self._broadcast_round_state("generated", self._round_data)
            return

        if "[buffering]" in line:
            if "total_response_cycle=" in line:
                self.session.status = "listening"
                self._threadsafe_broadcast({"type": "session_status", **asdict(self.session)})
                return
            self.session.status = "buffering"
            self._threadsafe_broadcast({"type": "session_status", **asdict(self.session)})
            return

        if "[playback] starting" in line:
            self.session.status = "playback"
            self._threadsafe_broadcast({"type": "session_status", **asdict(self.session)})
            self._broadcast_round_state("playback", self._round_data)
            return

        underrun = re.search(r"buffer underrun #(\d+)", line)
        if underrun:
            self._round_data["buffer_underruns"] = int(underrun.group(1))
            self._broadcast_round_state("underrun", self._round_data)
            return

        done = re.search(r"done; buffer_underruns=(\d+)", line)
        if done:
            self.session.status = "done"
            self._round_data["buffer_underruns"] = int(done.group(1))
            self._threadsafe_broadcast({"type": "session_status", **asdict(self.session)})
            self._broadcast_round_state("done", self._round_data)
            return

        metrics = re.search(
            r"\[metrics\] round=(\d+) status=([a-zA-Z_]+) first_event=([^ ]+) total_response=([^ ]+) underruns=(\d+)",
            line,
        )
        if metrics:
            round_id, status, first_event, total_response, underruns = metrics.groups()
            payload = {
                "type": "metrics",
                "round_id": int(round_id),
                "status": status,
                "first_event": first_event,
                "total_response": total_response,
                "underruns": int(underruns),
            }
            self._threadsafe_broadcast(payload)

        if "error" in line.lower() or "traceback" in line.lower():
            self.session.status = "error"
            self.session.last_error = line
            self._threadsafe_broadcast({"type": "session_status", **asdict(self.session)})
            self._threadsafe_broadcast({"type": "error", "message": line})

    def _broadcast_round_state(self, state: str, data: dict[str, Any]) -> None:
        self._threadsafe_broadcast(
            {
                "type": "round_state",
                "state": state,
                "round_id": self.session.round_id,
                "data": data,
            }
        )

    async def handle_payload(self, payload: dict[str, Any]) -> None:
        kind = payload.get("type")
        if kind == "refresh_devices":
            self.refresh_devices(force=True)
            await self.broadcast_devices()
        elif kind == "start_session":
            await self.start_session(payload.get("input_port"))
        elif kind == "stop_session":
            await self.stop_session()
        elif kind == "panic_all":
            if self.process is not None and self.process.poll() is None:
                self.send_live_control(kind)
            self.panic_all_outputs()
        elif kind in {
            "rhythm_learn_start",
            "rhythm_learn_finish",
            "rhythm_stop",
            "rhythm_stop_now",
            "rhythm_variation",
            "drum_record_start",
            "drum_record_finish",
            "drum_stop",
            "drum_stop_now",
        } or (
            isinstance(kind, str)
            and (
                kind.startswith("rhythm_save_")
                or kind.startswith("rhythm_load_")
                or kind.startswith("loop_save_")
                or kind.startswith("loop_toggle_")
                or kind.startswith("loop_stop_")
                or kind in {"loop_set_mode_response", "loop_set_mode_call_response"}
            )
        ):
            self.send_live_control(kind)
        elif kind == "rhythm_tap":
            self.send_live_control(
                kind,
                velocity=max(1, min(127, int(payload.get("velocity", 96)))),
                timestamp=time.monotonic(),
            )
        elif kind == "test_output":
            self.piano_host.launch()
            await self.broadcast_piano_host()
            self.send_test_note()
        elif kind in {"note_on", "note_off"}:
            self.send_virtual_note(
                kind,
                int(payload.get("pitch", 60)),
                int(payload.get("velocity", 100)),
            )
        elif kind == "set_params":
            params = payload.get("params", {})
            for key, value in params.items():
                if hasattr(self.config, key):
                    if key == "backend" and value != LIVE_BACKEND:
                        await self.manager.broadcast(
                            {
                                "type": "error",
                                "message": "现场版本固定使用流式 AMT 主生成器，不能切换生成后端。",
                            }
                        )
                        continue
                    if key == "response_strategy" and value != LIVE_RESPONSE_STRATEGY:
                        await self.manager.broadcast(
                            {
                                "type": "error",
                                "message": "Motif 只用于空输出或有限尾部补齐，不能替代 AMT 主回应。",
                            }
                        )
                        continue
                    setattr(self.config, key, value)
            await self.manager.broadcast({"type": "config", **asdict(self.config)})


controller = LiveStudioController()
app = FastAPI(title="MFP Live Studio")


if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.on_event("startup")
async def on_startup() -> None:
    controller.attach_loop(asyncio.get_running_loop())
    controller.refresh_devices(force=True)
    asyncio.create_task(controller.poll_devices_forever())


@app.get("/")
async def index() -> HTMLResponse:
    path = STATIC_DIR / "index.html"
    if not path.exists():
        return HTMLResponse("<h1>MFP Live Studio static files are missing.</h1>", status_code=500)
    return HTMLResponse(path.read_text(encoding="utf-8"))


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await controller.manager.connect(websocket)
    await controller.broadcast_devices()
    await controller.broadcast_session()
    await controller.broadcast_piano_host()
    await controller.broadcast_rhythm()
    await controller.broadcast_loop_bank()
    await controller.manager.broadcast({"type": "config", **asdict(controller.config)})
    try:
        while True:
            data = await websocket.receive_text()
            await controller.handle_payload(json.loads(data))
    except WebSocketDisconnect:
        controller.manager.disconnect(websocket)


def run(host: str = "127.0.0.1", port: int = 8000) -> None:
    uvicorn.run("interface_backend:app", host=host, port=port, reload=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="MFP Live Studio web interface")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    run(args.host, args.port)


if __name__ == "__main__":
    main()
