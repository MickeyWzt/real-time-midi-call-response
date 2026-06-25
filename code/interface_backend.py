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
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional


ROOT = Path(__file__).resolve().parents[1]
PROJECT_DEPS = ROOT / ".python_deps"
if PROJECT_DEPS.exists():
    sys.path.insert(0, str(PROJECT_DEPS))

import mido
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from vst_host_manager import get_piano_host_manager


STATIC_DIR = ROOT / "code" / "static"
LIVE_SCRIPT = ROOT / "code" / "live_call_response.py"
LIVE_LOG_PATH = ROOT / "logs" / "mfp_live_studio_live.log"
DEFAULT_OUTPUT_PORT = "Python_OUT"
DEFAULT_MODEL_ID = "stanford-crfm/music-small-800k"
DEFAULT_ARIA_MODEL_ID = str(ROOT / "model_weights" / "aria-medium-gen")
DEVICE_POLL_SECONDS = 1.0

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
    if piano_host_available:
        loopback_output = resolve_port(DEFAULT_OUTPUT_PORT, outputs)
        if loopback_output:
            return loopback_output

    return (
        resolve_port("Microsoft GS", outputs)
        or resolve_port("Wavetable", outputs)
        or (outputs[0] if outputs else None)
    )


def needs_piano_host(output_port: str) -> bool:
    return DEFAULT_OUTPUT_PORT.casefold() in output_port.casefold()


@dataclass
class StudioConfig:
    backend: str = "amt"
    model_id: str = DEFAULT_MODEL_ID
    aria_model_id: str = DEFAULT_ARIA_MODEL_ID
    response_seconds: float = 8.0
    max_events: int = 16
    top_p: float = 0.98
    temperature: float = 0.9
    latency_mode: str = "fast"
    max_underrun_seconds: float = 1.5
    min_cutoff: float = 0.45
    chord_cluster_window: float = 0.08
    endpoint_confirm_delay: float = 0.15
    output_port: str = DEFAULT_OUTPUT_PORT


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
        self.process: Optional[subprocess.Popen[str]] = None
        self.manager = ConnectionManager()
        self.piano_host = get_piano_host_manager()
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.virtual_out: Optional[mido.ports.BaseOutput] = None
        self.output_out: Optional[mido.ports.BaseOutput] = None
        self._last_device_signature: Optional[tuple[tuple[str, ...], tuple[str, ...]]] = None
        self._round_data: dict[str, Any] = {}

    def attach_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self.loop = loop

    def _threadsafe_broadcast(self, payload: dict[str, Any]) -> None:
        if self.loop is None:
            return
        self.loop.call_soon_threadsafe(
            lambda: asyncio.create_task(self.manager.broadcast(payload))
        )

    def refresh_devices(self, keep_manual: bool = False) -> DeviceState:
        inputs = mido.get_input_names()
        outputs = mido.get_output_names()
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
        )
        return self.devices

    async def poll_devices_forever(self) -> None:
        while True:
            try:
                old_input = self.devices.selected_input
                devices = self.refresh_devices(keep_manual=self.session.running)
                signature = (tuple(devices.inputs), tuple(devices.outputs))
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

    def _open_virtual_out(self) -> Optional[mido.ports.BaseOutput]:
        self.refresh_devices(keep_manual=True)
        port_name = self.devices.virtual_keyboard_output
        if port_name is None:
            return None
        if self.virtual_out is not None and not self.virtual_out.closed:
            return self.virtual_out
        self.virtual_out = mido.open_output(port_name)
        return self.virtual_out

    def _open_output_out(self) -> Optional[mido.ports.BaseOutput]:
        self.refresh_devices(keep_manual=True)
        port_name = self.devices.selected_output
        if port_name is None:
            return None
        if self.output_out is not None and not self.output_out.closed:
            return self.output_out
        self.output_out = mido.open_output(port_name)
        return self.output_out

    def send_virtual_note(self, kind: str, pitch: int, velocity: int = 100) -> None:
        outport = self._open_virtual_out()
        if outport is None:
            self._threadsafe_broadcast(
                {
                    "type": "error",
                    "message": "Virtual keyboard needs a loopMIDI Python_IN output port.",
                }
            )
            return
        outport.send(mido.Message(kind, note=int(pitch), velocity=int(velocity)))
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
        outport = self._open_output_out()
        if outport is None:
            self._threadsafe_broadcast(
                {
                    "type": "error",
                    "message": "No Python_OUT MIDI output found. Start loopMIDI and create Python_OUT.",
                }
            )
            return

        outport.send(mido.Message("note_on", note=pitch, velocity=velocity))
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
                outport.send(mido.Message("note_off", note=pitch, velocity=0))
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
        cmd = [
            sys.executable,
            "-u",
            str(LIVE_SCRIPT),
            "--backend",
            cfg.backend,
            "--model-id",
            cfg.model_id,
            "--aria-model-id",
            cfg.aria_model_id,
            "--offline",
            "--input-port",
            input_port,
            "--output-port",
            cfg.output_port,
            "--startup-test-note",
            "--response-seconds",
            str(cfg.response_seconds),
            "--max-events",
            str(cfg.max_events),
            "--top-p",
            str(cfg.top_p),
            "--temperature",
            str(cfg.temperature),
            "--latency-mode",
            cfg.latency_mode,
            "--max-underrun-seconds",
            str(cfg.max_underrun_seconds),
            "--musical-control",
            "--no-speculative-preload",
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
            "--min-cutoff",
            str(cfg.min_cutoff),
            "--chord-cluster-window",
            str(cfg.chord_cluster_window),
            "--endpoint-confirm-delay",
            str(cfg.endpoint_confirm_delay),
        ]
        if needs_piano_host(cfg.output_port):
            cmd.append("--monitor-input")
        return cmd

    async def start_session(self, requested_input: Optional[str] = None) -> None:
        if self.process is not None and self.process.poll() is None:
            await self.manager.broadcast(
                {"type": "log", "level": "info", "message": "Live session is already running."}
            )
            return

        self.refresh_devices()
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
        if resolved_output is None:
            self.session.last_error = "No MIDI output found. Start loopMIDI or enable Microsoft GS Wavetable Synth."
            await self.broadcast_session()
            await self.manager.broadcast({"type": "error", "message": self.session.last_error})
            return

        self.config.output_port = resolved_output
        if needs_piano_host(resolved_output):
            piano_status = self.piano_host.launch()
        await self.broadcast_piano_host()
        if needs_piano_host(resolved_output) and not piano_status.available:
            await self.manager.broadcast({"type": "error", "message": piano_status.message})

        cmd = self.build_live_command(resolved_input)
        env = os.environ.copy()
        deps = str(PROJECT_DEPS)
        if PROJECT_DEPS.exists():
            env["PYTHONPATH"] = deps + os.pathsep + env.get("PYTHONPATH", "")

        self.process = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
            env=env,
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
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=4)
            except subprocess.TimeoutExpired:
                self.process.kill()
        self.process = None
        self.session.running = False
        self.session.pid = None
        self.session.status = "stopped"
        self.session.model_status = "not loaded"
        await self.broadcast_session()

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

        if "device=cuda" in line.lower() or "cuda" in line.lower():
            self.session.model_status = "cuda ready"
            self._threadsafe_broadcast({"type": "session_status", **asdict(self.session)})

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
            self.refresh_devices()
            await self.broadcast_devices()
        elif kind == "start_session":
            await self.start_session(payload.get("input_port"))
        elif kind == "stop_session":
            await self.stop_session()
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
                    if key == "backend" and value not in {"amt", "aria"}:
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
    controller.refresh_devices()
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
