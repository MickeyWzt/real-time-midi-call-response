from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


ROOT = Path(__file__).resolve().parents[1]
PIANO_DIR = ROOT / "4fpiano-win"
PIANO_HOST_EXE = PIANO_DIR / "4Front Piano x64.exe"
PIANO_PLUGIN_DLL = PIANO_DIR / "4Front Piano x64.dll"


@dataclass
class PianoHostStatus:
    available: bool
    running: bool
    exe_path: str
    message: str


class PianoHostManager:
    def __init__(self) -> None:
        self._process: Optional[subprocess.Popen] = None

    def status(self) -> PianoHostStatus:
        if not PIANO_HOST_EXE.exists():
            return PianoHostStatus(
                available=False,
                running=False,
                exe_path=str(PIANO_HOST_EXE),
                message="4Front Piano host not found. Check D:\\Mickey\\MFP\\4fpiano-win.",
            )

        running = self._process is not None and self._process.poll() is None
        if running:
            message = "4Front Piano host is running."
        elif not PIANO_PLUGIN_DLL.exists():
            message = "Host found, but 4Front Piano x64.dll is missing."
        else:
            message = "4Front Piano host is ready."

        return PianoHostStatus(
            available=True,
            running=running,
            exe_path=str(PIANO_HOST_EXE),
            message=message,
        )

    def launch(self) -> PianoHostStatus:
        current = self.status()
        if not current.available:
            return current
        if current.running:
            return current

        self._process = subprocess.Popen(
            [str(PIANO_HOST_EXE)],
            cwd=str(PIANO_DIR),
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
        )
        return self.status()

    def stop(self) -> PianoHostStatus:
        if self._process is not None and self._process.poll() is None:
            self._process.terminate()
        self._process = None
        return self.status()


_manager = PianoHostManager()


def launch_vst_host() -> subprocess.Popen | None:
    status = _manager.launch()
    return _manager._process if status.running else None


def get_piano_host_manager() -> PianoHostManager:
    return _manager
