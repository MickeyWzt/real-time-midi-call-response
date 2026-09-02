from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import ANY, AsyncMock, Mock, patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from interface_backend import LiveStudioController, StudioConfig
from vst_host_manager import PianoHostManager


class MidiIsolationTests(unittest.TestCase):
    def test_live_interface_defaults_to_streaming_amt(self) -> None:
        config = StudioConfig()

        self.assertEqual(config.backend, "amt")
        self.assertEqual(config.response_strategy, "streaming_amt")
        self.assertEqual(config.amt_generation_budget, 4.0)
        self.assertGreater(config.max_underrun_seconds, config.amt_generation_budget)
        self.assertEqual(config.partial_fallback_max_share, 0.5)

    def test_live_command_locks_streaming_amt_and_live_controls(self) -> None:
        controller = LiveStudioController()
        controller.config.backend = "aria"
        controller.config.response_strategy = "motif"

        command = controller.build_live_command("Minilab3 MIDI")

        self.assertEqual(command[command.index("--backend") + 1], "amt")
        self.assertEqual(
            command[command.index("--response-strategy") + 1],
            "streaming_amt",
        )
        for flag in (
            "--musical-control",
            "--fallback-on-empty",
            "--no-duration-match",
            "--live-stop-on-target-notes",
            "--same-pitch-limit",
            "--dominant-pitch-max-share",
            "--response-length-ratio",
            "--response-note-ratio",
            "--live-style",
            "--partial-fallback-max-share",
        ):
            self.assertIn(flag, command)
        self.assertNotIn("--speculative-preload", command)

    def test_runtime_rejects_motif_as_normal_response_strategy(self) -> None:
        controller = LiveStudioController()
        controller.manager.broadcast = AsyncMock()

        asyncio.run(
            controller.handle_payload(
                {"type": "set_params", "params": {"response_strategy": "motif"}}
            )
        )

        self.assertEqual(controller.config.response_strategy, "streaming_amt")
        messages = [call.args[0] for call in controller.manager.broadcast.await_args_list]
        self.assertTrue(any(message.get("type") == "error" for message in messages))

    @unittest.skipIf(os.name == "nt", "macOS/Linux behavior")
    def test_external_daw_is_not_reported_as_missing_piano_host(self) -> None:
        status = PianoHostManager().status()
        self.assertTrue(status.available)
        self.assertIn("External DAW", status.message)

    def test_refresh_uses_isolated_helper_results(self) -> None:
        completed = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=json.dumps(
                {
                    "ok": True,
                    "inputs": ["IAC Driver Python_IN"],
                    "outputs": ["IAC Driver Python_IN", "IAC Driver Python_OUT"],
                }
            ),
            stderr="",
        )
        controller = LiveStudioController()
        with patch("interface_backend.subprocess.run", return_value=completed):
            devices = controller.refresh_devices(force=True)

        self.assertEqual(devices.selected_input, "IAC Driver Python_IN")
        self.assertEqual(devices.selected_output, "IAC Driver Python_OUT")
        self.assertIsNone(devices.midi_error)

    def test_helper_crash_does_not_crash_web_controller(self) -> None:
        completed = subprocess.CompletedProcess(
            args=[],
            returncode=-6,
            stdout="",
            stderr="CoreMIDI client is invalid",
        )
        controller = LiveStudioController()
        with patch("interface_backend.subprocess.run", return_value=completed):
            devices = controller.refresh_devices(force=True)

        self.assertEqual(devices.inputs, [])
        self.assertIn("CoreMIDI helper exited -6", devices.midi_error or "")

    def test_completed_response_returns_engine_to_listening(self) -> None:
        controller = LiveStudioController()
        controller._parse_live_log("[buffering] total_response_cycle=3.744s")

        self.assertEqual(controller.session.status, "listening")

    def test_live_command_uses_latest_paper_endpoint_settings(self) -> None:
        command = LiveStudioController().build_live_command("IAC Driver Python_IN")

        self.assertNotIn("--min-cutoff", command)
        self.assertNotIn("--max-cutoff", command)
        self.assertEqual(command[command.index("--endpoint-confirm-delay") + 1], "0.15")

    def test_live_command_uses_display_latency_budget(self) -> None:
        controller = LiveStudioController()
        command = controller.build_live_command("IAC Driver Python_IN")

        self.assertEqual(controller.config.response_seconds, 3.0)
        self.assertEqual(command[command.index("--amt-generation-budget") + 1], "4.0")

    def test_live_command_monitors_human_input_into_logic(self) -> None:
        command = LiveStudioController().build_live_command("Minilab3 MIDI")

        self.assertIn("--monitor-input", command)

    def test_panic_releases_both_logic_outputs(self) -> None:
        controller = LiveStudioController()
        controller._run_midi_helper = Mock(return_value=({"ok": True}, None))
        controller._threadsafe_broadcast = Mock()

        self.assertTrue(controller.panic_all_outputs())

        calls = {tuple(call.args) for call in controller._run_midi_helper.call_args_list}
        self.assertIn(("--panic", "--output-port", "Logic Pro Virtual In"), calls)
        self.assertIn(("--panic", "--output-port", "Python_OUT"), calls)

    def test_playback_note_is_broadcast_to_computer_test_audio(self) -> None:
        controller = LiveStudioController()
        controller._threadsafe_broadcast = Mock()
        controller._parse_live_log("[drum_out] note_on pitch= 36 velocity= 96")

        controller._threadsafe_broadcast.assert_called_once_with(
            {
                "type": "playback_note",
                "bus": "drum",
                "event": "note_on",
                "pitch": 36,
                "velocity": 96,
                "time": ANY,
            }
        )

    def test_logic_sound_test_uses_melody_track_not_drum_track(self) -> None:
        controller = LiveStudioController()
        controller.devices.outputs = ["IAC Driver Python_OUT", "Logic Pro Virtual In"]
        controller.refresh_devices = Mock(return_value=controller.devices)
        controller._send_note = Mock(return_value=True)
        controller._threadsafe_broadcast = Mock()

        with patch("interface_backend.threading.Timer"):
            controller.send_test_note()

        controller._send_note.assert_called_once_with(
            "Logic Pro Virtual In", "note_on", 60, 92
        )

    def test_rhythm_status_json_is_forwarded_to_interface(self) -> None:
        controller = LiveStudioController()
        controller._threadsafe_broadcast = Mock()
        controller._parse_live_log(
            '[rhythm] {"state":"running","tap_count":0,"bpm":120.0,'
            '"confidence":0.94,"bars":1,"steps_per_bar":16,'
            '"pattern":[0,4,8,12],"loop_seconds":2.0,"learning":false,'
            '"playing":true,"replacing":false,"stopping":false}'
        )

        self.assertEqual(controller.rhythm.bpm, 120.0)
        self.assertEqual(controller.rhythm.pattern, [0, 4, 8, 12])
        controller._threadsafe_broadcast.assert_called_once_with(
            {
                "type": "rhythm_status",
                "state": "running",
                "tap_count": 0,
                "bpm": 120.0,
                "confidence": 0.94,
                "bars": 1,
                "steps_per_bar": 16,
                "pattern": [0, 4, 8, 12],
                "loop_seconds": 2.0,
                "learning": False,
                "playing": True,
                "replacing": False,
                "stopping": False,
                "saved_slots": [],
                "current_slot": None,
                "queued_slot": None,
            }
        )


if __name__ == "__main__":
    unittest.main()
