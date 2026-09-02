"""Crash-isolated CoreMIDI operations for the Live Studio web service."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any


def emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False))


def resolve_port(requested: str, names: list[str]) -> str:
    if requested in names:
        return requested
    lowered = requested.casefold()
    for name in names:
        if lowered in name.casefold():
            return name
    raise RuntimeError(f"MIDI output not found: {requested}")


def list_ports(mido: Any) -> None:
    emit(
        {
            "ok": True,
            "inputs": list(mido.get_input_names()),
            "outputs": list(mido.get_output_names()),
        }
    )


def send_note(mido: Any, args: argparse.Namespace) -> None:
    output_port = resolve_port(args.output_port, list(mido.get_output_names()))
    with mido.open_output(output_port) as outport:
        outport.send(
            mido.Message(
                args.message,
                note=args.pitch,
                velocity=args.velocity,
                channel=0,
            )
        )
    emit({"ok": True, "output": output_port})


def panic(mido: Any, args: argparse.Namespace) -> None:
    output_port = resolve_port(args.output_port, list(mido.get_output_names()))
    with mido.open_output(output_port) as outport:
        for channel in range(16):
            outport.send(mido.Message("control_change", channel=channel, control=64, value=0))
            outport.send(mido.Message("control_change", channel=channel, control=123, value=0))
            outport.send(mido.Message("control_change", channel=channel, control=120, value=0))
            for note in range(128):
                outport.send(mido.Message("note_off", channel=channel, note=note, velocity=0))
    emit({"ok": True, "output": output_port, "panic": True})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--list-ports", action="store_true")
    mode.add_argument("--send-note", action="store_true")
    mode.add_argument("--panic", action="store_true")
    parser.add_argument("--output-port")
    parser.add_argument("--message", choices=["note_on", "note_off"])
    parser.add_argument("--pitch", type=int, default=60)
    parser.add_argument("--velocity", type=int, default=0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        import mido

        if args.list_ports:
            list_ports(mido)
        elif args.panic:
            if not args.output_port:
                raise RuntimeError("--output-port is required with --panic")
            panic(mido, args)
        else:
            if not args.output_port or not args.message:
                raise RuntimeError("--output-port and --message are required with --send-note")
            send_note(mido, args)
    except Exception as exc:
        emit({"ok": False, "error": str(exc)})
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
