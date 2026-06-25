"""
注入灵魂: one-shot Transformer call-and-response demo.

This script wires jthickstun/anticipation into our local system for a
non-realtime generation pass:

1. Read or create a short human Call MIDI file.
2. Convert it to the Anticipatory Music Transformer event representation.
3. Load the pretrained Transformer: stanford-crfm/music-medium-800k.
4. Generate a short Response after the Call using the author's generate().
5. Export the combined Call + Response to MIDI.

Example:

    python code/transformer_call_response_once.py --make-demo-input
    python code/transformer_call_response_once.py --offline

If the model cache is incomplete, run without --offline once to let
HuggingFace finish downloading the weights into D:\\Mickey\\MFP\\hf_cache.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Tuple


ROOT = Path(__file__).resolve().parents[1]
PROJECT_DEPS = ROOT / ".python_deps"
ANTICIPATION_REPO = ROOT / "code" / "anticipation"
HF_CACHE = ROOT / "hf_cache"
HF_HUB_CACHE = HF_CACHE / "hub"
DEFAULT_INPUT = ROOT / "code" / "demo_call.mid"
DEFAULT_OUTPUT = ROOT / "code" / "transformer_call_response.mid"
MODEL_ID = "stanford-crfm/music-medium-800k"

if PROJECT_DEPS.exists():
    sys.path.insert(0, str(PROJECT_DEPS))
sys.path.insert(0, str(ANTICIPATION_REPO))

os.environ.setdefault("HF_HOME", str(HF_CACHE))
os.environ.setdefault("HF_HUB_CACHE", str(HF_HUB_CACHE))
os.environ.setdefault("HF_ENDPOINT", "https://huggingface.co")


def require_runtime() -> None:
    missing = []
    for module_name in ("torch", "transformers", "mido", "anticipation"):
        try:
            __import__(module_name)
        except ImportError:
            missing.append(module_name)
    if missing:
        raise SystemExit(
            "Missing runtime modules: "
            + ", ".join(missing)
            + ". Install project deps first, e.g. pip install --target .python_deps torch transformers mido python-rtmidi."
        )


def make_demo_call_midi(path: Path) -> None:
    import mido

    path.parent.mkdir(parents=True, exist_ok=True)
    mid = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(100), time=0))
    track.append(mido.MetaMessage("time_signature", numerator=4, denominator=4, time=0))
    track.append(mido.Message("program_change", program=0, channel=0, time=0))

    # Two bars of 4/4 at 100 BPM, shaped as a longer C-pentatonic Call.
    notes: List[Tuple[int, int, int]] = [
        # pitch, absolute_onset_ticks, duration_ticks
        (60, 0, 240),
        (62, 240, 240),
        (64, 480, 360),
        (67, 900, 180),
        (69, 1080, 300),
        (67, 1440, 240),
        (64, 1680, 240),
        (62, 1920, 300),
        (60, 2280, 180),
        (62, 2460, 180),
        (64, 2640, 240),
        (67, 2880, 360),
        (69, 3300, 180),
        (67, 3480, 300),
    ]
    last_tick = 0
    for pitch, onset, duration in notes:
        delta = max(0, onset - last_tick)
        track.append(mido.Message("note_on", note=pitch, velocity=90, channel=0, time=delta))
        track.append(mido.Message("note_off", note=pitch, velocity=0, channel=0, time=duration))
        last_tick = onset + duration

    track.append(mido.MetaMessage("end_of_track", time=max(0, 3840 - last_tick)))
    mid.save(path)


def model_cache_snapshot() -> Tuple[bool, List[str]]:
    model_dir = HF_HUB_CACHE / "models--stanford-crfm--music-medium-800k"
    files = []
    if model_dir.exists():
        files = [str(p.relative_to(model_dir)) for p in model_dir.rglob("*") if p.is_file()]
    has_config = any(name.endswith("config.json") for name in files)
    has_weights = any(
        name.endswith(("pytorch_model.bin", "model.safetensors"))
        and ".no_exist" not in name
        and not name.endswith(".incomplete")
        for name in files
    )
    return has_config and has_weights, files


def print_cache_status() -> None:
    complete, files = model_cache_snapshot()
    print(f"[cache] model id       : {MODEL_ID}")
    print(f"[cache] hub cache dir  : {HF_HUB_CACHE}")
    print(f"[cache] complete       : {complete}")
    if not files:
        print("[cache] files          : none")
        return
    print("[cache] files:")
    for name in files:
        print(f"  - {name}")


def local_snapshot_dir() -> Optional[Path]:
    model_dir = HF_HUB_CACHE / "models--stanford-crfm--music-medium-800k" / "snapshots"
    if not model_dir.exists():
        return None
    for snapshot in model_dir.iterdir():
        if (snapshot / "config.json").exists() and (
            (snapshot / "pytorch_model.bin").exists()
            or (snapshot / "model.safetensors").exists()
        ):
            return snapshot
    return None


def load_model(offline: bool):
    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    else:
        os.environ.pop("HF_HUB_OFFLINE", None)
        os.environ.pop("TRANSFORMERS_OFFLINE", None)

    import torch
    from huggingface_hub import hf_hub_download
    from transformers import AutoModelForCausalLM

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[model] loading {MODEL_ID} on {device}")
    print(f"[model] offline mode: {offline}")

    model_source = MODEL_ID
    if offline:
        snapshot = local_snapshot_dir()
        if snapshot is None:
            raise SystemExit(
                "Offline mode requested, but no complete local model snapshot was found. "
                "Run once without --offline to download pytorch_model.bin."
            )
        model_source = str(snapshot)
        print(f"[model] local snapshot: {model_source}")

    if not offline:
        print("[model] ensuring pytorch_model.bin is present")
        hf_hub_download(
            repo_id=MODEL_ID,
            filename="pytorch_model.bin",
            cache_dir=str(HF_HUB_CACHE),
            resume_download=True,
        )
        snapshot = local_snapshot_dir()
        if snapshot is not None:
            model_source = str(snapshot)

    model = AutoModelForCausalLM.from_pretrained(
        model_source,
        cache_dir=str(HF_HUB_CACHE),
        local_files_only=offline,
        use_safetensors=False,
    )
    model.to(device)
    model.eval()
    return model


def events_after(events: Iterable[int], start_time: float) -> List[int]:
    from anticipation.config import TIME_RESOLUTION
    from anticipation.vocab import ATIME_OFFSET, CONTROL_OFFSET, TIME_OFFSET

    start_tick = int(round(TIME_RESOLUTION * start_time))
    selected: List[int] = []
    triples = list(events)
    for time_token, duration_token, note_token in zip(
        triples[0::3], triples[1::3], triples[2::3]
    ):
        if note_token < CONTROL_OFFSET:
            event_tick = time_token - TIME_OFFSET
        else:
            event_tick = time_token - ATIME_OFFSET
        if event_tick > start_tick:
            selected.extend([time_token, duration_token, note_token])
    return selected


def summarize_events(label: str, events: Iterable[int]) -> None:
    from anticipation import ops

    events = list(events)
    print(f"[{label}] events       : {len(events) // 3}")
    print(f"[{label}] max_time     : {ops.max_time(events):.2f}s")
    print(f"[{label}] instruments  : {dict(ops.get_instruments(events))}")


def generate_once(args: argparse.Namespace) -> None:
    require_runtime()

    from anticipation import ops
    from anticipation.convert import events_to_midi, midi_to_events
    from anticipation.sample import generate

    if args.endpoint:
        os.environ["HF_ENDPOINT"] = args.endpoint

    input_path = Path(args.input_midi)
    output_path = Path(args.output_midi)

    if args.make_demo_input or not input_path.exists():
        print(f"[input] creating demo Call MIDI: {input_path}")
        make_demo_call_midi(input_path)
    elif not input_path.exists():
        raise SystemExit(f"Input MIDI does not exist: {input_path}")

    print_cache_status()
    if args.check_only:
        return

    model = load_model(offline=args.offline)

    print(f"[input] reading Call MIDI: {input_path}")
    call_events = midi_to_events(str(input_path))
    call_events = ops.sort(call_events)
    start_time = ops.max_time(call_events)
    end_time = start_time + args.response_seconds
    summarize_events("call", call_events)

    print(
        f"[generate] start={start_time:.2f}s end={end_time:.2f}s "
        f"top_p={args.top_p}"
    )
    combined_events = generate(
        model,
        start_time=start_time,
        end_time=end_time,
        inputs=call_events,
        top_p=args.top_p,
        debug=args.debug,
    )
    combined_events = ops.sort(combined_events)
    summarize_events("combined", combined_events)

    response_events = events_after(combined_events, start_time)
    summarize_events("response", response_events)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    midi = events_to_midi(combined_events)
    midi.save(output_path)
    print(f"[output] saved combined Call+Response MIDI: {output_path}")

    response_path = output_path.with_name(output_path.stem + "_response_only.mid")
    events_to_midi(response_events).save(response_path)
    print(f"[output] saved Response-only MIDI       : {response_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="One-shot AMT Transformer call-response demo")
    parser.add_argument("--input-midi", default=str(DEFAULT_INPUT), help="human Call MIDI path")
    parser.add_argument("--output-midi", default=str(DEFAULT_OUTPUT), help="combined output MIDI path")
    parser.add_argument("--make-demo-input", action="store_true", help="create/overwrite a demo Call MIDI")
    parser.add_argument("--response-seconds", type=float, default=2.0, help="length of generated Response")
    parser.add_argument("--top-p", type=float, default=0.95, help="nucleus sampling threshold")
    parser.add_argument("--offline", action="store_true", help="load model only from local cache")
    parser.add_argument("--check-only", action="store_true", help="print dependency/cache status and exit")
    parser.add_argument("--debug", action="store_true", help="print verbose anticipation tokens")
    parser.add_argument(
        "--endpoint",
        default=None,
        help="optional HuggingFace endpoint, e.g. https://huggingface.co or https://hf-mirror.com",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    generate_once(args)


if __name__ == "__main__":
    main()
