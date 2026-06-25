"""
One-shot Aria MIDI continuation for Call-and-Response testing.

This script uses the locally cached Aria Hugging Face checkpoint plus
ariautils. It reads a Call MIDI prompt, generates a continuation, and exports
both combined and response-only MIDI files.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional


ROOT = Path(__file__).resolve().parents[1]
PROJECT_DEPS = ROOT / ".python_deps"
HF_CACHE = ROOT / "hf_cache"
HF_HUB_CACHE = HF_CACHE / "hub"
HF_MODULES_CACHE = ROOT / "hf_modules_cache"

if PROJECT_DEPS.exists():
    sys.path.insert(0, str(PROJECT_DEPS))

os.environ.setdefault("HF_HOME", str(HF_CACHE))
os.environ.setdefault("HF_HUB_CACHE", str(HF_HUB_CACHE))
os.environ.setdefault("HF_MODULES_CACHE", str(HF_MODULES_CACHE))


DEFAULT_MODEL_ID = str(ROOT / "model_weights" / "aria-medium-gen")
DEFAULT_INPUT = ROOT / "code" / "demo_call.mid"
DEFAULT_OUTPUT = ROOT / "code" / "aria_call_response.mid"


def require_runtime() -> None:
    missing = []
    for module_name in ("torch", "transformers", "safetensors", "ariautils", "mido"):
        try:
            __import__(module_name)
        except ImportError:
            missing.append(module_name)
    if missing:
        raise SystemExit(
            "Missing runtime modules: "
            + ", ".join(missing)
            + ". Use C:\\Python312\\python.exe with project .python_deps."
        )


def patch_transformers_for_aria() -> None:
    # Aria's tokenizer was authored for transformers 4.x. In our current
    # environment BatchEncoding lives in tokenization_utils_base.
    import transformers.tokenization_utils as tokenization_utils
    from transformers.tokenization_utils_base import BatchEncoding

    if not hasattr(tokenization_utils, "BatchEncoding"):
        tokenization_utils.BatchEncoding = BatchEncoding


def local_snapshot_dir(model_id: str) -> Optional[Path]:
    model_dir = HF_HUB_CACHE / f"models--{model_id.replace('/', '--')}" / "snapshots"
    if not model_dir.exists():
        return None
    for snapshot in model_dir.iterdir():
        if (snapshot / "config.json").exists() and (
            (snapshot / "model.safetensors").exists()
            or (snapshot / "pytorch_model.bin").exists()
        ):
            return snapshot
    return None


def load_tokenizer(snapshot: Path):
    patch_transformers_for_aria()
    sys.path.insert(0, str(snapshot))
    from tokenization_aria import AriaTokenizer

    return AriaTokenizer(add_eos_token=True)


def load_model(model_id: str, offline: bool, endpoint: Optional[str], hf_token: Optional[str]):
    if endpoint:
        os.environ["HF_ENDPOINT"] = endpoint
    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    else:
        os.environ.pop("HF_HUB_OFFLINE", None)
        os.environ.pop("TRANSFORMERS_OFFLINE", None)

    import torch
    from transformers import AutoModelForCausalLM

    model_path = Path(model_id)
    looks_like_path = model_path.is_absolute() or any(part in model_id for part in ("\\", "/"))
    if model_path.exists():
        snapshot = model_path
        source = str(model_path)
    elif looks_like_path:
        raise SystemExit(
            f"Aria model path does not exist: {model_path}. "
            "Download aria-medium-gen model.safetensors and run prepare_aria_gen_local.py first."
        )
    else:
        snapshot = local_snapshot_dir(model_id)
        source = str(snapshot) if snapshot is not None else model_id
    if offline and snapshot is None:
        raise SystemExit(
            f"No local snapshot for {model_id}. Run without --offline once, "
            "or download the checkpoint into hf_cache."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[model] loading {source} on {device}")
    model_kwargs = {
        "cache_dir": str(HF_HUB_CACHE),
        "local_files_only": offline,
        "trust_remote_code": True,
        "use_safetensors": True,
    }
    if hf_token:
        model_kwargs["token"] = hf_token
    model = AutoModelForCausalLM.from_pretrained(
        source,
        **model_kwargs,
    )
    model.to(device)
    model.eval()
    return model, Path(source) if Path(source).exists() else local_snapshot_dir(model_id)


def max_note_end_tick(midi_dict) -> int:
    if not midi_dict.note_msgs:
        return 0
    return max(int(note["data"]["end"]) for note in midi_dict.note_msgs)


def select_monophonic_notes(note_msgs):
    by_start = {}
    for note in note_msgs:
        start = int(note["data"]["start"])
        current = by_start.get(start)
        if current is None or int(note["data"]["pitch"]) > int(current["data"]["pitch"]):
            by_start[start] = note

    selected = []
    last_end = 0
    for note in sorted(by_start.values(), key=lambda item: (int(item["data"]["start"]), -int(item["data"]["pitch"]))):
        start = int(note["data"]["start"])
        end = int(note["data"]["end"])
        if start < last_end:
            start = last_end
        if end <= start:
            continue
        copied = {
            **note,
            "tick": start,
            "data": {
                **note["data"],
                "start": start,
                "end": end,
            },
        }
        selected.append(copied)
        last_end = end
    return selected


def response_only_midi_dict(decoded_midi_dict, prompt_end_tick: int, args: argparse.Namespace):
    from ariautils.midi import MidiDict

    msg_dict = decoded_midi_dict.get_msg_dict()
    ticks_per_beat = int(msg_dict["ticks_per_beat"])
    response_window_ticks = ticks_per_beat * args.response_bars * args.beats_per_bar
    response_end_tick = prompt_end_tick + response_window_ticks
    response_notes = []
    for note in msg_dict["note_msgs"]:
        start = int(note["data"]["start"])
        end = int(note["data"]["end"])
        if start < prompt_end_tick:
            continue
        if start >= response_end_tick:
            continue
        copied = {
            "type": "note",
            "data": {
                "pitch": int(note["data"]["pitch"]),
                "start": max(0, start - prompt_end_tick),
                "end": max(1, min(end, response_end_tick) - prompt_end_tick),
                "velocity": int(note["data"].get("velocity", 90)),
            },
            "tick": max(0, int(note["tick"]) - prompt_end_tick),
            "channel": int(note.get("channel", 0)),
        }
        response_notes.append(copied)

    if args.monophonic_response:
        response_notes = select_monophonic_notes(response_notes)

    new_dict = {
        "meta_msgs": [],
        "tempo_msgs": msg_dict["tempo_msgs"][:1] or [{"type": "tempo", "data": 600000, "tick": 0}],
        "pedal_msgs": [],
        "instrument_msgs": [{"type": "instrument", "data": 0, "tick": 0, "channel": 0}],
        "note_msgs": response_notes,
        "ticks_per_beat": ticks_per_beat,
        "metadata": {},
    }
    return MidiDict.from_msg_dict(new_dict)


def combined_prompt_response_midi_dict(prompt_midi_dict, response_midi_dict, prompt_end_tick: int):
    from ariautils.midi import MidiDict

    prompt_dict = prompt_midi_dict.get_msg_dict()
    response_dict = response_midi_dict.get_msg_dict()
    shifted_response_notes = []
    for note in response_dict["note_msgs"]:
        start = int(note["data"]["start"]) + prompt_end_tick
        end = int(note["data"]["end"]) + prompt_end_tick
        shifted_response_notes.append(
            {
                "type": "note",
                "data": {
                    "pitch": int(note["data"]["pitch"]),
                    "start": start,
                    "end": end,
                    "velocity": int(note["data"].get("velocity", 90)),
                },
                "tick": start,
                "channel": int(note.get("channel", 0)),
            }
        )

    new_dict = {
        "meta_msgs": prompt_dict["meta_msgs"],
        "tempo_msgs": prompt_dict["tempo_msgs"],
        "pedal_msgs": [],
        "instrument_msgs": prompt_dict["instrument_msgs"] or [{"type": "instrument", "data": 0, "tick": 0, "channel": 0}],
        "note_msgs": prompt_dict["note_msgs"] + shifted_response_notes,
        "ticks_per_beat": prompt_dict["ticks_per_beat"],
        "metadata": {},
    }
    return MidiDict.from_msg_dict(new_dict)


def generate_once(args: argparse.Namespace) -> None:
    if args.endpoint:
        os.environ["HF_ENDPOINT"] = args.endpoint
    require_runtime()
    patch_transformers_for_aria()

    import torch
    from ariautils.midi import MidiDict

    input_path = Path(args.input_midi)
    if not input_path.exists():
        raise SystemExit(f"Input MIDI does not exist: {input_path}")

    model, snapshot = load_model(args.model_id, args.offline, args.endpoint, args.hf_token)
    if snapshot is None:
        raise SystemExit(f"Could not resolve local snapshot for tokenizer: {args.model_id}")
    tokenizer = load_tokenizer(snapshot)

    prompt = MidiDict.from_midi(input_path)
    prompt_end_tick = max_note_end_tick(prompt)
    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(model.device)
    print(f"[input] tokens={input_ids.shape[-1]} notes={len(prompt.note_msgs)} prompt_end_tick={prompt_end_tick}")

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    decoded = tokenizer.decode(output_ids[0].detach().cpu().tolist())
    response = response_only_midi_dict(decoded, prompt_end_tick, args)
    combined = combined_prompt_response_midi_dict(prompt, response, prompt_end_tick)

    output_path = Path(args.output_midi)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_midi().save(output_path)
    print(f"[output] combined={output_path}")

    response_path = output_path.with_name(output_path.stem + "_response_only.mid")
    response.to_midi().save(response_path)
    print(f"[output] response_only={response_path}")
    raw_response_notes = sum(
        1
        for note in decoded.note_msgs
        if int(note["data"]["start"]) >= prompt_end_tick
    )
    print(
        f"[output] decoded_notes={len(decoded.note_msgs)} "
        f"raw_response_notes={raw_response_notes} exported_response_notes={len(response.note_msgs)}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="One-shot Aria MIDI continuation")
    parser.add_argument("--input-midi", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-midi", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--offline", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--endpoint", default=None)
    parser.add_argument("--hf-token", default=None, help="optional Hugging Face access token for gated/private checkpoints")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--response-bars", type=int, default=2)
    parser.add_argument("--beats-per-bar", type=int, default=4)
    parser.add_argument("--monophonic-response", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    generate_once(args)


if __name__ == "__main__":
    main()
