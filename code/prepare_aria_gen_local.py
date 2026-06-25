"""
Prepare a local aria-medium-gen folder from a downloaded model.safetensors.

The aria-medium-gen checkpoint uses the same architecture/tokenizer code as
aria-medium-base. If Hugging Face access is blocked, manually download the gen
model.safetensors and run this helper to build a local folder usable by
aria_call_response_once.py.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HF_HUB_CACHE = ROOT / "hf_cache" / "hub"
BASE_MODEL_ID = "loubb/aria-medium-base"
DEFAULT_OUTPUT = ROOT / "model_weights" / "aria-medium-gen"


def local_snapshot_dir(model_id: str) -> Path:
    model_dir = HF_HUB_CACHE / f"models--{model_id.replace('/', '--')}" / "snapshots"
    if not model_dir.exists():
        raise SystemExit(f"No local snapshot directory for {model_id}: {model_dir}")
    for snapshot in model_dir.iterdir():
        if (snapshot / "config.json").exists() and (snapshot / "model.safetensors").exists():
            return snapshot
    raise SystemExit(f"No complete local snapshot for {model_id}: {model_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare local aria-medium-gen folder")
    parser.add_argument("--safetensors", required=True, help="downloaded aria-medium-gen model.safetensors")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    weights = Path(args.safetensors)
    if not weights.exists():
        raise SystemExit(f"Safetensors file does not exist: {weights}")
    if weights.stat().st_size < 100_000_000:
        raise SystemExit(f"Safetensors file is suspiciously small: {weights} ({weights.stat().st_size} bytes)")

    base = local_snapshot_dir(BASE_MODEL_ID)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    for name in [
        "config.json",
        "configuration_aria.py",
        "modeling_aria.py",
        "tokenization_aria.py",
        "tokenizer_config.json",
    ]:
        shutil.copy2(base / name, output / name)
    shutil.copy2(weights, output / "model.safetensors")

    print(f"[done] local aria-medium-gen prepared: {output}")
    print("[run]")
    print(
        f"C:\\Python312\\python.exe {ROOT / 'code' / 'aria_call_response_once.py'} "
        f"--model-id {output} --offline --input-midi {ROOT / 'code' / 'demo_call.mid'} "
        f"--output-midi {ROOT / 'code' / 'aria_gen_call_response_2bar.mid'} "
        "--max-new-tokens 1024 --temperature 0.95 --top-p 0.98 --response-bars 2 --monophonic-response"
    )


if __name__ == "__main__":
    main()
