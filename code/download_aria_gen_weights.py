"""
Download Aria's generative checkpoint into the project.

The public Hugging Face model card points to aria-medium-gen, but the actual
generative weights are currently published in loubb/aria-medium-base as
model-gen.safetensors. This script downloads that file and stores it with a
clear local name.
"""

from __future__ import annotations

import argparse
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_URL = "https://huggingface.co/loubb/aria-medium-base/resolve/main/model-gen.safetensors?download=true"
DEFAULT_OUTPUT = ROOT / "downloads" / "aria-medium-gen-model.safetensors"


def format_size(num_bytes: int) -> str:
    value = float(num_bytes)
    for suffix in ("B", "KB", "MB", "GB"):
        if value < 1024.0:
            return f"{value:.1f}{suffix}"
        value /= 1024.0
    return f"{value:.1f}TB"


def download(url: str, output: Path, hf_token: str | None, overwrite: bool) -> None:
    if output.exists() and not overwrite:
        print(f"[skip] output already exists: {output} ({format_size(output.stat().st_size)})")
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    part_path = output.with_suffix(output.suffix + ".part")
    if part_path.exists():
        part_path.unlink()

    headers = {"User-Agent": "MFP-Aria-Downloader/1.0"}
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"
    request = urllib.request.Request(url, headers=headers)

    print(f"[download] {url}")
    print(f"[download] -> {output}")
    try:
        with urllib.request.urlopen(request, timeout=60) as response, part_path.open("wb") as handle:
            total_header = response.headers.get("Content-Length")
            total = int(total_header) if total_header and total_header.isdigit() else 0
            received = 0
            last_report = time.time()
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                received += len(chunk)
                now = time.time()
                if now - last_report >= 2.0:
                    if total:
                        pct = received / total * 100.0
                        print(f"[progress] {format_size(received)} / {format_size(total)} ({pct:.1f}%)")
                    else:
                        print(f"[progress] {format_size(received)}")
                    last_report = now
    except urllib.error.HTTPError as exc:
        part_path.unlink(missing_ok=True)
        if exc.code in (401, 403):
            raise SystemExit(
                f"Hugging Face refused access with HTTP {exc.code}. "
                "Log in on Hugging Face, accept any model access prompt if shown, "
                "then rerun with --hf-token YOUR_TOKEN."
            ) from exc
        raise
    except Exception:
        part_path.unlink(missing_ok=True)
        raise

    part_path.replace(output)
    print(f"[done] downloaded: {output} ({format_size(output.stat().st_size)})")
    print("[next]")
    print(
        f"C:\\Python312\\python.exe {ROOT / 'code' / 'prepare_aria_gen_local.py'} "
        f"--safetensors {output}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download Aria generative model-gen.safetensors")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        download(args.url, Path(args.output), args.hf_token, args.overwrite)
    except KeyboardInterrupt:
        print("\n[cancelled]", file=sys.stderr)
        raise SystemExit(130)


if __name__ == "__main__":
    main()
