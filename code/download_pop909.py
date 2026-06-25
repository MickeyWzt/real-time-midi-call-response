"""
Download POP909 dataset archive with Python stdlib streaming.

This is a convenience helper. If the network fails, manually download:
https://github.com/music-x-lab/POP909-Dataset/archive/refs/heads/master.zip
and pass it to build_call50_dataset.py with --dataset-zip.
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path


URL = "https://codeload.github.com/music-x-lab/POP909-Dataset/zip/refs/heads/master"


def download(url: str, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(output.suffix + ".part")
    print(f"[download] {url}")
    print(f"[download] -> {output}")
    with urllib.request.urlopen(url, timeout=60) as response, temp.open("wb") as handle:
        total = response.headers.get("Content-Length")
        total_int = int(total) if total and total.isdigit() else None
        downloaded = 0
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
            downloaded += len(chunk)
            if total_int:
                print(f"\r[download] {downloaded / total_int * 100:5.1f}%", end="")
            else:
                print(f"\r[download] {downloaded / (1024 * 1024):.1f} MB", end="")
    print()
    temp.replace(output)
    print(f"[done] {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download POP909 dataset archive")
    parser.add_argument("--url", default=URL)
    parser.add_argument("--output", default=str(Path(__file__).resolve().parents[1] / "ab_tests" / "POP909-Dataset-master.zip"))
    args = parser.parse_args()
    download(args.url, Path(args.output))


if __name__ == "__main__":
    main()
