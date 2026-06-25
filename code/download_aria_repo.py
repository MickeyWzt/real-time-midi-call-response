"""
Download and extract the official EleutherAI/aria repository without git.
"""

from __future__ import annotations

import shutil
import urllib.request
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
URL = "https://codeload.github.com/EleutherAI/aria/zip/refs/heads/main"
ZIP_PATH = ROOT / "ab_tests" / "aria-main.zip"
EXTRACT_DIR = ROOT / "ab_tests" / "aria_repo_extract"
TARGET_DIR = ROOT / "code" / "aria"


def main() -> None:
    ZIP_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"[download] {URL}")
    print(f"[download] -> {ZIP_PATH}")
    with urllib.request.urlopen(URL, timeout=60) as response, ZIP_PATH.open("wb") as handle:
        shutil.copyfileobj(response, handle)

    if EXTRACT_DIR.exists():
        shutil.rmtree(EXTRACT_DIR)
    EXTRACT_DIR.mkdir(parents=True)
    with zipfile.ZipFile(ZIP_PATH, "r") as archive:
        archive.extractall(EXTRACT_DIR)

    source_dirs = [item for item in EXTRACT_DIR.iterdir() if item.is_dir()]
    if not source_dirs:
        raise SystemExit("No extracted repo directory found.")
    source = source_dirs[0]
    if TARGET_DIR.exists():
        shutil.rmtree(TARGET_DIR)
    shutil.copytree(source, TARGET_DIR)
    print(f"[done] repo={TARGET_DIR}")


if __name__ == "__main__":
    main()
