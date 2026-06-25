"""
Run objective parameter search for the offline Call-and-Response generator.

Each preset creates an A/B run with a different control setting, evaluates the
generated Response-only MIDI files, and writes a combined leaderboard.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List


ROOT = Path(__file__).resolve().parents[1]


PRESETS: Dict[str, Dict[str, object]] = {
    "free_raw_reference": {
        "style": "free",
        "theory_control": False,
        "top_p": 0.98,
        "temperature": 0.9,
        "max_melodic_leap": 0,
    },
    "pentatonic_no_theory": {
        "style": "pentatonic_2bar_4_4",
        "theory_control": False,
        "top_p": 0.98,
        "temperature": 0.9,
        "max_melodic_leap": 0,
    },
    "pentatonic_balanced": {
        "style": "pentatonic_2bar_4_4",
        "theory_control": True,
        "top_p": 0.95,
        "temperature": 0.85,
        "max_melodic_leap": 7,
        "cadence_degree": "root",
    },
    "pentatonic_balanced_fifth_cadence": {
        "style": "pentatonic_2bar_4_4",
        "theory_control": True,
        "top_p": 0.95,
        "temperature": 0.85,
        "max_melodic_leap": 7,
        "cadence_degree": "fifth",
    },
    "pentatonic_conservative": {
        "style": "pentatonic_2bar_4_4",
        "theory_control": True,
        "top_p": 0.90,
        "temperature": 0.75,
        "max_melodic_leap": 5,
        "cadence_degree": "root",
    },
    "pentatonic_very_conservative": {
        "style": "pentatonic_2bar_4_4",
        "theory_control": True,
        "top_p": 0.85,
        "temperature": 0.65,
        "max_melodic_leap": 4,
        "cadence_degree": "root",
    },
    "pentatonic_creative": {
        "style": "pentatonic_2bar_4_4",
        "theory_control": True,
        "top_p": 0.98,
        "temperature": 1.00,
        "max_melodic_leap": 9,
        "cadence_degree": "root",
    },
    "pentatonic_creative_wide": {
        "style": "pentatonic_2bar_4_4",
        "theory_control": True,
        "top_p": 0.99,
        "temperature": 1.10,
        "max_melodic_leap": 12,
        "cadence_degree": "root",
    },
    "pentatonic_low_temp_no_strongbeat": {
        "style": "pentatonic_2bar_4_4",
        "theory_control": True,
        "strong_beat_stable": False,
        "top_p": 0.92,
        "temperature": 0.75,
        "max_melodic_leap": 7,
        "cadence_degree": "root",
    },
}


def default_output_dir() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return ROOT / "ab_tests" / f"objective_search_{stamp}"


def bool_flag(name: str, value: bool) -> List[str]:
    return [f"--{name}"] if value else [f"--no-{name}"]


def run_command(command: List[str]) -> None:
    print("[run] " + " ".join(command))
    subprocess.run(command, cwd=str(ROOT), check=True)


def base_generation_command(args: argparse.Namespace, run_dir: Path) -> List[str]:
    command = [
        sys.executable,
        str(ROOT / "code" / "offline_ab_test.py"),
        "--trials",
        str(args.trials),
        "--seed",
        str(args.seed),
        "--output-dir",
        str(run_dir),
        "--bpm",
        str(args.bpm),
        "--pentatonic-root",
        str(args.pentatonic_root),
        "--pentatonic-mode",
        args.pentatonic_mode,
    ]
    if args.input_midi:
        command.extend(["--input-midi", args.input_midi])
    else:
        command.extend(["--input-dir", args.input_dir])
    if args.include_medium:
        command.append("--include-medium")
    if args.include_aria:
        command.append("--include-aria")
        command.extend(["--aria-model-id", args.aria_model_id])
        if args.hf_token:
            command.extend(["--hf-token", args.hf_token])
        command.extend(["--aria-prompt-tokens", str(args.aria_prompt_tokens)])
        command.extend(["--aria-max-new-tokens", str(args.aria_max_new_tokens)])
        command.extend(["--aria-temperature", str(args.aria_temperature)])
        command.extend(["--aria-top-p", str(args.aria_top_p)])
        command.extend(bool_flag("aria-apply-style", args.aria_apply_style))
    if args.candidates:
        command.extend(["--candidates", args.candidates])
    if args.max_calls is not None:
        command.extend(["--max-calls", str(args.max_calls)])
    if args.offline:
        command.append("--offline")
    else:
        command.append("--no-offline")
    return command


def apply_preset(command: List[str], preset: Dict[str, object]) -> List[str]:
    result = list(command)
    for key, value in preset.items():
        option = key.replace("_", "-")
        if key == "theory_control":
            result.extend(bool_flag("theory-control", bool(value)))
        elif key == "strong_beat_stable":
            result.extend(bool_flag("strong-beat-stable", bool(value)))
        else:
            result.extend([f"--{option}", str(value)])
    return result


def evaluate_command(args: argparse.Namespace, run_dir: Path) -> List[str]:
    return [
        sys.executable,
        str(ROOT / "code" / "evaluate_melody_metrics.py"),
        "--run-dir",
        str(run_dir),
        "--scale-root",
        str(args.pentatonic_root),
        "--scale-mode",
        args.pentatonic_mode,
        "--beats-per-bar",
        "4",
    ]


def read_summary(path: Path, preset_name: str) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["preset"] = preset_name
    return rows


def write_leaderboard(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows.sort(key=lambda item: float(item.get("mean_objective_score", 0.0)), reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["overall_rank"] = str(rank)
    fields: List[str] = ["overall_rank", "preset", "candidate", "model_id", "sample_count"]
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def selected_presets(args: argparse.Namespace) -> List[str]:
    if args.presets:
        names = [item.strip() for item in args.presets.split(",") if item.strip()]
    else:
        names = list(PRESETS)
    unknown = [name for name in names if name not in PRESETS]
    if unknown:
        raise SystemExit(f"Unknown preset(s): {unknown}. Available: {sorted(PRESETS)}")
    return names


def run_search(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    leaderboard_rows: List[Dict[str, str]] = []

    for preset_name in selected_presets(args):
        run_dir = output_dir / preset_name
        if not args.aggregate_only:
            command = apply_preset(base_generation_command(args, run_dir), PRESETS[preset_name])
            run_command(command)
            run_command(evaluate_command(args, run_dir))
        summary_path = run_dir / "objective_summary.csv"
        if not summary_path.exists():
            raise SystemExit(f"Missing summary for preset '{preset_name}': {summary_path}")
        leaderboard_rows.extend(read_summary(run_dir / "objective_summary.csv", preset_name))

    leaderboard_path = output_dir / "objective_leaderboard.csv"
    write_leaderboard(leaderboard_path, leaderboard_rows)
    if leaderboard_rows:
        best = sorted(
            leaderboard_rows,
            key=lambda item: float(item.get("mean_objective_score", 0.0)),
            reverse=True,
        )[0]
        print(
            "[best] "
            f"preset={best.get('preset')} candidate={best.get('candidate')} "
            f"score={best.get('mean_objective_score')}"
        )
    print(f"[done] objective_search_dir={output_dir}")
    print(f"[done] leaderboard={leaderboard_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate and rank MIDI response settings with objective metrics")
    parser.add_argument("--input-midi", default=str(ROOT / "code" / "demo_call.mid"))
    parser.add_argument("--input-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260529)
    parser.add_argument("--offline", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-medium", action="store_true")
    parser.add_argument("--include-aria", action="store_true")
    parser.add_argument("--aria-model-id", default=str(ROOT / "model_weights" / "aria-medium-gen"))
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--aria-prompt-tokens", type=int, default=512)
    parser.add_argument("--aria-max-new-tokens", type=int, default=768)
    parser.add_argument("--aria-temperature", type=float, default=0.97)
    parser.add_argument("--aria-top-p", type=float, default=0.95)
    parser.add_argument("--aria-apply-style", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--candidates", default=None, help="comma-separated candidate names")
    parser.add_argument("--max-calls", type=int, default=None, help="limit Calls for smoke tests")
    parser.add_argument("--presets", default=None, help="comma-separated preset names; omit to run all")
    parser.add_argument("--aggregate-only", action="store_true", help="only rebuild leaderboard from existing summaries")
    parser.add_argument("--bpm", type=float, default=100.0)
    parser.add_argument("--pentatonic-root", type=int, default=60)
    parser.add_argument("--pentatonic-mode", choices=["major", "minor"], default="major")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.input_dir:
        args.input_midi = None
    if bool(args.input_midi) == bool(args.input_dir):
        raise SystemExit("Pass exactly one of --input-midi or --input-dir.")
    if args.trials < 1:
        raise SystemExit("--trials must be at least 1.")
    if args.aria_prompt_tokens < 1:
        raise SystemExit("--aria-prompt-tokens must be at least 1.")
    if args.aria_max_new_tokens < 1:
        raise SystemExit("--aria-max-new-tokens must be at least 1.")
    if args.aria_temperature <= 0:
        raise SystemExit("--aria-temperature must be positive.")
    if not 0.0 < args.aria_top_p <= 1.0:
        raise SystemExit("--aria-top-p must be in (0, 1].")


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    run_search(args)


if __name__ == "__main__":
    main()
