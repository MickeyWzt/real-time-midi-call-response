"""
Create simple publication-ready charts from objective_leaderboard.csv.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List


ROOT = Path(__file__).resolve().parents[1]
PROJECT_DEPS = ROOT / ".python_deps"
if PROJECT_DEPS.exists():
    sys.path.insert(0, str(PROJECT_DEPS))


def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def label(row: Dict[str, str]) -> str:
    candidate = row.get("candidate", "")
    preset = row.get("preset", "")
    return f"{preset}\n{candidate}"


def plot(args: argparse.Namespace) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        plot_svg(args)
        return

    leaderboard = Path(args.leaderboard)
    rows = read_rows(leaderboard)
    if not rows:
        raise SystemExit(f"No rows in leaderboard: {leaderboard}")

    output_dir = Path(args.output_dir) if args.output_dir else leaderboard.parent / "charts"
    output_dir.mkdir(parents=True, exist_ok=True)

    top_rows = sorted(rows, key=lambda row: float(row.get("mean_objective_score", 0.0)), reverse=True)[: args.top_n]
    names = [label(row) for row in reversed(top_rows)]
    scores = [float(row["mean_objective_score"]) for row in reversed(top_rows)]

    plt.figure(figsize=(11, max(5, len(top_rows) * 0.55)))
    plt.barh(names, scores, color="#3b82f6")
    plt.xlabel("Objective Score")
    plt.title("Top Objective Melody Generation Settings")
    plt.xlim(0, 1)
    plt.tight_layout()
    top_path = output_dir / "top_objective_scores.png"
    plt.savefig(top_path, dpi=200)
    plt.close()

    metric_keys = [
        "mean_tonality_score",
        "mean_rhythm_score",
        "mean_interval_score",
        "mean_repetition_score",
        "mean_pitch_diversity_score",
        "mean_compression_score",
    ]
    metric_labels = ["Tonality", "Rhythm", "Interval", "Repetition", "Pitch Diversity", "Compression"]
    grouped_rows = top_rows[: min(6, len(top_rows))]
    x = list(range(len(metric_keys)))
    width = 0.12

    plt.figure(figsize=(12, 6))
    for idx, row in enumerate(grouped_rows):
        values = [float(row.get(key, 0.0)) for key in metric_keys]
        positions = [item + (idx - len(grouped_rows) / 2) * width for item in x]
        plt.bar(positions, values, width=width, label=f"#{row.get('overall_rank')} {row.get('preset')}/{row.get('candidate')}")
    plt.xticks(x, metric_labels, rotation=20, ha="right")
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("Metric Breakdown for Top Settings")
    plt.legend(fontsize=8)
    plt.tight_layout()
    breakdown_path = output_dir / "metric_breakdown_top_settings.png"
    plt.savefig(breakdown_path, dpi=200)
    plt.close()

    print(f"[chart] {top_path}")
    print(f"[chart] {breakdown_path}")


def svg_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def plot_svg(args: argparse.Namespace) -> None:
    leaderboard = Path(args.leaderboard)
    rows = read_rows(leaderboard)
    if not rows:
        raise SystemExit(f"No rows in leaderboard: {leaderboard}")
    output_dir = Path(args.output_dir) if args.output_dir else leaderboard.parent / "charts"
    output_dir.mkdir(parents=True, exist_ok=True)

    top_rows = sorted(rows, key=lambda row: float(row.get("mean_objective_score", 0.0)), reverse=True)[: args.top_n]
    width = 1200
    row_h = 46
    margin_l = 430
    margin_r = 80
    margin_t = 70
    height = margin_t + row_h * len(top_rows) + 60
    bar_w = width - margin_l - margin_r
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="30" y="36" font-family="Arial" font-size="24" font-weight="700">Top Objective Melody Generation Settings</text>',
    ]
    for idx, row in enumerate(top_rows):
        y = margin_t + idx * row_h
        score = float(row.get("mean_objective_score", 0.0))
        name = f"#{row.get('overall_rank')} {row.get('preset')} / {row.get('candidate')}"
        parts.append(f'<text x="30" y="{y + 25}" font-family="Arial" font-size="15">{svg_escape(name)}</text>')
        parts.append(f'<rect x="{margin_l}" y="{y + 7}" width="{bar_w}" height="24" fill="#e5e7eb"/>')
        parts.append(f'<rect x="{margin_l}" y="{y + 7}" width="{bar_w * max(0.0, min(1.0, score)):.2f}" height="24" fill="#3b82f6"/>')
        parts.append(f'<text x="{margin_l + bar_w + 12}" y="{y + 25}" font-family="Arial" font-size="14">{score:.3f}</text>')
    parts.append("</svg>")
    top_path = output_dir / "top_objective_scores.svg"
    top_path.write_text("\n".join(parts), encoding="utf-8")

    metric_keys = [
        "mean_tonality_score",
        "mean_rhythm_score",
        "mean_interval_score",
        "mean_repetition_score",
        "mean_pitch_diversity_score",
        "mean_compression_score",
    ]
    metric_labels = ["Tonality", "Rhythm", "Interval", "Repetition", "PitchDiv", "Compression"]
    rows_for_breakdown = top_rows[: min(6, len(top_rows))]
    width = 1200
    height = 720
    chart_x = 90
    chart_y = 90
    chart_w = 850
    chart_h = 450
    colors = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2"]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="30" y="40" font-family="Arial" font-size="24" font-weight="700">Metric Breakdown for Top Settings</text>',
    ]
    group_w = chart_w / len(metric_keys)
    bar_w2 = group_w / (len(rows_for_breakdown) + 1)
    for i in range(6):
        y = chart_y + chart_h - i * chart_h / 5
        val = i / 5
        parts.append(f'<line x1="{chart_x}" y1="{y:.2f}" x2="{chart_x + chart_w}" y2="{y:.2f}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="50" y="{y + 4:.2f}" font-family="Arial" font-size="12">{val:.1f}</text>')
    for metric_idx, metric in enumerate(metric_keys):
        base_x = chart_x + metric_idx * group_w
        parts.append(f'<text x="{base_x + 8:.2f}" y="{chart_y + chart_h + 28}" font-family="Arial" font-size="12">{metric_labels[metric_idx]}</text>')
        for row_idx, row in enumerate(rows_for_breakdown):
            val = float(row.get(metric, 0.0))
            x = base_x + 6 + row_idx * bar_w2
            h = chart_h * max(0.0, min(1.0, val))
            y = chart_y + chart_h - h
            parts.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_w2 * 0.82:.2f}" height="{h:.2f}" fill="{colors[row_idx % len(colors)]}"/>')
    legend_x = 970
    legend_y = 95
    for idx, row in enumerate(rows_for_breakdown):
        y = legend_y + idx * 48
        parts.append(f'<rect x="{legend_x}" y="{y}" width="16" height="16" fill="{colors[idx % len(colors)]}"/>')
        parts.append(
            f'<text x="{legend_x + 24}" y="{y + 13}" font-family="Arial" font-size="12">'
            f'{svg_escape("#" + str(row.get("overall_rank")) + " " + row.get("preset", "") + "/" + row.get("candidate", ""))}</text>'
        )
    parts.append("</svg>")
    breakdown_path = output_dir / "metric_breakdown_top_settings.svg"
    breakdown_path.write_text("\n".join(parts), encoding="utf-8")

    print(f"[chart] {top_path}")
    print(f"[chart] {breakdown_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot objective search leaderboard charts")
    parser.add_argument("--leaderboard", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--top-n", type=int, default=12)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    plot(args)


if __name__ == "__main__":
    main()
