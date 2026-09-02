"""Build publication PDF figures from the harmonized integrity outputs."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import reportlab
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas


ROOT = Path(__file__).resolve().parents[1]
BLUE = (55 / 255, 94 / 255, 160 / 255)
GREEN = (105 / 255, 170 / 255, 70 / 255)
RED = (190 / 255, 78 / 255, 74 / 255)
GOLD = (205 / 255, 145 / 255, 48 / 255)
PURPLE = (126 / 255, 87 / 255, 194 / 255)
GRAY = (0.42, 0.42, 0.42)
LIGHT = (0.88, 0.88, 0.88)
DARK = (0.12, 0.12, 0.12)
FONT_REGULAR = "MFPVera"
FONT_BOLD = "MFPVeraBold"
_REPORTLAB_FONTS = Path(reportlab.__file__).resolve().parent / "fonts"
pdfmetrics.registerFont(TTFont(FONT_REGULAR, _REPORTLAB_FONTS / "Vera.ttf"))
pdfmetrics.registerFont(TTFont(FONT_BOLD, _REPORTLAB_FONTS / "VeraBd.ttf"))


def rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def rgb(pdf: canvas.Canvas, value: Tuple[float, float, float], fill: bool = True) -> None:
    if fill:
        pdf.setFillColorRGB(*value)
    else:
        pdf.setStrokeColorRGB(*value)


def text(pdf: canvas.Canvas, x: float, y: float, value: str, size: float = 8, bold: bool = False) -> None:
    rgb(pdf, DARK)
    pdf.setFont(FONT_BOLD if bold else FONT_REGULAR, size)
    pdf.drawString(x, y, value)


def centered(pdf: canvas.Canvas, x: float, y: float, value: str, size: float = 8, bold: bool = False) -> None:
    rgb(pdf, DARK)
    pdf.setFont(FONT_BOLD if bold else FONT_REGULAR, size)
    pdf.drawCentredString(x, y, value)


def title(pdf: canvas.Canvas, value: str, subtitle: str) -> None:
    text(pdf, 30, 444, value, 17, True)
    rgb(pdf, GRAY)
    pdf.setFont(FONT_REGULAR, 8.5)
    pdf.drawString(30, 427, subtitle)


def y_axis(
    pdf: canvas.Canvas,
    x: float,
    y: float,
    width: float,
    height: float,
    minimum: float,
    maximum: float,
    ticks: Sequence[float],
) -> None:
    rgb(pdf, DARK, fill=False)
    pdf.setLineWidth(0.6)
    pdf.line(x, y, x, y + height)
    pdf.line(x, y, x + width, y)
    for tick in ticks:
        position = y + (tick - minimum) / (maximum - minimum) * height
        rgb(pdf, LIGHT, fill=False)
        pdf.setLineWidth(0.35)
        pdf.line(x, position, x + width, position)
        rgb(pdf, GRAY)
        pdf.setFont(FONT_REGULAR, 7)
        pdf.drawRightString(x - 5, position - 2.5, f"{tick:.2f}")


def candidate_figure(data_dir: Path, clustered_dir: Path, output: Path) -> None:
    clustered = rows(clustered_dir / "candidate_call_cluster_summary.csv")
    all_rows = rows(data_dir / "harmonized_trial_metrics.csv")
    summary_fields = (
        "objective_score",
        "style_compliance_score",
        "non_style_structural_score",
    )
    grouped: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in all_rows:
        grouped[row["candidate"]].append(row)
    by_candidate = {
        candidate: {
            f"mean_{field}": sum(float(row[field]) for row in candidate_rows) / len(candidate_rows)
            for field in summary_fields
        }
        for candidate, candidate_rows in grouped.items()
    }
    by_candidate_clustered = {row["candidate"]: row for row in clustered}
    labels = ["Raw AMT", "Controlled AMT", "Motif baseline"]
    ids = ["amt_small_raw", "amt_small_controlled", "motif_transform_baseline"]
    colors = [GRAY, BLUE, GOLD]

    pdf = canvas.Canvas(str(output), pagesize=(720, 470), initialFontName=FONT_REGULAR)
    title(
        pdf,
        "Harmonized Call100 candidate comparison",
        "27,000 repeated rows; means and 95% intervals aggregate at the 100-call level",
    )

    text(pdf, 35, 394, "(a) Composite score with 95% CI", 10, True)
    x, y, width, height = 65, 190, 250, 180
    y_axis(pdf, x, y, width, height, 0.50, 0.80, [0.50, 0.60, 0.70, 0.80])
    for index, (label, candidate, color) in enumerate(zip(labels, ids, colors)):
        row = by_candidate[candidate]
        value = float(row["mean_objective_score"])
        clustered_row = by_candidate_clustered[candidate]
        low = float(clustered_row["call_cluster_ci95_low"])
        high = float(clustered_row["call_cluster_ci95_high"])
        center_x = x + 47 + index * 78
        bar_height = (value - 0.50) / 0.30 * height
        rgb(pdf, color)
        pdf.rect(center_x - 22, y, 44, bar_height, stroke=0, fill=1)
        rgb(pdf, DARK, fill=False)
        pdf.setLineWidth(0.8)
        low_y = y + (low - 0.50) / 0.30 * height
        high_y = y + (high - 0.50) / 0.30 * height
        pdf.line(center_x, low_y, center_x, high_y)
        pdf.line(center_x - 4, low_y, center_x + 4, low_y)
        pdf.line(center_x - 4, high_y, center_x + 4, high_y)
        centered(pdf, center_x, y - 15, label, 7.5)
        centered(pdf, center_x, y + bar_height + 7, f"{value:.3f}", 8, True)

    text(pdf, 365, 394, "(b) Score decomposition", 10, True)
    x2, y2, width2, height2 = 400, 260, 275, 105
    y_axis(pdf, x2, y2, width2, height2, 0.50, 0.90, [0.50, 0.70, 0.90])
    for index, (label, candidate) in enumerate(zip(labels, ids)):
        row = by_candidate[candidate]
        center_x = x2 + 52 + index * 86
        for offset, field, color in [
            (-10, "mean_style_compliance_score", GOLD),
            (10, "mean_non_style_structural_score", BLUE),
        ]:
            value = float(row[field])
            rgb(pdf, color)
            pdf.rect(center_x + offset - 8, y2, 16, (value - 0.50) / 0.40 * height2, stroke=0, fill=1)
        centered(pdf, center_x, y2 - 13, label.replace(" AMT", ""), 7)
    rgb(pdf, GOLD)
    pdf.rect(467, 379, 9, 6, stroke=0, fill=1)
    text(pdf, 480, 378, "Constraint target", 7)
    rgb(pdf, BLUE)
    pdf.rect(565, 379, 9, 6, stroke=0, fill=1)
    text(pdf, 578, 378, "Remaining components", 7)

    text(pdf, 365, 216, "(c) Controlled minus raw by preset", 10, True)
    paired: Dict[Tuple[str, str, int], Dict[str, float]] = defaultdict(dict)
    for row in all_rows:
        key = (row["preset"], row["call_id"], int(float(row["trial"])))
        paired[key][row["candidate"]] = float(row["objective_score"])
    differences: Dict[str, List[float]] = defaultdict(list)
    for (preset, _, _), values in paired.items():
        if "amt_small_controlled" in values and "amt_small_raw" in values:
            differences[preset].append(values["amt_small_controlled"] - values["amt_small_raw"])
    preset_labels = {
        "pentatonic_low_temp_no_strongbeat": "Low-temp",
        "pentatonic_no_theory": "No theory",
        "pentatonic_conservative": "Conservative",
        "pentatonic_balanced": "Balanced",
        "pentatonic_creative": "Creative",
        "pentatonic_creative_wide": "Wide",
    }
    ordered = sorted(
        ((preset_labels[key], sum(value) / len(value)) for key, value in differences.items()),
        key=lambda item: item[1],
        reverse=True,
    )
    base_x, base_y, chart_w = 455, 70, 205
    for index, (label, value) in enumerate(ordered):
        bar_y = base_y + (len(ordered) - 1 - index) * 21
        text(pdf, 385, bar_y + 2, label, 7.5)
        rgb(pdf, BLUE)
        pdf.rect(base_x, bar_y, value / 0.20 * chart_w, 10, stroke=0, fill=1)
        text(pdf, base_x + value / 0.20 * chart_w + 4, bar_y + 2, f"+{value:.3f}", 7, True)
    rgb(pdf, GRAY, fill=False)
    pdf.line(base_x, base_y - 8, base_x + chart_w, base_y - 8)
    for tick in [0.0, 0.1, 0.2]:
        tick_x = base_x + tick / 0.20 * chart_w
        pdf.line(tick_x, base_y - 11, tick_x, base_y - 5)
        centered(pdf, tick_x, base_y - 20, f"{tick:.1f}", 7)
    pdf.save()


def ablation_figure(integrity_dir: Path, output: Path) -> None:
    data = rows(integrity_dir / "ablation_score_decomposition.csv")
    pdf = canvas.Canvas(str(output), pagesize=(720, 420), initialFontName=FONT_REGULAR)
    text(pdf, 30, 392, "A0-A6 structural score decomposition", 17, True)
    rgb(pdf, GRAY)
    pdf.setFont(FONT_REGULAR, 8.5)
    pdf.drawString(30, 375, "Composite = 0.42 constraint-target + 0.58 remaining components; 9,000 rows/variant")
    x, y, width, height = 70, 75, 610, 265
    y_axis(pdf, x, y, width, height, 0.50, 0.85, [0.50, 0.60, 0.70, 0.80])
    fields = [
        ("mean_structural_compliance_composite", PURPLE, "Composite"),
        ("mean_style_compliance_score", GOLD, "Constraint-target"),
        ("mean_non_style_structural_score", BLUE, "Remaining"),
    ]
    group_width = width / len(data)
    for index, row in enumerate(data):
        center_x = x + group_width * (index + 0.5)
        for field_index, (field, color, _) in enumerate(fields):
            value = float(row[field])
            bar_x = center_x - 25 + field_index * 17
            rgb(pdf, color)
            pdf.rect(bar_x, y, 13, max(0, (value - 0.50) / 0.35 * height), stroke=0, fill=1)
        centered(pdf, center_x, y - 15, row["variant"], 8, True)
    legend_x = 225
    for index, (_, color, label) in enumerate(fields):
        rgb(pdf, color)
        pdf.rect(legend_x + index * 135, 352, 10, 7, stroke=0, fill=1)
        text(pdf, legend_x + 14 + index * 135, 351, label, 7.5)
    pdf.save()


def contribution_figure(integrity_dir: Path, output: Path) -> None:
    data = rows(integrity_dir / "ablation_score_decomposition.csv")
    metrics = [
        ("mean_structural_compliance_composite", PURPLE, "Composite"),
        ("mean_style_compliance_score", GOLD, "Constraint-target"),
        ("mean_non_style_structural_score", BLUE, "Remaining"),
    ]
    steps = []
    for left, right in zip(data, data[1:]):
        steps.append(
            (
                f"{right['variant']}-{left['variant']}",
                [float(right[field]) - float(left[field]) for field, _, _ in metrics],
            )
        )
    pdf = canvas.Canvas(str(output), pagesize=(720, 390), initialFontName=FONT_REGULAR)
    text(pdf, 30, 363, "Stepwise module contribution by score family", 17, True)
    rgb(pdf, GRAY)
    pdf.setFont(FONT_REGULAR, 8.5)
    pdf.drawString(30, 346, "A5/A6 directly target declared metrics; remaining components can move in either direction")
    axis_x, axis_y, axis_w = 250, 65, 400
    minimum, maximum = -0.04, 0.22
    zero_x = axis_x + (0 - minimum) / (maximum - minimum) * axis_w
    rgb(pdf, LIGHT, fill=False)
    for tick in [-0.04, 0.0, 0.04, 0.08, 0.12, 0.16, 0.20]:
        tick_x = axis_x + (tick - minimum) / (maximum - minimum) * axis_w
        pdf.line(tick_x, axis_y - 12, tick_x, 318)
        centered(pdf, tick_x, axis_y - 24, f"{tick:+.2f}", 7)
    rgb(pdf, DARK, fill=False)
    pdf.setLineWidth(0.8)
    pdf.line(zero_x, axis_y - 4, zero_x, 318)
    for index, (step, values) in enumerate(steps):
        center_y = 300 - index * 40
        text(pdf, 45, center_y - 2, step, 8, True)
        for metric_index, (value, (_, color, _)) in enumerate(zip(values, metrics)):
            bar_y = center_y + 9 - metric_index * 11
            end_x = axis_x + (value - minimum) / (maximum - minimum) * axis_w
            rgb(pdf, color)
            pdf.rect(min(zero_x, end_x), bar_y, abs(end_x - zero_x), 7, stroke=0, fill=1)
    for index, (_, color, label) in enumerate(metrics):
        rgb(pdf, color)
        pdf.rect(300 + index * 125, 329, 10, 7, stroke=0, fill=1)
        text(pdf, 314 + index * 125, 328, label, 7.5)
    pdf.save()


def endpoint_figure(endpoint_dir: Path, output: Path) -> None:
    summary = [
        row
        for row in rows(endpoint_dir / "endpoint_condition_summary.csv")
        if row["deadline_s"] == "2.0" and row["condition_family"] == "main"
    ]
    short = {
        "adaptive_full": "Adaptive full",
        "adaptive_no_clustering": "No cluster",
        "adaptive_no_confirmation": "No confirm",
        "adaptive_neither": "Neither",
        "fixed_300ms": "Fixed 300",
        "fixed_500ms": "Fixed 500",
        "fixed_800ms": "Fixed 800",
    }
    pdf = canvas.Canvas(str(output), pagesize=(720, 440), initialFontName=FONT_REGULAR)
    text(pdf, 30, 412, "Call100 MIDI-VAD first-commit benchmark", 17, True)
    rgb(pdf, GRAY)
    pdf.setFont(FONT_REGULAR, 8.5)
    pdf.drawString(30, 395, "Final Note-Off proxy boundary; 100 ms early tolerance and 2 s post-boundary deadline")

    text(pdf, 35, 364, "(a) Boundary precision, recall, and custom PS-F1", 10, True)
    x, y, width, height = 65, 95, 390, 245
    y_axis(pdf, x, y, width, height, 0.0, 1.0, [0.0, 0.25, 0.50, 0.75, 1.0])
    fields = [("precision", BLUE, "Precision"), ("recall", GOLD, "Recall"), ("f1", PURPLE, "PS-F1")]
    group_width = width / len(summary)
    for index, row in enumerate(summary):
        center_x = x + group_width * (index + 0.5)
        for metric_index, (field, color, _) in enumerate(fields):
            value = float(row[field])
            rgb(pdf, color)
            pdf.rect(center_x - 15 + metric_index * 10, y, 8, value * height, stroke=0, fill=1)
        centered(pdf, center_x, y - 12, short[row["condition"]], 7.5)
    for index, (_, color, label) in enumerate(fields):
        rgb(pdf, color)
        pdf.rect(190 + index * 90, 350, 9, 6, stroke=0, fill=1)
        text(pdf, 203 + index * 90, 349, label, 7)

    text(pdf, 485, 364, "(b) Error and safety trade-off", 10, True)
    text(pdf, 490, 340, "Condition", 7, True)
    text(pdf, 585, 340, "Premature", 7, True)
    text(pdf, 655, 340, "Median s", 7, True)
    for index, row in enumerate(summary):
        row_y = 315 - index * 34
        text(pdf, 490, row_y, short[row["condition"]], 7.5)
        premature = int(float(row["premature_decision_count"]))
        rgb(pdf, GOLD)
        pdf.rect(585, row_y - 1, premature / 100 * 55, 7, stroke=0, fill=1)
        text(pdf, 585 + premature / 100 * 55 + 3, row_y, str(premature), 8)
        text(pdf, 662, row_y, f"{float(row['median_signed_error_s']):.3f}", 8)
    rgb(pdf, GRAY)
    pdf.setFont(FONT_REGULAR, 7)
    pdf.drawString(485, 55, "Proxy boundaries are reproducible but not human annotations.")
    pdf.save()


def latency_figure(latency_dir: Path, clustered_dir: Path, output: Path) -> None:
    descriptive = rows(latency_dir / "latency_summary_by_condition.csv")
    clustered = rows(clustered_dir / "latency_call_cluster_summary.csv")
    by_condition = {row["condition"]: row for row in descriptive}
    by_condition_clustered = {row["condition"]: row for row in clustered}
    conditions = ["L0_preload_off", "L1_preload_on"]
    labels = ["Preload off", "Preload on"]
    colors = [GRAY, BLUE]

    pdf = canvas.Canvas(str(output), pagesize=(720, 420), initialFontName=FONT_REGULAR)
    title(
        pdf,
        "Speculative-preload scheduler replay",
        "9,000 repeated rows/condition; uncertainty aggregates at the 100-call level",
    )

    text(pdf, 35, 362, "(a) First-event readiness lower bound", 10, True)
    x, y, width, height = 70, 90, 300, 245
    y_axis(pdf, x, y, width, height, 0.0, 350.0, [0.0, 100.0, 200.0, 300.0])
    text(pdf, 40, 340, "ms", 7)
    for index, (condition, label, color) in enumerate(zip(conditions, labels, colors)):
        row = by_condition[condition]
        cluster_row = by_condition_clustered[condition]
        center_x = x + 90 + index * 130
        mean_value = float(row["mean_latency_ms"])
        p95_value = float(row["p95_latency_ms"])
        low = float(cluster_row["call_cluster_ci95_low_ms"])
        high = float(cluster_row["call_cluster_ci95_high_ms"])
        for offset, value, alpha in [(-18, mean_value, 1.0), (18, p95_value, 0.55)]:
            rgb(pdf, tuple(component * alpha + (1 - alpha) for component in color))
            pdf.rect(center_x + offset - 13, y, 26, value / 350.0 * height, stroke=0, fill=1)
        low_y = y + low / 350.0 * height
        high_y = y + high / 350.0 * height
        rgb(pdf, DARK, fill=False)
        pdf.line(center_x - 18, low_y, center_x - 18, high_y)
        pdf.line(center_x - 22, low_y, center_x - 14, low_y)
        pdf.line(center_x - 22, high_y, center_x - 14, high_y)
        centered(pdf, center_x, y - 15, label, 8)
        centered(pdf, center_x - 18, y + mean_value / 350.0 * height + 7, f"{mean_value:.1f}", 7, True)
        centered(pdf, center_x + 18, y + p95_value / 350.0 * height + 7, f"{p95_value:.1f}", 7)
    rgb(pdf, BLUE)
    pdf.rect(190, 350, 9, 6, stroke=0, fill=1)
    text(pdf, 203, 349, "Mean (call-cluster 95% CI)", 7)
    rgb(pdf, (0.88, 0.95, 0.84))
    pdf.rect(310, 350, 9, 6, stroke=0, fill=1)
    text(pdf, 323, 349, "P95 (descriptive)", 7)

    text(pdf, 430, 362, "(b) First-event readiness-target miss", 10, True)
    x2, y2, width2, height2 = 465, 90, 210, 245
    y_axis(pdf, x2, y2, width2, height2, 0.0, 0.30, [0.0, 0.10, 0.20, 0.30])
    for index, (condition, label, color) in enumerate(zip(conditions, labels, colors)):
        row = by_condition_clustered[condition]
        center_x = x2 + 62 + index * 88
        value = float(row["startup_deadline_miss_rate"])
        low = float(row["startup_miss_call_cluster_ci95_low"])
        high = float(row["startup_miss_call_cluster_ci95_high"])
        rgb(pdf, color)
        pdf.rect(center_x - 18, y2, 36, value / 0.30 * height2, stroke=0, fill=1)
        rgb(pdf, DARK, fill=False)
        low_y = y2 + low / 0.30 * height2
        high_y = y2 + high / 0.30 * height2
        pdf.line(center_x, low_y, center_x, high_y)
        pdf.line(center_x - 4, low_y, center_x + 4, low_y)
        pdf.line(center_x - 4, high_y, center_x + 4, high_y)
        centered(pdf, center_x, y2 - 15, label.replace("Preload ", ""), 8)
        centered(pdf, center_x, y2 + value / 0.30 * height2 + 7, f"{100 * value:.1f}%", 8, True)
    rgb(pdf, GRAY)
    pdf.setFont(FONT_REGULAR, 7)
    pdf.drawString(430, 55, "MIDI send, audio onset, and full-stream underrun are not estimated here.")
    pdf.save()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--harmonized-dir",
        type=Path,
        default=ROOT / "results" / "call100_harmonized_evaluation",
    )
    parser.add_argument(
        "--integrity-dir",
        type=Path,
        default=ROOT / "results" / "evaluation_integrity_audit",
    )
    parser.add_argument(
        "--endpoint-dir",
        type=Path,
        default=ROOT / "results" / "endpoint_benchmark_call100",
    )
    parser.add_argument(
        "--latency-dir",
        type=Path,
        default=ROOT / "results" / "call100_ablation_latency",
    )
    parser.add_argument(
        "--clustered-dir",
        type=Path,
        default=ROOT / "results" / "paper_clustered_statistics",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidate_figure(
        args.harmonized_dir,
        args.clustered_dir,
        args.output_dir / "call100_candidate_level_comparison.pdf",
    )
    ablation_figure(args.integrity_dir, args.output_dir / "ablation_objective_scores.pdf")
    contribution_figure(args.integrity_dir, args.output_dir / "module_contribution_plot.pdf")
    endpoint_figure(args.endpoint_dir, args.output_dir / "endpoint_benchmark.pdf")
    latency_figure(
        args.latency_dir,
        args.clustered_dir,
        args.output_dir / "preload_latency_chart.pdf",
    )
    print(f"[done] output={args.output_dir}")


if __name__ == "__main__":
    main()
