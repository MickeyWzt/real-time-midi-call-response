from __future__ import annotations

import csv
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
PRESET = "pentatonic_conservative"
RUN_DIR = WORKSPACE / "ab_tests" / "objective_search_call100_trials15" / PRESET
MANIFEST = WORKSPACE / "ab_tests" / "calls_100_public_final" / "call100_manifest.csv"
ANSWER_KEY = RUN_DIR / "answer_key.csv"
METRICS = RUN_DIR / "objective_metrics.csv"

PUBLIC_DIR = ROOT / "public"
MIDI_DIR = PUBLIC_DIR / "midi"
PRIVATE_DIR = ROOT / "private"

CANDIDATE_CONTROLLED = "amt_small_controlled"
CANDIDATE_RAW = "amt_small_raw"
CANDIDATE_MOTIF = "motif_transform_baseline"

EXPECTED_ORIGINS = {
    "public_midi_extract": 6,
    "real_public_dataset": 4,
    "artificial_stress": 2,
}
EXPECTED_COMPARISONS = {
    "controlled_vs_raw": 8,
    "controlled_vs_motif": 4,
}
EXPECTED_CANDIDATES = {
    CANDIDATE_CONTROLLED: 12,
    CANDIDATE_RAW: 8,
    CANDIDATE_MOTIF: 4,
}

CALL_PLAN = [
    # 6 POP909 calls.
    ("P005", CANDIDATE_RAW),
    ("P012", CANDIDATE_RAW),
    ("P021", CANDIDATE_RAW),
    ("P034", CANDIDATE_RAW),
    ("P041", CANDIDATE_MOTIF),
    ("P047", CANDIDATE_MOTIF),
    # 4 real public dataset calls.
    ("R08", CANDIDATE_RAW),
    ("R15", CANDIDATE_RAW),
    ("R04", CANDIDATE_RAW),
    ("R28", CANDIDATE_MOTIF),
    # 2 artificial stress / hard cases.
    ("A03", CANDIDATE_RAW),
    ("A07", CANDIDATE_MOTIF),
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def median(values: list[float]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def stable_side(pair_id: str) -> bool:
    digest = hashlib.sha1(pair_id.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 2 == 0


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fallback_count(answer_row: dict[str, str]) -> int:
    return int(answer_row.get("fallback_count") or 0)


def select_median_sample(
    metrics_by_call_candidate: dict[tuple[str, str], list[dict[str, str]]],
    answer_by_sample: dict[str, dict[str, str]],
    call_id: str,
    candidate: str,
) -> tuple[dict[str, str], dict[str, str], float, float]:
    rows = metrics_by_call_candidate[(call_id, candidate)]
    scored = [row for row in rows if row.get("objective_score")]
    if len(scored) != 15:
        raise RuntimeError(f"Expected 15 scored rows for {call_id}/{candidate}, found {len(scored)}")
    scores = [float(row["objective_score"]) for row in scored]
    target = median(scores)
    eligible = [
        row
        for row in scored
        if fallback_count(answer_by_sample[row["sample_id"]]) == 0
    ]
    if not eligible:
        raise RuntimeError(f"No no-fallback rows for {call_id}/{candidate}")
    selected = min(
        eligible,
        key=lambda row: (
            abs(float(row["objective_score"]) - target),
            row.get("sample_id", ""),
        ),
    )
    answer = answer_by_sample[selected["sample_id"]]
    return selected, answer, target, abs(float(selected["objective_score"]) - target)


def clean_public_midi() -> None:
    MIDI_DIR.mkdir(parents=True, exist_ok=True)
    for path in MIDI_DIR.glob("*.mid"):
        path.unlink()


def verify_public_copy(public_path: Path, required_source: Path, forbidden_source: Path | None = None) -> None:
    public_hash = file_sha256(public_path)
    if public_hash != file_sha256(required_source):
        raise RuntimeError(f"{public_path.name} does not match required source {required_source}")
    if forbidden_source is not None and public_hash == file_sha256(forbidden_source):
        raise RuntimeError(f"{public_path.name} unexpectedly matches combined MIDI {forbidden_source}")


def validate_selection(public_pairs: list[dict[str, str]], private_rows: list[dict[str, str]]) -> None:
    if len(public_pairs) != 12 or len(private_rows) != 12:
        raise RuntimeError(f"Expected 12 pairs, found public={len(public_pairs)} private={len(private_rows)}")

    origins = Counter(row["origin"] for row in private_rows)
    if dict(origins) != EXPECTED_ORIGINS:
        raise RuntimeError(f"Unexpected origin counts: {dict(origins)}")

    comparisons = Counter(row["comparison_type"] for row in private_rows)
    if dict(comparisons) != EXPECTED_COMPARISONS:
        raise RuntimeError(f"Unexpected comparison counts: {dict(comparisons)}")

    candidates = Counter()
    for row in private_rows:
        if CANDIDATE_CONTROLLED not in {row["a_candidate"], row["b_candidate"]}:
            raise RuntimeError(f"Pair {row['pair_id']} does not include the controlled candidate")
        if int(row["a_fallback_count"]) != 0 or int(row["b_fallback_count"]) != 0:
            raise RuntimeError(
                f"Pair {row['pair_id']} uses fallback: A={row['a_fallback_count']} B={row['b_fallback_count']}"
            )
        candidates.update([row["a_candidate"], row["b_candidate"]])

    if dict(candidates) != EXPECTED_CANDIDATES:
        raise RuntimeError(f"Unexpected candidate counts: {dict(candidates)}")


def main() -> int:
    manifest_rows = {row["call_id"]: row for row in read_csv(MANIFEST)}
    answer_by_sample = {row["sample_id"]: row for row in read_csv(ANSWER_KEY)}
    metrics_by_call_candidate: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in read_csv(METRICS):
        metrics_by_call_candidate[(row["call_id"], row["candidate"])].append(row)

    clean_public_midi()
    PRIVATE_DIR.mkdir(parents=True, exist_ok=True)
    public_pairs = []
    private_rows = []

    for index, (call_id, comparison_candidate) in enumerate(CALL_PLAN, start=1):
        pair_id = f"P{index:03d}"
        controlled_metric, controlled_answer, controlled_median, controlled_distance = select_median_sample(
            metrics_by_call_candidate,
            answer_by_sample,
            call_id,
            CANDIDATE_CONTROLLED,
        )
        comparison_metric, comparison_answer, comparison_median, comparison_distance = select_median_sample(
            metrics_by_call_candidate,
            answer_by_sample,
            call_id,
            comparison_candidate,
        )

        comparison_type = "controlled_vs_raw" if comparison_candidate == CANDIDATE_RAW else "controlled_vs_motif"
        controlled_side_a = stable_side(f"{pair_id}:{call_id}:{comparison_type}")
        if controlled_side_a:
            side_a = (CANDIDATE_CONTROLLED, controlled_metric, controlled_answer, controlled_median, controlled_distance)
            side_b = (comparison_candidate, comparison_metric, comparison_answer, comparison_median, comparison_distance)
        else:
            side_a = (comparison_candidate, comparison_metric, comparison_answer, comparison_median, comparison_distance)
            side_b = (CANDIDATE_CONTROLLED, controlled_metric, controlled_answer, controlled_median, controlled_distance)

        call_name = f"{pair_id}_call.mid"
        a_name = f"{pair_id}_A.mid"
        b_name = f"{pair_id}_B.mid"
        source = manifest_rows[call_id]
        call_source = Path(source["midi_path"])
        a_response_source = Path(side_a[2]["response_only_midi"])
        b_response_source = Path(side_b[2]["response_only_midi"])
        a_combined_source = Path(side_a[2]["combined_midi"])
        b_combined_source = Path(side_b[2]["combined_midi"])
        shutil.copy2(call_source, MIDI_DIR / call_name)
        shutil.copy2(a_response_source, MIDI_DIR / a_name)
        shutil.copy2(b_response_source, MIDI_DIR / b_name)
        verify_public_copy(MIDI_DIR / call_name, call_source)
        verify_public_copy(MIDI_DIR / a_name, a_response_source, a_combined_source)
        verify_public_copy(MIDI_DIR / b_name, b_response_source, b_combined_source)

        public_pairs.append(
            {
                "pair_id": pair_id,
                "call_file": f"midi/{call_name}",
                "a_file": f"midi/{a_name}",
                "b_file": f"midi/{b_name}",
            }
        )
        private_rows.append(
            {
                "pair_id": pair_id,
                "call_id": call_id,
                "origin": source["origin"],
                "source_dataset": source["source_dataset"],
                "category": source["category"],
                "sub_category": source["sub_category"],
                "preset": PRESET,
                "comparison_type": comparison_type,
                "controlled_side": "A" if side_a[0] == CANDIDATE_CONTROLLED else "B",
                "a_candidate": side_a[0],
                "b_candidate": side_b[0],
                "a_sample_id": side_a[2]["sample_id"],
                "b_sample_id": side_b[2]["sample_id"],
                "a_trial": side_a[2]["trial"],
                "b_trial": side_b[2]["trial"],
                "a_objective_score": side_a[1]["objective_score"],
                "b_objective_score": side_b[1]["objective_score"],
                "a_candidate_median_objective_score": f"{side_a[3]:.6f}",
                "b_candidate_median_objective_score": f"{side_b[3]:.6f}",
                "a_distance_from_candidate_median": f"{side_a[4]:.9f}",
                "b_distance_from_candidate_median": f"{side_b[4]:.9f}",
                "call_file": f"midi/{call_name}",
                "a_file": f"midi/{a_name}",
                "b_file": f"midi/{b_name}",
                "a_fallback_count": fallback_count(side_a[2]),
                "b_fallback_count": fallback_count(side_b[2]),
                "a_model_id": side_a[2]["model_id"],
                "b_model_id": side_b[2]["model_id"],
                "a_response_notes": side_a[2]["response_notes"],
                "b_response_notes": side_b[2]["response_notes"],
                "call_duration_seconds": side_a[2]["call_duration_seconds"],
                "a_response_seconds": side_a[2]["response_seconds"],
                "b_response_seconds": side_b[2]["response_seconds"],
            }
        )

    validate_selection(public_pairs, private_rows)

    (PUBLIC_DIR / "study-pairs.json").write_text(
        json.dumps(public_pairs, indent=2) + "\n",
        encoding="utf-8",
    )
    with (PRIVATE_DIR / "answer_key.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(private_rows[0].keys()))
        writer.writeheader()
        writer.writerows(private_rows)

    print(f"Wrote {len(public_pairs)} public pairs")
    print(f"Public MIDI files: {len(list(MIDI_DIR.glob('*.mid')))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
