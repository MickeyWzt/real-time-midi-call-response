from __future__ import annotations

import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ANSWER_KEY = ROOT / "private" / "answer_key.csv"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def side_to_candidate(answer_row: dict[str, str], choice: str) -> str:
    if choice == "same":
        return "same"
    if choice == "A":
        return answer_row["a_candidate"]
    if choice == "B":
        return answer_row["b_candidate"]
    return "unknown"


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python tools/analyze_results.py path/to/BlindListeningResponses.csv", file=sys.stderr)
        return 2

    responses_path = Path(sys.argv[1])
    answer_rows = {row["pair_id"]: row for row in read_csv(ANSWER_KEY)}
    responses = read_csv(responses_path)

    overall = Counter()
    by_pair: dict[str, Counter[str]] = defaultdict(Counter)
    unique_submissions = set()

    for row in responses:
        pair_id = row.get("pair_id", "")
        answer_row = answer_rows.get(pair_id)
        if not answer_row:
            overall["unknown_pair"] += 1
            continue
        choice = row.get("choice", "")
        candidate = side_to_candidate(answer_row, choice)
        overall[candidate] += 1
        by_pair[pair_id][candidate] += 1
        if row.get("submission_id"):
            unique_submissions.add(row["submission_id"])

    print(f"Responses: {len(responses)} rows")
    print(f"Participants/submissions: {len(unique_submissions)}")
    print("\nOverall preference counts")
    for key, value in overall.most_common():
        print(f"  {key}: {value}")

    print("\nPair-level counts")
    for pair_id in sorted(answer_rows):
        row = answer_rows[pair_id]
        counts = by_pair[pair_id]
        print(
            f"  {pair_id} {row['call_id']} {row.get('comparison_type', '')}: "
            f"controlled={counts['amt_small_controlled']}, "
            f"raw={counts['amt_small_raw']}, "
            f"motif={counts['motif_transform_baseline']}, "
            f"same={counts['same']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
