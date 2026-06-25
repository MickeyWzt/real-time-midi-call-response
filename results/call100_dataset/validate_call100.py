from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import mido


OUT_DIR = Path(__file__).resolve().parent
MANIFEST = OUT_DIR / "call100_manifest.csv"
FINGERPRINT = OUT_DIR / "dataset_fingerprint.json"

CATEGORY_QUOTAS = {
    "clear_motif_melody": 12,
    "expressive_rubato": 8,
    "chord_arpeggio_polyphonic": 8,
    "sparse_short": 5,
    "dense_fast": 5,
    "chromatic_outside": 5,
    "false_ending_pause": 4,
    "repetition_tail": 3,
}

EXPECTED_PUBLIC_SOURCES = {
    "POP909": 10,
    "MAESTRO_v3_midi_only": 15,
    "ASAP": 10,
    "Lakh_MIDI_Clean_or_Slakh2100": 10,
    "GiantMIDI_or_MAESTRO_complex": 5,
}


def sha1_file(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_rows() -> list[dict[str, str]]:
    with MANIFEST.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def midi_note_counts(path: Path) -> tuple[int, int]:
    mid = mido.MidiFile(path)
    non_drum = 0
    drum = 0
    for track in mid.tracks:
        for msg in track:
            if msg.type == "note_on" and getattr(msg, "velocity", 0) > 0:
                if getattr(msg, "channel", -1) == 9:
                    drum += 1
                else:
                    non_drum += 1
    return non_drum, drum


def fail(errors: list[str], message: str) -> None:
    errors.append(message)


def warn(warnings: list[str], message: str) -> None:
    warnings.append(message)


def validate() -> int:
    errors: list[str] = []
    warnings: list[str] = []
    if not MANIFEST.exists():
        print(f"ERROR missing manifest: {MANIFEST}")
        return 1
    rows = read_rows()

    if len(rows) != 100:
        fail(errors, f"total rows must be 100, got {len(rows)}")

    call_ids = [row.get("call_id", "") for row in rows]
    dup_ids = [cid for cid, count in Counter(call_ids).items() if count > 1]
    if dup_ids:
        fail(errors, "duplicate call_id values: " + ", ".join(sorted(dup_ids)))

    seen_fps: dict[str, str] = {}
    for row in rows:
        call_id = row.get("call_id", "<missing>")
        midi_path = Path(row.get("midi_path", ""))
        if not midi_path.exists():
            fail(errors, f"{call_id}: midi_path missing: {midi_path}")
            continue
        try:
            non_drum, drum = midi_note_counts(midi_path)
        except Exception as exc:
            fail(errors, f"{call_id}: unreadable MIDI {midi_path}: {exc}")
            continue
        if non_drum <= 0:
            fail(errors, f"{call_id}: no non-drum notes")
        if non_drum <= 0 and drum > 0:
            fail(errors, f"{call_id}: drum-only MIDI")
        try:
            manifest_note_count = int(float(row.get("note_count", "0")))
        except ValueError:
            manifest_note_count = 0
        if manifest_note_count <= 0:
            fail(errors, f"{call_id}: manifest note_count is not > 0")
        try:
            duration = float(row.get("duration_sec", "0"))
        except ValueError:
            duration = 0.0
        if not (1.0 <= duration <= 8.0):
            fail(errors, f"{call_id}: duration_sec outside [1.0, 8.0]: {duration}")
        fp = sha1_file(midi_path)
        manifest_fp = row.get("fingerprint_sha1", "")
        if manifest_fp and manifest_fp != fp:
            fail(errors, f"{call_id}: fingerprint_sha1 does not match file bytes")
        if fp in seen_fps:
            fail(errors, f"{call_id}: duplicate fingerprint with {seen_fps[fp]}")
        seen_fps[fp] = call_id

    public_rows = [row for row in rows if row.get("split") == "public_addition"]
    if len(public_rows) != 50:
        fail(errors, f"public_addition rows must be 50, got {len(public_rows)}")

    public_category_counts = Counter(row.get("category", "") for row in public_rows)
    for category, quota in CATEGORY_QUOTAS.items():
        actual = public_category_counts.get(category, 0)
        if actual < quota:
            fail(errors, f"category quota not met for {category}: need {quota}, got {actual}")

    public_source_counts = Counter(row.get("source_dataset", "") for row in public_rows)
    missing_sources = set()
    if FINGERPRINT.exists():
        try:
            payload = json.loads(FINGERPRINT.read_text(encoding="utf-8"))
            for dataset, meta in payload.get("source_datasets", {}).items():
                if meta.get("missing_reason"):
                    missing_sources.add(dataset)
        except Exception as exc:
            warn(warnings, f"could not parse dataset_fingerprint.json: {exc}")
    else:
        warn(warnings, f"missing dataset_fingerprint.json: {FINGERPRINT}")

    for dataset, expected in EXPECTED_PUBLIC_SOURCES.items():
        actual = public_source_counts.get(dataset, 0)
        if dataset in missing_sources:
            if actual:
                warn(warnings, f"{dataset}: marked missing but has {actual} rows")
            else:
                warn(warnings, f"{dataset}: missing locally; expected {expected}, got 0")
            continue
        if actual != expected:
            warn(warnings, f"{dataset}: expected near {expected}, got {actual}")

    print(f"rows={len(rows)}")
    print("source_dataset_public=" + dict(public_source_counts).__repr__())
    print("category_public=" + dict(public_category_counts).__repr__())
    print(f"errors={len(errors)}")
    print(f"warnings={len(warnings)}")
    for message in warnings:
        print("WARNING " + message)
    for message in errors:
        print("ERROR " + message)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(validate())
