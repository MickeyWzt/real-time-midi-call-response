"""Audit non-subjective feature associations between Call100 inputs and responses.

This script intentionally measures only three reproducible MIDI-feature relations:

* transposition/register-invariant melodic-contour correlation;
* transposition-invariant interval-2gram multiset Dice overlap; and
* duration-normalized 16-bin onset-profile cosine similarity.

For every ``(preset, candidate, trial)`` block, an observed Call--response
feature score is compared with the mean score against all valid mismatched
responses in the same block.  The resulting association excess is a finite-benchmark,
non-subjective feature association.  It is not a measure of musical quality,
perceptual appropriateness, harmonic fit, or live interaction quality.

The script reuses existing local Call100 MIDI and response MIDI files.  It does
not run AMT inference or regenerate any response.  Published outputs contain
only filenames and SHA-256 hashes, never local filesystem paths.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, Mapping, Optional, Sequence

import numpy as np

from evaluate_melody_metrics import read_notes


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parents[1]
PRESETS = (
    "pentatonic_no_theory",
    "pentatonic_balanced",
    "pentatonic_conservative",
    "pentatonic_creative",
    "pentatonic_creative_wide",
    "pentatonic_low_temp_no_strongbeat",
)
CANDIDATES = (
    "amt_small_raw",
    "amt_small_controlled",
    "motif_transform_baseline",
)
METRICS = ("contour_r", "interval_2gram_dice", "onset_profile_cosine")


@dataclass(frozen=True)
class OnsetNote:
    """A note-on needed for chord-aware symbolic feature extraction."""

    onset_seconds: float
    pitch: int


@dataclass(frozen=True)
class OnsetCluster:
    """One start-anchored chord/onset cluster."""

    onset_seconds: float
    median_pitch: float


@dataclass(frozen=True)
class FeatureBundle:
    """Cached phrase features used in the three association measurements."""

    contour: Optional[tuple[float, ...]]
    interval_2grams: Optional[Counter[tuple[int, int]]]
    onset_profile: Optional[tuple[float, ...]]


@dataclass(frozen=True)
class PairRecord:
    """A validated local pair, with only path-redacted fields emitted later."""

    call_id: str
    origin: str
    source_dataset: str
    category: str
    preset: str
    candidate: str
    trial: str
    seed: str
    source_experiment: str
    source_variant: str
    response_midi_file: str
    response_midi_sha256: str
    call_midi_file: str
    call_midi_sha256: str
    call_path: Path
    response_path: Path


def cluster_onsets(
    notes: Sequence[OnsetNote], cluster_window_seconds: float = 0.08
) -> list[OnsetCluster]:
    """Cluster Note-Ons using the endpoint detector's start-anchored rule."""

    if cluster_window_seconds < 0:
        raise ValueError("cluster_window_seconds cannot be negative")
    ordered = sorted(notes, key=lambda item: (item.onset_seconds, item.pitch))
    if not ordered:
        return []
    clusters: list[OnsetCluster] = []
    current: list[OnsetNote] = []
    start = ordered[0].onset_seconds
    for note in ordered:
        if current and note.onset_seconds - start > cluster_window_seconds:
            pitches = sorted(item.pitch for item in current)
            midpoint = len(pitches) // 2
            median_pitch = (
                float(pitches[midpoint])
                if len(pitches) % 2
                else (pitches[midpoint - 1] + pitches[midpoint]) / 2.0
            )
            clusters.append(OnsetCluster(start, median_pitch))
            current = []
            start = note.onset_seconds
        current.append(note)
    pitches = sorted(item.pitch for item in current)
    midpoint = len(pitches) // 2
    median_pitch = (
        float(pitches[midpoint])
        if len(pitches) % 2
        else (pitches[midpoint - 1] + pitches[midpoint]) / 2.0
    )
    clusters.append(OnsetCluster(start, median_pitch))
    return clusters


def _resample_by_event_index(values: Sequence[float], points: int) -> Optional[tuple[float, ...]]:
    if len(values) < 3 or points < 3:
        return None
    output: list[float] = []
    last_index = len(values) - 1
    for point in range(points):
        position = point * last_index / (points - 1)
        lower = int(math.floor(position))
        upper = min(lower + 1, last_index)
        fraction = position - lower
        output.append(values[lower] * (1.0 - fraction) + values[upper] * fraction)
    center = mean(output)
    centered = [value - center for value in output]
    norm = math.sqrt(sum(value * value for value in centered))
    if norm <= 1e-12:
        return None
    return tuple(value / norm for value in centered)


def _contour_vector(events: Sequence[tuple[float, float]], points: int = 16) -> Optional[tuple[float, ...]]:
    return _resample_by_event_index([pitch for _, pitch in events], points)


def contour_correlation(
    call: Sequence[tuple[float, float]], response: Sequence[tuple[float, float]], points: int = 16
) -> Optional[float]:
    """Return Pearson contour correlation after event-index resampling.

    Centering and unit normalization remove absolute pitch/register; timing is
    deliberately excluded because it is assessed separately by the onset profile.
    """

    call_vector = _contour_vector(call, points)
    response_vector = _contour_vector(response, points)
    if call_vector is None or response_vector is None:
        return None
    return max(-1.0, min(1.0, sum(a * b for a, b in zip(call_vector, response_vector))))


def _interval_2grams(events: Sequence[tuple[float, float]]) -> Optional[Counter[tuple[int, int]]]:
    if len(events) < 3:
        return None
    pitches = [pitch for _, pitch in events]
    intervals = [max(-12, min(12, int(round(right - left)))) for left, right in zip(pitches, pitches[1:])]
    if len(intervals) < 2:
        return None
    return Counter(tuple(intervals[index : index + 2]) for index in range(len(intervals) - 1))


def _dice(left: Counter[tuple[int, int]], right: Counter[tuple[int, int]]) -> float:
    overlap = sum(min(count, right.get(token, 0)) for token, count in left.items())
    denominator = sum(left.values()) + sum(right.values())
    return 2.0 * overlap / denominator if denominator else 0.0


def interval_2gram_dice(
    call: Sequence[tuple[float, float]], response: Sequence[tuple[float, float]]
) -> Optional[float]:
    """Return transposition-invariant multiset Dice overlap of interval 2grams."""

    call_grams = _interval_2grams(call)
    response_grams = _interval_2grams(response)
    if call_grams is None or response_grams is None:
        return None
    return _dice(call_grams, response_grams)


def _onset_profile(events: Sequence[tuple[float, float]], bins: int = 16) -> Optional[tuple[float, ...]]:
    if len(events) < 2 or bins < 2:
        return None
    start = events[0][0]
    end = events[-1][0]
    duration = end - start
    if duration <= 1e-12:
        return None
    counts = [0.0] * bins
    for onset, _ in events:
        normalized = (onset - start) / duration
        index = min(bins - 1, int(normalized * bins))
        counts[index] += 1.0
    norm = math.sqrt(sum(value * value for value in counts))
    if norm <= 1e-12:
        return None
    return tuple(value / norm for value in counts)


def onset_profile_cosine(
    call: Sequence[tuple[float, float]], response: Sequence[tuple[float, float]], bins: int = 16
) -> Optional[float]:
    """Return cosine similarity of duration-normalized onset-cluster profiles."""

    call_profile = _onset_profile(call, bins)
    response_profile = _onset_profile(response, bins)
    if call_profile is None or response_profile is None:
        return None
    return max(0.0, min(1.0, sum(a * b for a, b in zip(call_profile, response_profile))))


def association_excesses(matrix: Sequence[Sequence[float]]) -> tuple[list[float], list[float], list[float]]:
    """Compare every true pair to all other same-condition response pairings."""

    count = len(matrix)
    if count < 2 or any(len(row) != count for row in matrix):
        raise ValueError("association matrix must be square with at least two rows")
    actual = [float(matrix[index][index]) for index in range(count)]
    mismatched = [
        sum(float(value) for other, value in enumerate(row) if other != index) / (count - 1)
        for index, row in enumerate(matrix)
    ]
    return actual, mismatched, [value - baseline for value, baseline in zip(actual, mismatched)]


def _optional_association_excesses(
    matrix: Sequence[Sequence[Optional[float]]],
) -> list[tuple[Optional[float], Optional[float], Optional[float], int]]:
    """Calculate a dynamic-denominator association excess for every row.

    A metric can be unavailable for a short or degenerate phrase.  In that
    case, the denominator is the number of *valid* off-diagonal similarities,
    not the nominal block size minus one.  A row without a matched score or a
    valid mismatch is deliberately ``NA``.
    """

    count = len(matrix)
    if count < 2 or any(len(row) != count for row in matrix):
        raise ValueError("association matrix must be square with at least two rows")
    result: list[tuple[Optional[float], Optional[float], Optional[float], int]] = []
    for index, row in enumerate(matrix):
        true_pair = row[index]
        alternatives = [float(value) for other, value in enumerate(row) if other != index and value is not None]
        valid_mismatch_count = len(alternatives)
        if true_pair is None:
            result.append((None, None, None, valid_mismatch_count))
            continue
        if not alternatives:
            result.append((None, None, None, 0))
            continue
        baseline = mean(alternatives)
        result.append((float(true_pair), baseline, float(true_pair) - baseline, valid_mismatch_count))
    return result


def association_vectors_from_counts(
    matrix: Sequence[Sequence[Optional[float]]], counts: Sequence[int],
) -> tuple[list[Optional[float]], list[Optional[float]], list[Optional[float]]]:
    """Rebuild association vectors after selecting a multiset of paired units.

    ``counts[j]`` is the multiplicity of the matched Call--response unit ``j``
    in a resample.  For a focal retained Call ``i``, every selected response is
    included in its mismatch bank except one occurrence of response ``i``.
    Thus deleting a Call (count zero) removes both its matched unit and its
    response from every other Call's null bank.  Missing feature similarities
    are omitted from both numerator and denominator.
    """

    score_array = np.asarray(matrix, dtype=float)
    count_array = np.asarray(counts, dtype=float)
    if score_array.ndim != 2 or score_array.shape[0] != score_array.shape[1] or score_array.shape[0] < 2:
        raise ValueError("association matrix must be square with at least two rows")
    if count_array.shape != (score_array.shape[0],):
        raise ValueError("counts must have one non-negative entry per matrix row")
    if np.any(~np.isfinite(count_array)) or np.any(count_array < 0) or np.any(count_array != np.floor(count_array)):
        raise ValueError("counts must be finite non-negative integers")

    finite = np.isfinite(score_array)
    weighted_scores = np.where(finite, score_array, 0.0) @ count_array
    weighted_counts = finite.astype(float) @ count_array
    diagonal = np.diag(score_array)
    diagonal_finite = np.isfinite(diagonal)
    denominator = weighted_counts - diagonal_finite.astype(float)
    numerator = weighted_scores - np.where(diagonal_finite, diagonal, 0.0)
    valid = (count_array > 0) & diagonal_finite & (denominator > 0)

    actual: list[Optional[float]] = []
    mismatched: list[Optional[float]] = []
    excess: list[Optional[float]] = []
    for index, is_valid in enumerate(valid):
        if not is_valid:
            actual.append(None)
            mismatched.append(None)
            excess.append(None)
            continue
        observed = float(diagonal[index])
        baseline = float(numerator[index] / denominator[index])
        actual.append(observed)
        mismatched.append(baseline)
        excess.append(observed - baseline)
    return actual, mismatched, excess


def exclude_same_source_mismatches(
    matrix: Sequence[Sequence[Optional[float]]],
    call_ids: Sequence[str],
    clusters: Mapping[str, str],
) -> np.ndarray:
    """Mask off-diagonal responses that share the focal Call's source.

    The matched diagonal remains available even when several Calls originate
    from the same POP909 song. Only those same-source alternatives are
    removed from each focal Call's mismatch bank.
    """

    score_array = np.asarray(matrix, dtype=float).copy()
    if score_array.ndim != 2 or score_array.shape[0] != score_array.shape[1]:
        raise ValueError("association matrix must be square")
    if len(call_ids) != score_array.shape[0]:
        raise ValueError("call_ids must follow the matrix row/column order")
    missing = [call_id for call_id in call_ids if call_id not in clusters]
    if missing:
        raise ValueError(f"missing source clusters for Call IDs: {missing[:5]}")
    for row_index, left_call in enumerate(call_ids):
        for column_index, right_call in enumerate(call_ids):
            if row_index != column_index and clusters[left_call] == clusters[right_call]:
                score_array[row_index, column_index] = np.nan
    return score_array


def _feature_bundle(events: Sequence[tuple[float, float]]) -> FeatureBundle:
    return FeatureBundle(
        contour=_contour_vector(events),
        interval_2grams=_interval_2grams(events),
        onset_profile=_onset_profile(events),
    )


def _feature_similarity(metric: str, call: FeatureBundle, response: FeatureBundle) -> Optional[float]:
    if metric == "contour_r":
        if call.contour is None or response.contour is None:
            return None
        return max(-1.0, min(1.0, sum(a * b for a, b in zip(call.contour, response.contour))))
    if metric == "interval_2gram_dice":
        if call.interval_2grams is None or response.interval_2grams is None:
            return None
        return _dice(call.interval_2grams, response.interval_2grams)
    if metric == "onset_profile_cosine":
        if call.onset_profile is None or response.onset_profile is None:
            return None
        return max(0.0, min(1.0, sum(a * b for a, b in zip(call.onset_profile, response.onset_profile))))
    raise ValueError(f"Unknown metric: {metric}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sample_id(response_midi_file: str) -> str:
    name = Path(response_midi_file).stem
    return name[: -len("_response_only")] if name.endswith("_response_only") else name


def _run_directory(candidate: str, preset: str, ablation_root: Path, motif_root: Path) -> Path:
    if candidate == "amt_small_raw":
        return ablation_root / f"{preset}__A0_raw_amt"
    if candidate == "amt_small_controlled":
        return ablation_root / f"{preset}__A6_full_controlled"
    if candidate == "motif_transform_baseline":
        return motif_root / preset
    raise ValueError(f"Unsupported candidate: {candidate}")


def _load_answer_keys(ablation_root: Path, motif_root: Path) -> Dict[tuple[str, str], tuple[Path, Mapping[str, Mapping[str, str]]]]:
    output: Dict[tuple[str, str], tuple[Path, Mapping[str, Mapping[str, str]]]] = {}
    for preset in PRESETS:
        for candidate in CANDIDATES:
            run_directory = _run_directory(candidate, preset, ablation_root, motif_root)
            answer_key_path = run_directory / "answer_key.csv"
            response_directory = run_directory / "responses"
            if not answer_key_path.exists() or not response_directory.exists():
                raise FileNotFoundError(f"Missing local run material for {candidate}/{preset}")
            with answer_key_path.open("r", newline="", encoding="utf-8-sig") as handle:
                answers = {row["sample_id"]: row for row in csv.DictReader(handle)}
            output[(candidate, preset)] = (response_directory, answers)
    return output


def _load_source_clusters(manifest_path: Path) -> Dict[str, str]:
    clusters: Dict[str, str] = {}
    with manifest_path.open("r", newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            call_id = row["call_id"]
            if row.get("origin") == "artificial_stress":
                clusters[call_id] = f"artificial_{call_id}"
                continue
            source_group = row.get("source_group", "")
            source_file = Path(row.get("source_file", "")).name
            match = re.fullmatch(r"song_(\d+)", source_group) or re.match(r"R\d+_(\d+)", source_file)
            clusters[call_id] = f"pop909_{int(match.group(1)):03d}" if match else f"other_{call_id}"
    if len(clusters) != 100:
        raise RuntimeError(f"Expected 100 Call100 manifest rows, found {len(clusters)}")
    if len(set(clusters.values())) != 87:
        raise RuntimeError(f"Expected 87 source clusters, found {len(set(clusters.values()))}")
    return clusters


def resolve_pair_records(
    metrics_path: Path,
    ablation_root: Path,
    motif_root: Path,
) -> list[PairRecord]:
    """Resolve and validate all public-metric rows to local Call/response MIDI."""

    answer_keys = _load_answer_keys(ablation_root, motif_root)
    call_directory = ablation_root / "call_inputs"
    if not call_directory.exists():
        raise FileNotFoundError(f"Missing Call100 input directory: {call_directory}")
    call_hashes = {path.stem: _sha256(path) for path in call_directory.glob("*.mid")}
    if len(call_hashes) != 100:
        raise RuntimeError(f"Expected 100 local Call MIDI files, found {len(call_hashes)}")
    records: list[PairRecord] = []
    with metrics_path.open("r", newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            candidate = row["candidate"]
            preset = row["preset"]
            if candidate not in CANDIDATES or preset not in PRESETS:
                raise RuntimeError(f"Unexpected candidate/preset row: {candidate}/{preset}")
            response_directory, answer_key = answer_keys[(candidate, preset)]
            response_name = row["response_midi_file"]
            sample_id = _sample_id(response_name)
            answer = answer_key.get(sample_id)
            if answer is None:
                raise RuntimeError(f"No answer-key mapping for {candidate}/{preset}/{sample_id}")
            expected = (row["call_id"], row["trial"], row["seed"], candidate)
            observed = (answer["call_id"], answer["trial"], answer["seed"], answer["candidate"])
            if expected != observed:
                raise RuntimeError(f"Pair mapping mismatch for {candidate}/{preset}/{sample_id}: {expected} != {observed}")
            response_path = response_directory / response_name
            call_path = call_directory / f"{row['call_id']}.mid"
            if not response_path.exists() or not call_path.exists():
                raise FileNotFoundError(f"Missing MIDI for {candidate}/{preset}/{sample_id}")
            response_hash = _sha256(response_path)
            if row.get("response_midi_sha256") and response_hash != row["response_midi_sha256"]:
                raise RuntimeError(f"Response SHA-256 mismatch for {candidate}/{preset}/{sample_id}")
            records.append(
                PairRecord(
                    call_id=row["call_id"],
                    origin=row["origin"],
                    source_dataset=row["source_dataset"],
                    category=row["category"],
                    preset=preset,
                    candidate=candidate,
                    trial=row["trial"],
                    seed=row["seed"],
                    source_experiment=row["source_experiment"],
                    source_variant=row["source_variant"],
                    response_midi_file=response_name,
                    response_midi_sha256=response_hash,
                    call_midi_file=call_path.name,
                    call_midi_sha256=call_hashes[row["call_id"]],
                    call_path=call_path,
                    response_path=response_path,
                )
            )
    if len(records) != 27000:
        raise RuntimeError(f"Expected 27,000 harmonized rows, found {len(records)}")
    return records


def _read_phrase(path: Path) -> list[tuple[float, float]]:
    notes, _ = read_notes(path)
    clusters = cluster_onsets([OnsetNote(note.onset_seconds, note.pitch) for note in notes])
    return [(cluster.onset_seconds, cluster.median_pitch) for cluster in clusters]


def _mean_available(values: Sequence[Optional[float]]) -> Optional[float]:
    available = [float(value) for value in values if value is not None]
    return float(mean(available)) if available else None


def _association_statistics(
    matrix: Sequence[Sequence[Optional[float]]], counts: Sequence[int]
) -> dict[str, object]:
    """Rebuild every block's mismatch bank, then average by Call.

    The analysis target is the mean of each Call's valid per-block association
    statistics over the six presets and 15 trials.  Keeping the block axis
    here ensures that a delete-one replicate reconstructs the original
    same-condition null in each block rather than reusing precomputed
    Call-level excess values.
    """

    score_array = np.asarray(matrix, dtype=float)
    if score_array.ndim == 2:
        score_array = score_array[np.newaxis, :, :]
    if score_array.ndim != 3 or score_array.shape[1] != score_array.shape[2] or score_array.shape[1] < 2:
        raise ValueError("association matrix must be square, or a stack of square matrices")
    count_array = np.asarray(counts, dtype=float)
    call_count = score_array.shape[1]
    if count_array.shape != (call_count,) or np.any(~np.isfinite(count_array)) or np.any(count_array < 0):
        raise ValueError("counts must have one non-negative finite entry per Call")

    finite = np.isfinite(score_array)
    weighted_scores = np.einsum("bij,j->bi", np.where(finite, score_array, 0.0), count_array)
    weighted_counts = np.einsum("bij,j->bi", finite.astype(float), count_array)
    diagonal = np.diagonal(score_array, axis1=1, axis2=2)
    diagonal_finite = np.isfinite(diagonal)
    denominator = weighted_counts - diagonal_finite.astype(float)
    numerator = weighted_scores - np.where(diagonal_finite, diagonal, 0.0)
    valid = (count_array[np.newaxis, :] > 0) & diagonal_finite & (denominator > 0)
    actual_by_block = np.where(valid, diagonal, np.nan)
    mismatched_by_block = np.where(valid, numerator / np.where(denominator > 0, denominator, 1.0), np.nan)
    excess_by_block = np.where(valid, actual_by_block - mismatched_by_block, np.nan)

    def mean_by_call(values: np.ndarray) -> list[Optional[float]]:
        return [
            float(values[:, index][np.isfinite(values[:, index])].mean())
            if np.any(np.isfinite(values[:, index]))
            else None
            for index in range(call_count)
        ]

    actual = mean_by_call(actual_by_block)
    mismatched = mean_by_call(mismatched_by_block)
    excess = mean_by_call(excess_by_block)
    return {
        "actual": actual,
        "mismatched": mismatched,
        "excess": excess,
        "actual_mean": _mean_available(actual),
        "mismatched_mean": _mean_available(mismatched),
        "association_excess_mean": _mean_available(excess),
        "call_count": sum(value is not None for value in excess),
    }


def _jackknife_se(replicates: Sequence[Optional[float]]) -> Optional[float]:
    """Return the classical delete-one jackknife standard error."""

    estimates = [float(value) for value in replicates if value is not None]
    if len(estimates) < 2:
        return None
    estimate_mean = mean(estimates)
    return math.sqrt((len(estimates) - 1) / len(estimates) * sum((value - estimate_mean) ** 2 for value in estimates))


def _normal_ci(point: Optional[float], standard_error: Optional[float]) -> tuple[Optional[float], Optional[float]]:
    if point is None or standard_error is None:
        return None, None
    return point - 1.96 * standard_error, point + 1.96 * standard_error


def _delete_one_call_counts(call_count: int) -> list[np.ndarray]:
    return [np.array([0 if item == omitted else 1 for item in range(call_count)], dtype=int) for omitted in range(call_count)]


def _delete_one_source_counts(call_ids: Sequence[str], clusters: Mapping[str, str]) -> list[np.ndarray]:
    source_clusters = sorted({clusters[call_id] for call_id in call_ids})
    return [
        np.array([0 if clusters[call_id] == omitted else 1 for call_id in call_ids], dtype=int)
        for omitted in source_clusters
    ]


def _jackknife_fields(
    point: Optional[float], replicates: Sequence[Optional[float]], prefix: str
) -> dict[str, object]:
    standard_error = _jackknife_se(replicates)
    lower, upper = _normal_ci(point, standard_error)
    available = [float(value) for value in replicates if value is not None]
    return {
        f"{prefix}_jackknife_replicates": len(available),
        f"{prefix}_jackknife_se": standard_error,
        f"{prefix}_jackknife_ci95_low": lower,
        f"{prefix}_jackknife_ci95_high": upper,
        f"{prefix}_leave_one_out_min": min(available) if available else None,
        f"{prefix}_leave_one_out_max": max(available) if available else None,
    }


def _equal_source_mean(
    values: Sequence[Optional[float]], call_ids: Sequence[str], clusters: Mapping[str, str]
) -> tuple[Optional[float], int]:
    """Average valid Calls within each source, then average sources equally."""

    grouped: Dict[str, list[float]] = defaultdict(list)
    for call_id, value in zip(call_ids, values):
        if value is not None:
            grouped[clusters[call_id]].append(float(value))
    source_means = [mean(grouped[source]) for source in sorted(grouped) if grouped[source]]
    return (float(mean(source_means)) if source_means else None, len(source_means))


def _summary_row(
    metric: str,
    candidate: str,
    matrix: Sequence[Sequence[Optional[float]]],
    call_ids: Sequence[str],
    clusters: Mapping[str, str],
    valid_trial_rows: int,
) -> dict[str, object]:
    """Summarize one candidate using rebuilt Call/source delete-one banks."""

    full_counts = np.ones(len(call_ids), dtype=int)
    point = _association_statistics(matrix, full_counts)
    call_replicates = [_association_statistics(matrix, counts) for counts in _delete_one_call_counts(len(call_ids))]
    source_replicates = [
        _association_statistics(matrix, counts)
        for counts in _delete_one_source_counts(call_ids, clusters)
    ]
    source_points: Dict[str, Optional[float]] = {}
    source_cluster_count = 0
    for statistic in ("actual", "mismatched", "excess"):
        source_point, available_sources = _equal_source_mean(point[statistic], call_ids, clusters)
        source_points[statistic] = source_point
        source_cluster_count = max(source_cluster_count, available_sources)
    output: dict[str, object] = {
        "metric": metric,
        "candidate": candidate,
        "trial_rows": 9000,
        "valid_trial_rows": valid_trial_rows,
        "trial_coverage": valid_trial_rows / 9000,
        "call_count": point["call_count"],
        "pairwise_cells_with_any_valid_trial": int(
            np.any(np.isfinite(np.asarray(matrix, dtype=float)), axis=0).sum()
        ),
        "actual_mean": point["actual_mean"],
        "mismatched_mean": point["mismatched_mean"],
        "association_excess_mean": point["association_excess_mean"],
        "actual_source_mean": source_points["actual"],
        "mismatched_source_mean": source_points["mismatched"],
        "association_excess_source_mean": source_points["excess"],
        "source_cluster_count": source_cluster_count,
    }
    for statistic in ("actual_mean", "mismatched_mean", "association_excess_mean"):
        output.update(
            _jackknife_fields(
                point[statistic], [replicate[statistic] for replicate in call_replicates], f"{statistic}_call"
            )
        )
        source_statistic = {
            "actual_mean": "actual",
            "mismatched_mean": "mismatched",
            "association_excess_mean": "excess",
        }[statistic]
        output.update(
            _jackknife_fields(
                source_points[source_statistic],
                [
                    _equal_source_mean(replicate[source_statistic], call_ids, clusters)[0]
                    for replicate in source_replicates
                ],
                f"{statistic}_source",
            )
        )
    return output


def _paired_excess_values(left: Mapping[str, object], right: Mapping[str, object]) -> list[Optional[float]]:
    return [
        float(left_value) - float(right_value) if left_value is not None and right_value is not None else None
        for left_value, right_value in zip(left["excess"], right["excess"])
    ]


def _paired_excess_mean(left: Mapping[str, object], right: Mapping[str, object]) -> tuple[Optional[float], int, int]:
    values = _paired_excess_values(left, right)
    differences = [value for value in values if value is not None]
    return float(mean(differences)) if differences else None, len(differences), sum(value > 0 for value in differences)


def _comparison_row(
    metric: str,
    comparison: str,
    left_matrix: Sequence[Sequence[Optional[float]]],
    right_matrix: Sequence[Sequence[Optional[float]]],
    call_ids: Sequence[str],
    clusters: Mapping[str, str],
) -> dict[str, object]:
    """Compute a contrast with paired deletes and rebuilt mismatch banks."""

    full_counts = np.ones(len(call_ids), dtype=int)
    left_point = _association_statistics(left_matrix, full_counts)
    right_point = _association_statistics(right_matrix, full_counts)
    point, call_count, positive_calls = _paired_excess_mean(left_point, right_point)
    source_point, source_cluster_count = _equal_source_mean(
        _paired_excess_values(left_point, right_point), call_ids, clusters
    )

    def contrast_for_counts(counts: np.ndarray) -> tuple[Optional[float], Optional[float]]:
        differences = _paired_excess_values(
            _association_statistics(left_matrix, counts), _association_statistics(right_matrix, counts)
        )
        call_value = _mean_available(differences)
        source_value, _ = _equal_source_mean(differences, call_ids, clusters)
        return call_value, source_value

    call_replicates = [contrast_for_counts(counts)[0] for counts in _delete_one_call_counts(len(call_ids))]
    source_replicates = [contrast_for_counts(counts)[1] for counts in _delete_one_source_counts(call_ids, clusters)]
    output: dict[str, object] = {
        "metric": metric,
        "comparison": comparison,
        "call_count": call_count,
        "mean_difference": point,
        "source_cluster_mean_difference": source_point,
        "positive_calls": positive_calls,
        "source_cluster_count": source_cluster_count,
    }
    output.update(_jackknife_fields(point, call_replicates, "mean_difference_call"))
    output.update(_jackknife_fields(source_point, source_replicates, "mean_difference_source"))
    return output


def _format_value(value: object) -> object:
    if isinstance(value, float):
        return f"{value:.6f}"
    return value


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write to {path.name}")
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _format_value(row.get(field, "")) for field in fields})


def _markdown_table(rows: Sequence[Mapping[str, object]], fields: Sequence[str]) -> list[str]:
    lines = ["| " + " | ".join(fields) + " |", "|" + "|".join("---" for _ in fields) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(_format_value(row.get(field, ""))) for field in fields) + " |")
    return lines


def run_analysis(args: argparse.Namespace) -> dict[str, int]:
    records = resolve_pair_records(args.metrics, args.ablation_root, args.motif_root)
    clusters = _load_source_clusters(args.manifest)
    call_feature_cache: Dict[Path, FeatureBundle] = {}
    response_feature_cache: Dict[Path, FeatureBundle] = {}
    for record in records:
        if record.call_path not in call_feature_cache:
            call_feature_cache[record.call_path] = _feature_bundle(_read_phrase(record.call_path))
        if record.response_path not in response_feature_cache:
            response_feature_cache[record.response_path] = _feature_bundle(_read_phrase(record.response_path))

    by_block: Dict[tuple[str, str, str], list[PairRecord]] = defaultdict(list)
    for record in records:
        by_block[(record.preset, record.candidate, record.trial)].append(record)
    if len(by_block) != 270:
        raise RuntimeError(f"Expected 270 candidate/preset/trial blocks, found {len(by_block)}")

    call_ids = sorted(clusters)
    call_index = {call_id: index for index, call_id in enumerate(call_ids)}
    same_source_offdiagonal_cells_per_block = sum(
        1
        for left_index, left_call in enumerate(call_ids)
        for right_index, right_call in enumerate(call_ids)
        if left_index != right_index and clusters[left_call] == clusters[right_call]
    )
    trial_rows: list[dict[str, object]] = []
    valid_trial_rows: Dict[str, Dict[str, int]] = {
        metric: {candidate: 0 for candidate in CANDIDATES} for metric in METRICS
    }
    block_matrices: Dict[str, Dict[str, list[np.ndarray]]] = {
        metric: {candidate: [] for candidate in CANDIDATES}
        for metric in METRICS
    }
    for block_key in sorted(by_block):
        block = sorted(by_block[block_key], key=lambda record: record.call_id)
        if len(block) != 100 or len({record.call_id for record in block}) != 100:
            raise RuntimeError(f"Expected 100 unique Calls in block {block_key}")
        per_metric: Dict[str, list[tuple[Optional[float], Optional[float], Optional[float], int]]] = {}
        for metric in METRICS:
            matrix = [
                [
                    _feature_similarity(
                        metric,
                        call_feature_cache[left.call_path],
                        response_feature_cache[right.response_path],
                    )
                    for right in block
                ]
                for left in block
            ]
            per_metric[metric] = _optional_association_excesses(matrix)
            candidate = block[0].candidate
            global_matrix = np.full((len(call_ids), len(call_ids)), np.nan, dtype=float)
            for row_index, row in enumerate(matrix):
                global_row = call_index[block[row_index].call_id]
                for column_index, value in enumerate(row):
                    if value is None:
                        continue
                    global_column = call_index[block[column_index].call_id]
                    global_matrix[global_row, global_column] = value
            block_matrices[metric][candidate].append(global_matrix)
        for index, record in enumerate(block):
            row: dict[str, object] = {
                "call_id": record.call_id,
                "origin": record.origin,
                "source_dataset": record.source_dataset,
                "source_cluster": clusters[record.call_id],
                "category": record.category,
                "preset": record.preset,
                "candidate": record.candidate,
                "trial": record.trial,
                "seed": record.seed,
                "source_experiment": record.source_experiment,
                "source_variant": record.source_variant,
                "call_midi_file": record.call_midi_file,
                "call_midi_sha256": record.call_midi_sha256,
                "response_midi_file": record.response_midi_file,
                "response_midi_sha256": record.response_midi_sha256,
            }
            for metric in METRICS:
                actual, mismatch, excess, valid_mismatch_count = per_metric[metric][index]
                row[f"{metric}_actual"] = actual
                row[f"{metric}_mismatched_mean"] = mismatch
                row[f"{metric}_association_excess"] = excess
                row[f"{metric}_mismatched_valid_count"] = valid_mismatch_count
                row[f"{metric}_available"] = int(excess is not None)
                if actual is not None and mismatch is not None and excess is not None:
                    valid_trial_rows[metric][record.candidate] += 1
            trial_rows.append(row)
    if len(trial_rows) != 27000:
        raise RuntimeError(f"Expected 27,000 trial rows, found {len(trial_rows)}")

    association_matrices: Dict[str, Dict[str, np.ndarray]] = {
        metric: {} for metric in METRICS
    }
    for metric in METRICS:
        for candidate in CANDIDATES:
            if len(block_matrices[metric][candidate]) != 90:
                raise RuntimeError(f"Expected 90 preset/trial matrices for {metric}/{candidate}")
            association_matrices[metric][candidate] = np.stack(block_matrices[metric][candidate])

    candidate_summary: list[dict[str, object]] = []
    for metric in METRICS:
        for candidate in CANDIDATES:
            candidate_summary.append(
                _summary_row(
                    metric,
                    candidate,
                    association_matrices[metric][candidate],
                    call_ids,
                    clusters,
                    valid_trial_rows[metric][candidate],
                )
            )

    comparisons: list[dict[str, object]] = []
    comparison_specs = (
        ("controlled_minus_raw", "amt_small_controlled", "amt_small_raw"),
        ("controlled_minus_motif", "amt_small_controlled", "motif_transform_baseline"),
        ("motif_minus_raw", "motif_transform_baseline", "amt_small_raw"),
    )
    for metric in METRICS:
        for label, left, right in comparison_specs:
            comparisons.append(
                _comparison_row(
                    metric,
                    label,
                    association_matrices[metric][left],
                    association_matrices[metric][right],
                    call_ids,
                    clusters,
                )
            )

    source_excluded_sensitivity: list[dict[str, object]] = []
    for metric in METRICS:
        masked_controlled = np.stack(
            [
                exclude_same_source_mismatches(matrix, call_ids, clusters)
                for matrix in association_matrices[metric]["amt_small_controlled"]
            ]
        )
        masked_raw = np.stack(
            [
                exclude_same_source_mismatches(matrix, call_ids, clusters)
                for matrix in association_matrices[metric]["amt_small_raw"]
            ]
        )
        sensitivity_row = _comparison_row(
            metric,
            "controlled_minus_raw",
            masked_controlled,
            masked_raw,
            call_ids,
            clusters,
        )
        source_excluded_sensitivity.append(
            {
                "sensitivity": "exclude_same_source_mismatches",
                **sensitivity_row,
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "call_response_association_trial_metrics.csv", trial_rows)
    _write_csv(args.output_dir / "candidate_association_summary.csv", candidate_summary)
    _write_csv(args.output_dir / "candidate_association_comparisons.csv", comparisons)
    _write_csv(
        args.output_dir / "source_excluded_mismatch_sensitivity.csv",
        source_excluded_sensitivity,
    )
    report = [
        "# Call--Response Feature Association Analysis",
        "",
        "This analysis measures reproducible symbolic feature association between each Call100 input and its response. It is not a measure of musical quality, listener preference, harmonic appropriateness, perceptual turn taking, or live co-performance usefulness.",
        "",
        "## Pairing and null construction",
        "",
        "All 27,000 harmonized rows were resolved through the existing local answer keys. Every result row stores only MIDI filenames and SHA-256 values; local filesystem paths are not published. For every `(preset, candidate, trial)` block, the observed pair is compared with the mean of all valid mismatched response pairings in the same 100-Call block. The per-row `*_mismatched_valid_count` records the dynamic denominator; an unavailable matched score or zero valid mismatches yields `NA`. Association excess is `actual feature similarity - mean valid-mismatched feature similarity`.",
        "",
        "For each candidate and metric, association statistics are computed separately in all six-preset-by-15-trial blocks and then averaged within each Call. The primary 95% interval is a classical leave-one-Call-out jackknife normal interval. Every deletion removes the matched Call--response unit and rebuilds every retained Call's mismatch bank in every block before recomputing and aggregating association excess. The 87-source-cluster point and interval use the analogous leave-one-source-cluster-out jackknife with the same equal-source estimand as the other 87-source analyses: average valid Calls within each retained source, then average retained source means with equal weight. Candidate contrasts use the same paired Call/source deletions, rebuild both banks in every replicate, and retain a Call only when both candidate excesses are defined after within-Call aggregation. No hypothesis-test p-values are reported.",
        "",
        "## Metrics",
        "",
        "- `contour_r`: Pearson correlation of 16-point linearly event-index-resampled onset-cluster median-pitch contours; vectors are centered and L2-normalized independently. At least three clusters and centered norm above `1e-12` are required.",
        "- `interval_2gram_dice`: multiset Dice `2*sum(min(count_call, count_response))/(ngrams_call+ngrams_response)` over adjacent signed interval-pairs. Median-pitch differences use Python ties-to-even integer rounding before clipping to [-12, 12] semitones; at least three clusters are required.",
        "- `onset_profile_cosine`: cosine similarity of 16-bin onset-cluster density profiles after normalizing each phrase by its own onset-span. Bin index is `min(15, floor(16*(t-t0)/(tlast-t0)))`; counts are L2-normalized, and at least two clusters with span above `1e-12` are required. Duration matching and grid projection can affect this feature, so it remains a control/association check only.",
        "",
        "Short or degenerate phrases receive `NA` for a metric rather than a synthetic zero; coverage is reported below.",
        "",
        "## Candidate association-excess summaries",
        "",
        *_markdown_table(
            candidate_summary,
            [
                "metric",
                "candidate",
                "valid_trial_rows",
                "trial_coverage",
                "call_count",
                "association_excess_mean",
                "association_excess_mean_call_jackknife_ci95_low",
                "association_excess_mean_call_jackknife_ci95_high",
                "association_excess_source_mean",
                "association_excess_mean_source_jackknife_ci95_low",
                "association_excess_mean_source_jackknife_ci95_high",
            ],
        ),
        "",
        "## Call- and source-level candidate contrasts on association excess",
        "",
        *_markdown_table(
            comparisons,
            [
                "metric",
                "comparison",
                "call_count",
                "mean_difference",
                "mean_difference_call_jackknife_ci95_low",
                "mean_difference_call_jackknife_ci95_high",
                "source_cluster_mean_difference",
                "mean_difference_source_jackknife_ci95_low",
                "mean_difference_source_jackknife_ci95_high",
            ],
        ),
        "",
        "## Same-source mismatch exclusion sensitivity",
        "",
        f"The primary mismatch bank includes all other Calls in the same condition, including another excerpt from the same source song. As a sensitivity analysis, the table below removes the {same_source_offdiagonal_cells_per_block} ordered off-diagonal cells per 100-Call block whose response source cluster matches the focal Call, then rebuilds every mismatch bank and repeats the paired Call/source jackknives. The matched diagonal is retained. This exclusion does not change the direction or interval interpretation of any controlled-minus-raw contrast.",
        "",
        *_markdown_table(
            source_excluded_sensitivity,
            [
                "metric",
                "comparison",
                "call_count",
                "mean_difference",
                "mean_difference_call_jackknife_ci95_low",
                "mean_difference_call_jackknife_ci95_high",
                "source_cluster_mean_difference",
                "mean_difference_source_jackknife_ci95_low",
                "mean_difference_source_jackknife_ci95_high",
            ],
        ),
        "",
        "## Interpretation boundary",
        "",
        "Positive association excess means that the selected symbolic feature is more similar for the observed Call--response pairing than for same-condition mismatches in this finite development benchmark. It does not establish musical quality, conversational appropriateness, generalization, human-perceived contingency, or causal benefit of a module.",
        "",
    ]
    (args.output_dir / "report.md").write_text("\n".join(report), encoding="utf-8")
    provenance = {
        "analysis": "call_response_feature_association_v4_source_exclusion_sensitivity",
        "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "midi_pairing": {
            "harmonized_rows": len(records),
            "calls": 100,
            "source_clusters": 87,
            "response_rows_per_candidate": 9000,
            "mapping": "answer-key call_id/trial/seed/candidate validation; response SHA-256 verified against harmonized metrics",
            "path_redaction": "published CSVs contain MIDI filenames and SHA-256 values only",
        },
        "same_condition_mismatch_null": {
            "block": ["preset", "candidate", "trial"],
            "block_calls": 100,
            "nominal_alternative_responses_per_observed_pair": 99,
            "valid_alternative_denominator": "dynamic count of available off-diagonal feature similarities; written per trial row",
            "association_excess": "actual_similarity - mean_valid_mismatched_similarity",
        },
        "same_source_exclusion_sensitivity": {
            "output": "source_excluded_mismatch_sensitivity.csv",
            "comparison": "controlled_minus_raw",
            "rule": "retain the matched diagonal; exclude an off-diagonal response when its source cluster equals the focal Call source cluster",
            "source_cluster_definition": "77 POP909 song IDs plus 10 artificial Calls",
            "ordered_same_source_offdiagonal_cells_per_100_call_block": same_source_offdiagonal_cells_per_block,
            "inference": "rebuild all masked mismatch banks for the point estimate and every paired delete-one Call/source jackknife replicate",
            "interpretation": "all three controlled-minus-raw directions and interval conclusions are unchanged",
        },
        "inference": {
            "aggregation": "rebuild block-level association statistics in all six presets and 15 trials, then average valid statistics within each Call",
            "call_ci": "classical leave-one-Call-out jackknife standard error and point_estimate +/- 1.96*SE; deletion rebuilds all retained mismatch banks",
            "source_estimate": "equal-weight source mean: average valid Calls within each retained source, then average retained source means equally",
            "source_ci": "classical leave-one-source-cluster-out jackknife standard error and equal-source point_estimate +/- 1.96*SE; deletion rebuilds all retained mismatch banks",
            "candidate_contrasts": "paired Call/source deletions with both candidate mismatch banks rebuilt in every replicate; a Call is retained only when both candidate within-Call excesses are defined; source contrast uses equal-weight retained source means",
            "resampling": "none; deterministic delete-one jackknife",
        },
        "metrics": {
            "onset_cluster_window_seconds": 0.08,
            "onset_cluster_rule": "start-anchored inclusive window; even cluster median is the arithmetic mean of the two middle pitches",
            "contour_points": 16,
            "contour_interpolation": "linear interpolation at equally spaced event-index positions; mean-centered and L2-normalized; fewer than 3 clusters or centered norm <= 1e-12 is NA",
            "interval_ngram_size": 2,
            "interval_clip_semitones": [-12, 12],
            "interval_rounding_and_dice": "Python ties-to-even round of median-pitch differences, then clipping; multiset Dice is 2*sum(min counts)/(total counts); fewer than 3 clusters is NA",
            "onset_profile_bins": 16,
            "onset_profile_rule": "min(15, floor(16*(t-t0)/(tlast-t0))); L2-normalized counts; fewer than 2 clusters or onset span <= 1e-12 is NA",
            "short_phrase_policy": "NA with coverage reported",
            "claim_boundary": "non-subjective symbolic feature association only; not musical quality",
        },
        "input_hashes": {
            "harmonized_trial_metrics.csv": _sha256(args.metrics),
            "call100_manifest_public.csv": _sha256(args.manifest),
        },
    }
    (args.output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "trial_rows": len(trial_rows),
        "candidate_summary_rows": len(candidate_summary),
        "comparison_rows": len(comparisons),
        "source_excluded_sensitivity_rows": len(source_excluded_sensitivity),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metrics",
        type=Path,
        default=REPO_ROOT / "results" / "call100_harmonized_evaluation" / "harmonized_trial_metrics.csv",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPO_ROOT / "results" / "call100_dataset" / "call100_manifest_public.csv",
    )
    parser.add_argument(
        "--ablation-root",
        type=Path,
        default=WORKSPACE_ROOT / "ab_tests" / "objective_ablation_call100_trials15_latency",
    )
    parser.add_argument(
        "--motif-root",
        type=Path,
        default=WORKSPACE_ROOT / "ab_tests" / "objective_search_call100_trials15",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "call_response_association",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run_analysis(args)
    print(
        "Call--Response association analysis complete: "
        f"{result['trial_rows']} trial rows, {result['candidate_summary_rows']} summaries, "
        f"{result['comparison_rows']} comparisons, "
        f"{result['source_excluded_sensitivity_rows']} source-excluded sensitivity rows."
    )


if __name__ == "__main__":
    main()
