"""Regression tests for the non-subjective Call--Response association analysis."""

from __future__ import annotations

import unittest

import numpy as np

from analyze_call_response_association import (
    OnsetNote,
    _association_statistics,
    _equal_source_mean,
    _optional_association_excesses,
    association_excesses,
    association_vectors_from_counts,
    cluster_onsets,
    contour_correlation,
    exclude_same_source_mismatches,
    interval_2gram_dice,
    onset_profile_cosine,
)


class CallResponseAssociationMetricTests(unittest.TestCase):
    def test_cluster_onsets_uses_detector_start_anchored_80ms_window(self) -> None:
        notes = [
            OnsetNote(onset_seconds=0.000, pitch=60),
            OnsetNote(onset_seconds=0.079, pitch=64),
            OnsetNote(onset_seconds=0.081, pitch=67),
            OnsetNote(onset_seconds=0.161, pitch=69),
        ]

        clusters = cluster_onsets(notes, cluster_window_seconds=0.08)

        self.assertEqual(len(clusters), 2)
        self.assertEqual(clusters[0].onset_seconds, 0.0)
        self.assertEqual(clusters[0].median_pitch, 62.0)
        self.assertEqual(clusters[1].onset_seconds, 0.081)
        self.assertEqual(clusters[1].median_pitch, 68.0)

    def test_contour_correlation_is_transposition_and_register_invariant(self) -> None:
        call = [(0.0, 60.0), (0.2, 62.0), (0.5, 65.0), (0.8, 64.0)]
        response = [(0.0, 72.0), (0.3, 74.0), (0.6, 77.0), (1.0, 76.0)]

        score = contour_correlation(call, response, points=16)

        self.assertIsNotNone(score)
        self.assertAlmostEqual(score, 1.0, places=12)

    def test_contour_correlation_marks_short_or_constant_sequences_not_available(self) -> None:
        self.assertIsNone(contour_correlation([(0.0, 60.0), (0.5, 62.0)], [(0.0, 72.0), (0.5, 74.0)]))
        self.assertIsNone(
            contour_correlation(
                [(0.0, 60.0), (0.5, 60.0), (1.0, 60.0)],
                [(0.0, 72.0), (0.5, 74.0), (1.0, 76.0)],
            )
        )

    def test_interval_2gram_dice_is_transposition_invariant(self) -> None:
        call = [(0.0, 60.0), (0.2, 62.0), (0.4, 65.0), (0.6, 64.0)]
        response = [(0.0, 72.0), (0.2, 74.0), (0.4, 77.0), (0.6, 76.0)]

        self.assertAlmostEqual(interval_2gram_dice(call, response), 1.0, places=12)

    def test_interval_2gram_dice_returns_none_without_two_intervals(self) -> None:
        self.assertIsNone(interval_2gram_dice([(0.0, 60.0), (0.2, 62.0)], [(0.0, 72.0), (0.2, 74.0)]))

    def test_onset_profile_cosine_is_invariant_to_phrase_duration_scale(self) -> None:
        call = [(0.0, 60.0), (0.25, 62.0), (0.75, 64.0), (1.0, 65.0)]
        response = [(0.0, 72.0), (0.5, 74.0), (1.5, 76.0), (2.0, 77.0)]

        self.assertAlmostEqual(onset_profile_cosine(call, response, bins=16), 1.0, places=12)

    def test_association_excesses_compare_the_diagonal_to_all_same_condition_mismatches(self) -> None:
        matrix = [[1.0, 0.10, 0.30], [0.20, 0.90, 0.40], [0.50, 0.60, 0.80]]

        actual, mismatched, excess = association_excesses(matrix)

        self.assertEqual(actual, [1.0, 0.9, 0.8])
        for observed, expected in zip(mismatched, [0.2, 0.3, 0.55]):
            self.assertAlmostEqual(observed, expected, places=12)
        for observed, expected in zip(excess, [0.8, 0.6, 0.25]):
            self.assertAlmostEqual(observed, expected, places=12)

    def test_association_excesses_rejects_non_square_or_singleton_blocks(self) -> None:
        with self.assertRaises(ValueError):
            association_excesses([[1.0]])
        with self.assertRaises(ValueError):
            association_excesses([[1.0, 0.0]])

    def test_optional_association_uses_the_count_of_valid_mismatches(self) -> None:
        matrix = [
            [1.0, None, 0.4],
            [None, 0.8, None],
            [0.2, None, 0.7],
        ]

        values = _optional_association_excesses(matrix)

        self.assertEqual(values[0][:2] + (values[0][3],), (1.0, 0.4, 1))
        self.assertAlmostEqual(values[0][2], 0.6, places=12)
        self.assertEqual(values[1], (None, None, None, 0))
        self.assertEqual(values[2][:2] + (values[2][3],), (0.7, 0.2, 1))
        self.assertAlmostEqual(values[2][2], 0.5, places=12)

    def test_optional_association_records_valid_mismatch_count_when_matched_score_is_missing(self) -> None:
        matrix = [
            [1.0, 0.2, 0.4],
            [0.3, None, 0.7],
            [0.5, 0.6, 0.8],
        ]

        values = _optional_association_excesses(matrix)

        self.assertEqual(values[1], (None, None, None, 2))

    def test_resampled_units_rebuild_the_mismatch_bank_before_association_aggregation(self) -> None:
        matrix = [
            [1.0, 0.10, 0.30],
            [0.20, 0.90, 0.40],
            [0.50, 0.60, 0.80],
        ]

        actual, mismatched, excess = association_vectors_from_counts(matrix, [2, 1, 0])

        self.assertAlmostEqual(actual[0], 1.0, places=12)
        self.assertAlmostEqual(mismatched[0], 0.55, places=12)
        self.assertAlmostEqual(excess[0], 0.45, places=12)
        self.assertAlmostEqual(actual[1], 0.90, places=12)
        self.assertAlmostEqual(mismatched[1], 0.20, places=12)
        self.assertAlmostEqual(excess[1], 0.70, places=12)
        self.assertIsNone(actual[2])
        self.assertIsNone(mismatched[2])
        self.assertIsNone(excess[2])

    def test_deleted_call_removes_its_response_from_every_retained_mismatch_bank(self) -> None:
        matrix = [
            [1.0, 0.10, 0.30],
            [0.20, 0.90, 0.40],
            [0.50, 0.60, 0.80],
        ]

        actual, mismatched, excess = association_vectors_from_counts(matrix, [1, 0, 1])

        self.assertAlmostEqual(actual[0], 1.0, places=12)
        self.assertAlmostEqual(mismatched[0], 0.30, places=12)
        self.assertAlmostEqual(excess[0], 0.70, places=12)
        self.assertIsNone(actual[1])
        self.assertAlmostEqual(actual[2], 0.80, places=12)
        self.assertAlmostEqual(mismatched[2], 0.50, places=12)
        self.assertAlmostEqual(excess[2], 0.30, places=12)

    def test_same_source_sensitivity_masks_only_off_diagonal_alternatives(self) -> None:
        matrix = [
            [1.0, 0.10, 0.30],
            [0.20, 0.90, 0.40],
            [0.50, 0.60, 0.80],
        ]
        call_ids = ["call_a", "call_b", "call_c"]
        clusters = {
            "call_a": "source_1",
            "call_b": "source_1",
            "call_c": "source_2",
        }

        masked = exclude_same_source_mismatches(matrix, call_ids, clusters)

        self.assertAlmostEqual(masked[0, 0], 1.0, places=12)
        self.assertTrue(np.isnan(masked[0, 1]))
        self.assertTrue(np.isnan(masked[1, 0]))
        self.assertAlmostEqual(masked[0, 2], 0.30, places=12)
        self.assertAlmostEqual(masked[2, 0], 0.50, places=12)

    def test_block_level_statistics_apply_dynamic_denominators_before_call_aggregation(self) -> None:
        first_block = [
            [1.0, 0.10, 0.50],
            [0.20, 0.90, 0.40],
            [0.50, 0.60, 0.80],
        ]
        second_block = [
            [1.0, 0.90, None],
            [0.20, 0.90, 0.40],
            [0.50, 0.60, 0.80],
        ]

        statistics = _association_statistics([first_block, second_block], [1, 1, 1])

        # Call 0 has excess 0.70 in block 1 and 0.10 in block 2.  Averaging
        # pairwise scores first would incorrectly use an implicit fixed bank.
        self.assertAlmostEqual(statistics["actual"][0], 1.0, places=12)
        self.assertAlmostEqual(statistics["mismatched"][0], 0.60, places=12)
        self.assertAlmostEqual(statistics["excess"][0], 0.40, places=12)

    def test_source_estimand_weights_sources_equally_after_within_source_call_mean(self) -> None:
        source_mean, source_count = _equal_source_mean(
            [0.0, 0.2, 1.0],
            ["call_a", "call_b", "call_c"],
            {"call_a": "source_1", "call_b": "source_1", "call_c": "source_2"},
        )

        self.assertEqual(source_count, 2)
        self.assertAlmostEqual(source_mean, 0.55, places=12)


if __name__ == "__main__":
    unittest.main()
