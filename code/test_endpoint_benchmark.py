from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from midi_vad_endpoint import MidiEndpointVAD
from run_endpoint_benchmark import FixedCutoffVAD, metric_bundle


class EndpointBenchmarkTests(unittest.TestCase):
    def test_confirmation_cancels_internal_pause_candidate(self) -> None:
        candidates = []
        cancels = []
        decisions = []
        detector = FixedCutoffVAD(
            0.3,
            endpoint_confirm_delay=0.15,
            on_candidate_endpoint=candidates.append,
            on_candidate_cancel=cancels.append,
            on_endpoint=decisions.append,
        )
        detector.observe_note_on(60, timestamp=0.0)
        detector.tick(0.31)
        detector.observe_note_on(62, timestamp=0.40)
        detector.tick(0.71)
        detector.tick(0.86)
        self.assertEqual(len(candidates), 2)
        self.assertEqual(len(cancels), 1)
        self.assertEqual(len(decisions), 1)

    def test_no_confirmation_commits_during_same_pause(self) -> None:
        decisions = []
        detector = FixedCutoffVAD(
            0.3, endpoint_confirm_delay=0.0, on_endpoint=decisions.append
        )
        detector.observe_note_on(60, timestamp=0.0)
        detector.tick(0.31)
        detector.observe_note_on(62, timestamp=0.40)
        self.assertEqual(len(decisions), 1)

    def test_metric_bundle_counts_premature_first_commit_as_fp_and_fn(self) -> None:
        rows = [
            {
                "first_commit_outcome_2000ms": "premature",
                "matched_2000ms": 0,
                "false_positive_count_2000ms": 1,
                "false_negative_count_2000ms": 1,
                "endpoint_error_s_2000ms": "",
            },
            {
                "first_commit_outcome_2000ms": "success",
                "matched_2000ms": 1,
                "false_positive_count_2000ms": 0,
                "false_negative_count_2000ms": 0,
                "endpoint_error_s_2000ms": 0.5,
            },
        ]
        metrics = metric_bundle(rows, 2.0)
        self.assertAlmostEqual(metrics["precision"], 0.5)
        self.assertAlmostEqual(metrics["recall"], 0.5)
        self.assertAlmostEqual(metrics["f1"], 0.5)
        self.assertAlmostEqual(metrics["premature_sensitive_first_commit_f1"], 0.5)
        self.assertEqual(metrics["success_count"], 1.0)
        self.assertEqual(metrics["premature_count"], 1.0)
        self.assertEqual(metrics["late_count"], 0.0)
        self.assertEqual(metrics["missed_count"], 0.0)
        self.assertEqual(metrics["signed_error_effective_n"], 1.0)

    def test_density_boundary_uses_minimum_intensity(self) -> None:
        detector = MidiEndpointVAD(min_intensity=0.25)
        self.assertEqual(detector.mu_tempo(), 0.25)
        detector.observe_note_on(60, timestamp=1.0)
        self.assertEqual(detector.mu_tempo(), 0.25)


if __name__ == "__main__":
    unittest.main()
