import unittest
from types import SimpleNamespace

from rhythm_guide import RhythmGuideEngine


class RhythmGuideTests(unittest.TestCase):
    def test_channel_one_minilab_notes_are_captured_as_rhythm_taps(self) -> None:
        engine = RhythmGuideEngine(default_bpm=120)
        engine.start_learning(0.0)
        for timestamp in (0.0, 0.5, 1.0, 1.5):
            captured = engine.capture(
                SimpleNamespace(type="note_on", velocity=96, channel=0, note=60),
                timestamp,
            )
            self.assertTrue(captured)

        self.assertEqual(engine.status().tap_count, 4)
        self.assertTrue(engine.finish_learning(1.7))
        self.assertAlmostEqual(engine.status().bpm, 120.0, delta=0.5)

    def test_repeated_start_does_not_erase_captured_taps(self) -> None:
        engine = RhythmGuideEngine(default_bpm=120)
        self.assertTrue(engine.start_learning(0.0))
        engine.capture_tap(96, 0.0)

        self.assertFalse(engine.start_learning(0.2))
        self.assertEqual(engine.status().tap_count, 1)

    def test_single_tap_does_not_create_a_loop(self) -> None:
        engine = RhythmGuideEngine(default_bpm=120, auto_finish_seconds=1.0)
        engine.start_learning(0.0)
        engine.capture_tap(96, 0.0)

        self.assertFalse(engine.maybe_finish_learning(2.0))
        self.assertFalse(engine.finish_learning(2.0))
        self.assertTrue(engine.status().learning)

    def test_long_demonstration_is_folded_into_one_bar(self) -> None:
        engine = RhythmGuideEngine(default_bpm=120)
        engine.start_learning(0.0)
        for index in range(22):
            engine.capture_tap(96, index * 0.375)

        self.assertTrue(engine.finish_learning(8.0))
        status = engine.status()
        self.assertEqual(status.bars, 1)
        self.assertLessEqual(max(status.pattern), 15)
        self.assertLess(status.loop_seconds, 4.5)

    def test_quarter_note_taps_learn_120_bpm(self) -> None:
        engine = RhythmGuideEngine(default_bpm=100)
        engine.start_learning(0.0)
        for timestamp in (0.0, 0.5, 1.0, 1.5):
            engine.capture(SimpleNamespace(type="note_on", velocity=100), timestamp)
        self.assertTrue(engine.finish_learning(1.7))

        status = engine.status()
        self.assertAlmostEqual(status.bpm, 120.0, delta=0.5)
        self.assertEqual(status.pattern, (0, 4, 8, 12))
        self.assertTrue(status.playing)

    def test_human_timing_jitter_does_not_select_half_tempo(self) -> None:
        engine = RhythmGuideEngine(default_bpm=100)
        engine.start_learning(0.0)
        for timestamp in (0.0, 0.456, 0.957, 1.459):
            engine.capture_tap(96, timestamp)
        engine.finish_learning(1.7)

        status = engine.status()
        self.assertAlmostEqual(status.bpm, 119.8, delta=1.0)
        self.assertEqual(status.pattern, (0, 4, 8, 12))

    def test_taps_generate_automatic_kit_pattern(self) -> None:
        engine = RhythmGuideEngine(default_bpm=120)
        engine.start_learning(0.0)
        for timestamp in (0.0, 0.5, 1.0, 1.5):
            engine.capture_tap(100, timestamp)
        engine.finish_learning(1.7)

        events = engine.tick(1.79)
        notes = {event.note for event in events if event.kind == "note_on"}
        self.assertIn(36, notes)
        self.assertIn(42, notes)

    def test_learning_auto_finishes_after_silence(self) -> None:
        engine = RhythmGuideEngine(default_bpm=120, auto_finish_seconds=1.0)
        engine.start_learning(0.0)
        engine.capture_tap(90, 0.0)
        engine.capture_tap(90, 0.5)
        engine.capture_tap(90, 1.0)

        self.assertFalse(engine.maybe_finish_learning(1.99))
        self.assertTrue(engine.maybe_finish_learning(2.01))
        self.assertTrue(engine.status().playing)

    def test_relearn_queues_replacement_at_boundary(self) -> None:
        engine = RhythmGuideEngine(default_bpm=120)
        engine.start_learning(0.0)
        for timestamp in (0.0, 0.5, 1.0, 1.5):
            engine.capture_tap(96, timestamp)
        engine.finish_learning(1.7)

        engine.start_learning(1.8)
        for timestamp in (1.8, 2.2, 2.6, 3.0):
            engine.capture_tap(96, timestamp)
        engine.finish_learning(3.1)
        self.assertTrue(engine.status().replacing)

    def test_stop_at_boundary_clears_pattern(self) -> None:
        engine = RhythmGuideEngine(default_bpm=120)
        engine.start_learning(0.0)
        for timestamp in (0.0, 0.5, 1.0, 1.5):
            engine.capture_tap(96, timestamp)
        engine.finish_learning(1.7)
        self.assertTrue(engine.request_stop())

        engine.tick(4.0)
        self.assertFalse(engine.status().playing)

    def test_rhythm_slot_can_be_saved_and_recalled(self) -> None:
        engine = RhythmGuideEngine(default_bpm=120)
        engine.start_learning(0.0)
        for timestamp in (0.0, 0.5, 1.0, 1.5):
            engine.capture_tap(96, timestamp)
        engine.finish_learning(1.7)

        self.assertTrue(engine.save_slot("A"))
        engine.emergency_stop()
        self.assertTrue(engine.load_slot("A", 3.0))
        self.assertEqual(engine.status().current_slot, "A")
        self.assertEqual(engine.status().saved_slots, ("A",))

    def test_loading_slot_while_playing_waits_for_boundary(self) -> None:
        engine = RhythmGuideEngine(default_bpm=120)
        engine.start_learning(0.0)
        for timestamp in (0.0, 0.5, 1.0, 1.5):
            engine.capture_tap(96, timestamp)
        engine.finish_learning(1.7)
        engine.save_slot("A")

        self.assertTrue(engine.load_slot("A", 2.0))
        self.assertEqual(engine.status().queued_slot, "A")
        engine.tick(4.0)
        self.assertEqual(engine.status().current_slot, "A")

    def test_variation_stays_inside_one_bar(self) -> None:
        engine = RhythmGuideEngine(default_bpm=120)
        engine.start_learning(0.0)
        for timestamp in (0.0, 0.5, 1.0, 1.5):
            engine.capture_tap(96, timestamp)
        engine.finish_learning(1.7)

        self.assertTrue(engine.create_variation(2.0))
        self.assertTrue(all(0 <= step < 16 for step in engine.status().pattern))


if __name__ == "__main__":
    unittest.main()
