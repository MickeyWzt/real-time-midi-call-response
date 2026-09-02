import unittest
from types import SimpleNamespace

from drum_loop import is_drum_pad_note


class DrumPadDetectionTests(unittest.TestCase):
    def test_low_keyboard_note_is_not_mistaken_for_pad(self) -> None:
        message = SimpleNamespace(type="note_on", note=45, velocity=100, channel=0)
        self.assertFalse(is_drum_pad_note(message, 36, 51, 9))

    def test_drum_channel_is_detected_outside_note_range(self) -> None:
        message = SimpleNamespace(type="note_on", note=60, velocity=100, channel=9)
        self.assertTrue(is_drum_pad_note(message, 36, 51, 9))

    def test_regular_melodic_note_is_not_detected(self) -> None:
        message = SimpleNamespace(type="note_on", note=60, velocity=100, channel=0)
        self.assertFalse(is_drum_pad_note(message, 36, 51, 9))

    def test_note_range_is_fallback_when_channel_is_missing(self) -> None:
        message = SimpleNamespace(type="note_on", note=45, velocity=100)
        self.assertTrue(is_drum_pad_note(message, 36, 51, 9))


if __name__ == "__main__":
    unittest.main()
