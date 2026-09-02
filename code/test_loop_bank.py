import unittest

from loop_bank import LoopEvent, ResponseLoopBank


class ResponseLoopBankTests(unittest.TestCase):
    def test_saved_loop_repeats_and_stops_at_boundary(self):
        bank = ResponseLoopBank(minimum_loop_seconds=0.5)
        self.assertTrue(bank.save("A", [LoopEvent(0, "note_on", 60, 90), LoopEvent(0.2, "note_off", 60, 0)], 1.0))
        self.assertTrue(bank.toggle("A", 10.0))
        self.assertEqual([event.kind for event in bank.tick(10.0)], ["note_on"])
        self.assertEqual([event.kind for event in bank.tick(10.2)], ["note_off"])
        self.assertEqual([event.kind for event in bank.tick(11.0)], ["note_on"])
        self.assertTrue(bank.request_stop("A"))
        self.assertEqual([event.kind for event in bank.tick(12.0)], ["note_off"])
        self.assertFalse(bank.status()[0].playing)

    def test_stop_all_sends_note_offs(self):
        bank = ResponseLoopBank()
        bank.save("A", [LoopEvent(0, "note_on", 60, 90)], 1.0)
        bank.toggle("A", 0.0)
        bank.tick(0.0)
        offs = bank.stop_all_now()
        self.assertEqual([(event.kind, event.note) for event in offs], [("note_off", 60)])

    def test_overlapping_same_pitch_notes_are_released_together(self):
        bank = ResponseLoopBank()
        bank.save(
            "A",
            [
                LoopEvent(0.0, "note_on", 60, 90),
                LoopEvent(0.1, "note_on", 60, 80),
                LoopEvent(0.2, "note_off", 60, 0),
                LoopEvent(0.3, "note_off", 60, 0),
            ],
            1.0,
        )
        bank.toggle("A", 0.0)
        bank.tick(0.0)
        bank.tick(0.1)
        self.assertEqual(bank.stop_now("A"), [(LoopEvent(0.0, "note_off", 60, 0, 0))])

    def test_overwriting_playing_slot_releases_old_notes(self):
        bank = ResponseLoopBank()
        bank.save("A", [LoopEvent(0.0, "note_on", 60, 90)], 1.0)
        bank.toggle("A", 0.0)
        bank.tick(0.0)

        bank.save("A", [LoopEvent(0.0, "note_on", 64, 90)], 1.0)

        self.assertEqual(bank.tick(0.1), [LoopEvent(0.0, "note_off", 60, 0, 0)])


if __name__ == "__main__":
    unittest.main()
