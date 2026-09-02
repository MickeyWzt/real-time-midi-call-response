import threading
import unittest
from types import SimpleNamespace

from live_call_response import (
    CapturedNote,
    LiveCallResponseApp,
    MusicalResponseController,
    PhraseRecorder,
    ResponsePlan,
    GeneratedEvent,
    analyze_call_phrase,
    apply_live_pentatonic_style,
    build_partial_motif_completion,
    build_rescue_response_events,
    build_response_plan,
    partial_fallback_limit,
    project_live_event_to_call_scale,
    quantize_onset_to_rhythm_grid,
    should_capture_rhythm_pad,
    should_use_motif_fallback,
)
from midi_vad_endpoint import EndpointDecision, MidiNoteEvent


class MusicalResponseTests(unittest.TestCase):
    def test_full_motif_fallback_is_only_for_empty_amt_output(self):
        allowed = should_use_motif_fallback(
            generated_count=0,
            fallback_on_empty=True,
            backend="amt",
            response_strategy="controlled_amt",
            stopped=False,
        )

        self.assertTrue(allowed)
        self.assertTrue(
            should_use_motif_fallback(
                generated_count=0,
                fallback_on_empty=True,
                backend="amt",
                response_strategy="streaming_amt",
                stopped=False,
            )
        )
        for override in (
            {"generated_count": 1},
            {"fallback_on_empty": False},
            {"backend": "aria"},
            {"response_strategy": "motif"},
            {"stopped": True},
        ):
            inputs = {
                "generated_count": 0,
                "fallback_on_empty": True,
                "backend": "amt",
                "response_strategy": "controlled_amt",
                "stopped": False,
            }
            inputs.update(override)
            self.assertFalse(should_use_motif_fallback(**inputs))

    def test_phrase_scale_prefers_played_collection(self):
        notes = [
            CapturedNote(index * 0.25, pitch, 90, duration=0.18)
            for index, pitch in enumerate((60, 62, 64, 67, 69, 72))
        ]

        profile = analyze_call_phrase(notes)

        self.assertEqual(profile.scale_root, 0)
        self.assertEqual(profile.scale_mode, "major")

    def test_empty_model_fallback_uses_published_tail_inversion(self):
        notes = [
            CapturedNote(index * 0.25, pitch, 92, duration=0.16)
            for index, pitch in enumerate((60, 62, 64, 65))
        ]
        profile = analyze_call_phrase(notes)
        plan = ResponsePlan(2.0, 4, 48, 84, 2, 0.35, 8)

        events = build_rescue_response_events(notes, profile, plan, None, 0.25, 0.04, 2.5)

        self.assertEqual(len(events), 4)
        center = round(profile.mean_pitch)
        self.assertEqual(
            [event.pitch for event in events],
            [center - (note.pitch - center) + 5 for note in notes],
        )
        self.assertEqual(events[0].tick, 0)
        self.assertTrue(all(later.tick > earlier.tick for earlier, later in zip(events, events[1:])))

    def test_controller_preserves_chromatic_model_output_within_register(self):
        notes = [CapturedNote(0.0, 60, 90, duration=0.2), CapturedNote(0.3, 64, 90, duration=0.2)]
        profile = analyze_call_phrase(notes)
        plan = ResponsePlan(2.0, 4, 55, 72, 2, 0.35, 8)
        controller = MusicalResponseController(profile, plan)
        first = controller.accept(GeneratedEvent(0, 20, 60, 0, 120))
        chromatic = controller.accept(GeneratedEvent(25, 20, 61, 0, 127))
        clipped = controller.accept(GeneratedEvent(50, 20, 90, 0, 127))

        self.assertEqual(first.pitch, 60)
        self.assertEqual(chromatic.pitch, 61)
        self.assertEqual(clipped.pitch, 72)
        self.assertEqual(chromatic.velocity, 127)

    def test_controller_rejects_excessive_same_pitch_run(self):
        notes = [CapturedNote(0.0, 60, 90, duration=0.2)]
        profile = analyze_call_phrase(notes)
        plan = ResponsePlan(2.0, 4, 48, 72, 2, 0.35, 8)
        controller = MusicalResponseController(profile, plan)
        controller.accept(GeneratedEvent(0, 20, 60, 0, 90))
        controller.accept(GeneratedEvent(25, 20, 60, 0, 90))

        self.assertEqual(
            controller.rejection_reason(GeneratedEvent(50, 20, 60, 0, 90)),
            "repeat",
        )

    def test_live_paper_style_projects_scale_limits_leaps_and_adds_cadence(self):
        notes = [
            CapturedNote(index * 0.2, pitch, 90, duration=0.12)
            for index, pitch in enumerate((60, 62, 64, 67, 69, 67, 64, 62, 60))
        ]
        profile = analyze_call_phrase(notes)
        plan = ResponsePlan(2.0, 6, 48, 81, 2, 0.35, 8)
        raw = [
            GeneratedEvent(index * 25, 20, pitch, 0, 90)
            for index, pitch in enumerate((61, 66, 65, 65, 71, 63))
        ]

        styled = apply_live_pentatonic_style(raw, profile, plan)

        allowed = {0, 2, 4, 7, 9}
        self.assertTrue(all(event.pitch % 12 in allowed for event in styled))
        self.assertTrue(all(abs(b.pitch - a.pitch) <= 7 for a, b in zip(styled, styled[1:])))
        self.assertTrue(all(a.pitch != b.pitch for a, b in zip(styled, styled[1:])))
        self.assertEqual(styled[-1].pitch % 12, 0)

    def test_short_call_can_receive_a_compact_response(self):
        profile = analyze_call_phrase([CapturedNote(0.0, 60, 92, duration=0.12)])
        plan = build_response_plan(
            profile=profile,
            response_seconds=3.0,
            max_events=12,
            pitch_min=36,
            pitch_max=96,
            response_length_ratio=1.0,
            response_note_ratio=1.0,
            same_pitch_limit=2,
            dominant_pitch_max_share=0.35,
            resample_attempts=8,
        )

        self.assertEqual(plan.target_seconds, 0.85)

    def test_partial_fallback_never_outnumbers_amt_events(self):
        self.assertEqual(partial_fallback_limit(0, 12, 0.5), 0)
        self.assertEqual(partial_fallback_limit(2, 12, 0.5), 2)
        self.assertEqual(partial_fallback_limit(7, 12, 0.5), 5)
        self.assertEqual(partial_fallback_limit(12, 12, 0.5), 0)

    def test_partial_motif_completion_starts_after_streamed_amt(self):
        amt = [
            GeneratedEvent(100, 20, 60, 0),
            GeneratedEvent(140, 20, 64, 0),
            GeneratedEvent(180, 20, 67, 0),
        ]
        template = [
            GeneratedEvent(0, 16, 62, 0),
            GeneratedEvent(25, 16, 65, 0),
            GeneratedEvent(50, 16, 69, 0),
        ]

        completed = build_partial_motif_completion(
            amt,
            template,
            fill_count=2,
            grid_ticks=25,
            controller=None,
        )

        self.assertEqual(len(completed), 2)
        self.assertEqual([event.tick for event in completed], [205, 230])
        self.assertTrue(all(event.tick > amt[-1].tick for event in completed))

    def test_streaming_key_projection_uses_call_key_not_fixed_c(self):
        profile = analyze_call_phrase(
            [
                CapturedNote(0.0, 62, 92, duration=0.25),
                CapturedNote(0.4, 66, 92, duration=0.25),
                CapturedNote(0.8, 69, 92, duration=0.25),
                CapturedNote(1.2, 74, 92, duration=0.25),
            ]
        )
        plan = build_response_plan(
            profile=profile,
            response_seconds=3.0,
            max_events=12,
            pitch_min=36,
            pitch_max=96,
            response_length_ratio=1.0,
            response_note_ratio=1.0,
            same_pitch_limit=2,
            dominant_pitch_max_share=0.35,
            resample_attempts=8,
        )

        projected = project_live_event_to_call_scale(
            GeneratedEvent(100, 20, 61, 0),
            profile,
            plan,
            previous_pitch=None,
        )

        self.assertIn(projected.pitch % 12, {2, 4, 6, 9, 11})
        self.assertNotEqual(projected.pitch % 12, 0)

    def test_streaming_amt_yields_neural_events_without_batch_shaping(self):
        class FakeGenerator:
            def generate_events(self, *_args, **_kwargs):
                yield GeneratedEvent(100, 20, 60, 0)
                yield GeneratedEvent(125, 20, 64, 0)
                yield GeneratedEvent(150, 20, 67, 0)

        app = LiveCallResponseApp.__new__(LiveCallResponseApp)
        app.args = SimpleNamespace(amt_generation_budget=1.0)
        app.stop_event = threading.Event()
        app.generator = FakeGenerator()

        events = list(
            app._stream_budgeted_amt_events(
                call_events=[],
                response_seconds=2.0,
                target_count=2,
            )
        )

        self.assertEqual([event.pitch for event in events], [60, 64])

    def test_learned_rhythm_pulls_response_onsets_to_sixteenth_grid(self):
        # At 120 BPM the 1/16 grid is 0.125 s. Full-strength quantization
        # moves a 0.31 s onset to the nearest grid point at 0.25 s.
        self.assertAlmostEqual(
            quantize_onset_to_rhythm_grid(0.31, bpm=120.0, strength=1.0),
            0.25,
        )
        self.assertAlmostEqual(
            quantize_onset_to_rhythm_grid(0.31, bpm=120.0, strength=0.0),
            0.31,
        )

    def test_rhythm_learning_only_captures_the_dedicated_pad_lane(self):
        class MidiMessage:
            def __init__(self, note, channel):
                self.type = "note_on"
                self.note = note
                self.channel = channel

        self.assertFalse(should_capture_rhythm_pad(MidiMessage(60, 0), True, 36, 51, 9))
        self.assertTrue(should_capture_rhythm_pad(MidiMessage(36, 9), True, 36, 51, 9))
        self.assertFalse(should_capture_rhythm_pad(MidiMessage(36, 9), False, 36, 51, 9))

    def test_phrase_completed_during_ai_playback_is_queued_then_handed_off(self):
        # Build only the small live-state surface needed for this concurrency
        # test; loading an AMT model is irrelevant here.
        app = LiveCallResponseApp.__new__(LiveCallResponseApp)
        app.recorder = PhraseRecorder(default_duration=0.1)
        app.busy = threading.Event()
        app.busy.set()
        app.pending_lock = threading.Lock()
        app.pending_response = None
        app.round_id = 0
        app.round_lock = threading.Lock()

        app.recorder.note_on(64, 92, 0, 10.0)
        app.recorder.note_off(64, 0, 10.2)
        decision = EndpointDecision(
            phrase=[MidiNoteEvent(timestamp=10.0, pitch=64, velocity=92)],
            cut_time=10.3,
            last_event_time=10.0,
            silence=0.3,
            mu_tempo=2.0,
            tau_cutoff=0.3,
            survival=0.05,
            theta=0.05,
        )

        app.on_endpoint(decision)

        self.assertIsNotNone(app.pending_response)
        queued_phrase, queued_decision = app.pending_response
        self.assertEqual([note.pitch for note in queued_phrase], [64])
        self.assertIs(queued_decision, decision)

        started = []
        app._start_response_cycle = lambda phrase, endpoint, preload=None, **kwargs: started.append(
            (phrase, endpoint, preload, kwargs)
        )
        app._start_pending_response_or_mark_idle()

        self.assertEqual(len(started), 1)
        self.assertEqual([note.pitch for note in started[0][0]], [64])
        self.assertTrue(started[0][3]["queued"])
        self.assertTrue(app.busy.is_set())
        self.assertIsNone(app.pending_response)


if __name__ == "__main__":
    unittest.main()
