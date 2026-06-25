# Call100 Public Final Dataset Report

- dataset_version: `call100_public_final_v1`
- num_calls: `100`
- base copied from calls_50: `50`
- public additions: `50`

## Source Dataset Distribution

| value | count |
|---|---:|
| POP909 | 50 |
| calls_50 | 50 |

## Public Addition Source Plan vs Actual

| source_dataset | target | actual | status |
|---|---:|---:|---|
| POP909 | 10 | 50 | available but target not exact |
| MAESTRO_v3_midi_only | 15 | 0 | missing locally; skipped and backfilled from available public MIDI |
| ASAP | 10 | 0 | missing locally; skipped and backfilled from available public MIDI |
| Lakh_MIDI_Clean_or_Slakh2100 | 10 | 0 | missing locally; skipped and backfilled from available public MIDI |
| GiantMIDI_or_MAESTRO_complex | 5 | 0 | missing locally; skipped and backfilled from available public MIDI |

## Category Distribution

| value | count |
|---|---:|
| real_two_bar_monophonic_major_minor | 40 |
| clear_motif_melody | 12 |
| chord_arpeggio_polyphonic | 8 |
| expressive_rubato | 8 |
| chromatic_outside | 5 |
| dense_fast | 5 |
| sparse_short | 5 |
| false_ending_pause | 4 |
| repetition_tail | 3 |
| antecedent_half_cadence | 1 |
| arched_contour | 1 |
| blues_chromatic | 1 |
| dense_motion | 1 |
| large_leap_gap_fill | 1 |
| modal_question | 1 |
| pentatonic_clear_motif | 1 |
| repetition_stress_test | 1 |
| sparse_cadential | 1 |
| syncopation | 1 |

## Required Public Category Quotas

| category | quota | actual_public_additions |
|---|---:|---:|
| clear_motif_melody | 12 | 12 |
| expressive_rubato | 8 | 8 |
| chord_arpeggio_polyphonic | 8 | 8 |
| sparse_short | 5 | 5 |
| dense_fast | 5 | 5 |
| chromatic_outside | 5 | 5 |
| false_ending_pause | 4 | 4 |
| repetition_tail | 3 | 3 |

## Feature Summary

| feature | min | mean | max |
|---|---:|---:|---:|
| note_count | 3 | 13.95 | 49 |
| duration_sec | 1.035 | 4.667 | 8 |
| pitch_range | 2 | 14.13 | 58 |
| onset_density | 0.875 | 3.814 | 30.37 |

## Polyphony Rate Distribution

| value | count |
|---|---:|
| mono_or_near_mono | 74 |
| high_polyphony | 24 |
| light_polyphony | 1 |
| medium_polyphony | 1 |

## Scale Fit Distribution

| value | count |
|---|---:|
| >=0.95 | 91 |
| 0.85-0.95 | 4 |
| 0.70-0.85 | 3 |
| <0.70 | 2 |

## Missing Sources

- `MAESTRO_v3_midi_only`: No local directory or MIDI file matching maestro
- `ASAP`: No local directory or MIDI file matching asap
- `Lakh_MIDI_Clean_or_Slakh2100`: No local directory or MIDI file matching lakh, lmd, slakh
- `GiantMIDI_or_MAESTRO_complex`: No local directory or MIDI file matching giant, maestro

## Selection Notes

- Skipped missing requested sources (ASAP, GiantMIDI_or_MAESTRO_complex, Lakh_MIDI_Clean_or_Slakh2100, MAESTRO_v3_midi_only) and redistributed public additions to available source(s): POP909.

## License And Citation Notes

- `calls_50`: copied from the existing local benchmark directory; no files in calls_50 were modified.
- `POP909`: local repository includes an MIT LICENSE file and README citation for `pop909-ismir2020`.
- Other requested public datasets were not imported in this run unless listed with nonzero actual counts above.
- Manifest rows carry `license_note` and `citation_key` per call.

## Found Local Source Roots

- `POP909`: available locally during dataset construction; source files are not redistributed in this code archive.
