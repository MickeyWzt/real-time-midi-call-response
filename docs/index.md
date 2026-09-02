# Turn-Based MIDI Call-and-Response with a Frozen AMT

Code, paper, and verified summary outputs for **Adapting a Frozen Anticipatory Music Transformer for Turn-Based MIDI Call-and-Response**.

[GitHub repository](https://github.com/MickeyWzt/real-time-midi-call-response) | [Paper PDF](../paper/Real_Time_MIDI_Call_and_Response_Generation_Using_Autoregressive_Transformers.pdf) | [GitHub release v1.2.0](https://github.com/MickeyWzt/real-time-midi-call-response/releases/tag/v1.2.0) | [Zenodo v1.2.0 DOI](https://doi.org/10.5281/zenodo.22241768)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22241768.svg)](https://doi.org/10.5281/zenodo.22241768)

![System overview](../paper/System_overview.png)

## What This Project Does

The system adapts an offline Anticipatory Music Transformer to live MIDI co-performance. It listens to a human call phrase, detects the phrase endpoint with MIDI-VAD logic, generates an AI response, applies phrase-level control, and schedules MIDI playback with a latency-aware buffer.

## Evidence Included

| Evidence layer | Scale | Main takeaway |
| --- | ---: | --- |
| Direct first-commit endpoint replay | 2,000 call-condition rows | Adaptive custom PS-F1 is `0.619` at the two-second deadline; fixed 800 ms is `0.500`, with 34 versus 50 premature commits. PS-F1 is premature-sensitive, not conventional event-detection F1. |
| Harmonized Call100 comparison | 27,000 trials | Shared-batch controlled AMT improves raw AMT by `+0.171642`; the motif baseline remains `0.020072` higher. |
| Call--response feature association | 27,000 matched rows | Bank-rebuilding Call and equal-source jackknives agree that control increases onset-profile association relative to raw AMT but decreases interval-2gram association; no uniform correspondence gain is claimed. |
| A0-A6 module ablation | 63,000 rows | Composite rises from `0.560968` to `0.732610`; A4 motif fallback activates `0/9000` times. |
| Preload scheduler replay | 18,000 rows | With `150 ms` of pre-commit overlap, the modeled first-generated-event readiness lower bound decreases from `101.543 ms` to `86.545 ms`; target misses decrease from `20.744%` to `5.244%`. MIDI-send and audio-onset times are not measured. |
| Human evaluation | Excluded | No human-participant data are analyzed or reported; v1.2.0 retains only an ethics notice while historical aggregates remain in the prior immutable tag. |

The combined evidence supports an engineering adaptation and measurable structural control. It does not establish universal endpoint accuracy, faster intrinsic decoding, or perceptual superiority.

## Included Materials

- realtime MIDI engine and local browser studio
- Call100 dataset manifest and validation scripts
- trial-level structural metrics, exact score specification, and integrity audits
- endpoint, Call--response feature-association, ablation, and scheduler-replay summaries
- ethics notice for the excluded formative listening exercise; aggregate outputs are removed from v1.2.0
- paper PDF and LaTeX source
- citation and Zenodo metadata

Large model weights, generated MIDI responses, participant-level exports, exclusion identifiers, audio sample libraries, VST plugins, private answer keys, and deployment credentials are excluded.

## Citation

GitHub release [v1.2.0](https://github.com/MickeyWzt/real-time-midi-call-response/releases/tag/v1.2.0) and immutable DOI [10.5281/zenodo.22241768](https://doi.org/10.5281/zenodo.22241768) identify this revision. The stable all-versions Zenodo concept DOI remains [10.5281/zenodo.20838083](https://doi.org/10.5281/zenodo.20838083).

v1.2.0 citation using the stable all-versions DOI:

```bibtex
@software{wang_hu_2026_realtime_midi_call_response,
  author = {Wang, Zitong and Hu, Sitong},
  title = {Adapting a Frozen Anticipatory Music Transformer for Turn-Based MIDI Call-and-Response},
  year = {2026},
  version = {1.2.0},
  doi = {10.5281/zenodo.22241768},
  url = {https://doi.org/10.5281/zenodo.22241768}
}
```
