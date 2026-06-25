# Real-Time MIDI Call-and-Response Generation

Code, paper, and verified summary outputs for **Real-Time MIDI Call-and-Response Generation Using Autoregressive Transformers**.

[GitHub repository](https://github.com/MickeyWzt/real-time-midi-call-response) | [Paper PDF](../paper/Real_Time_MIDI_Call_and_Response_Generation_Using_Autoregressive_Transformers.pdf) | [Zenodo DOI](https://doi.org/10.5281/zenodo.20838084) | [Release archive](https://github.com/MickeyWzt/real-time-midi-call-response/releases)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20838084.svg)](https://doi.org/10.5281/zenodo.20838084)

![System overview](../paper/System_overview.png)

## What This Project Does

The system adapts an offline Anticipatory Music Transformer to live MIDI co-performance. It listens to a human call phrase, detects the phrase endpoint with MIDI-VAD logic, generates an AI response, applies phrase-level control, and schedules MIDI playback with a latency-aware buffer.

## Evidence Included

| Evidence layer | Scale | Main takeaway |
| --- | ---: | --- |
| Call100 objective comparison | 27,000 trials | Controlled AMT improves raw AMT by `+0.063907`, while the motif baseline remains strong. |
| A0-A6 module ablation | 63,000 rows | Full control raises mean objective score from `0.560968` to `0.732610`. |
| Preload latency logging | 18,000 rows | Speculative preload reduces mean endpoint-to-first-MIDI latency from `161.303 ms` to `85.534 ms`. |

The metrics are structural and runtime proxies. They do not replace listening tests or live user studies.

## Included Materials

- realtime MIDI engine and local browser studio
- Call100 dataset manifest and validation scripts
- objective evaluation, ablation, and latency summary tables
- paper PDF and LaTeX source
- future listening-study static interface
- citation and Zenodo metadata

Large model weights, caches, generated MIDI responses, audio sample libraries, VST plugins, and private answer keys are excluded.

## Citation

The versioned software DOI is [10.5281/zenodo.20838084](https://doi.org/10.5281/zenodo.20838084). The all-versions concept DOI is [10.5281/zenodo.20838083](https://doi.org/10.5281/zenodo.20838083).

```bibtex
@software{wang_hu_2026_realtime_midi_call_response,
  author = {Wang, Zitong and Hu, Sitong},
  title = {Real-Time MIDI Call-and-Response Generation Using Autoregressive Transformers},
  year = {2026},
  version = {1.0.0},
  doi = {10.5281/zenodo.20838084},
  url = {https://doi.org/10.5281/zenodo.20838084}
}
```
