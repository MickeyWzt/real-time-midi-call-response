# Real-Time MIDI Call-and-Response Generation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Release](https://img.shields.io/github/v/release/MickeyWzt/real-time-midi-call-response)](https://github.com/MickeyWzt/real-time-midi-call-response/releases)

Code and supporting materials for the paper **Real-Time MIDI Call-and-Response Generation Using Autoregressive Transformers** by Wang Zitong and Hu Sitong.

This repository wraps an offline autoregressive symbolic-music Transformer for live MIDI call-and-response performance. The system listens to a human MIDI phrase, detects a likely phrase endpoint, generates a response with an Anticipatory Music Transformer backend, applies phrase-level musical control, and schedules MIDI playback with latency-aware buffering.

The repository is prepared for GitHub Pages and Zenodo software archiving. After Zenodo ingests a GitHub release, cite the archived software DOI in addition to the paper.

## Paper Summary

The paper asks whether an offline autoregressive Transformer can be made usable for real-time MIDI co-performance without retraining. The implementation combines:

- continuous-time MIDI event modeling
- adaptive MIDI-VAD phrase endpoint detection
- asynchronous decoding and micro-buffered playback
- phrase-level control for repetition, duration, fallback, and style constraints
- objective Call100 evaluation and latency logging

Key verified results included in this release:

- Call100 comparison: 27,000 objective trial records across raw AMT, controlled AMT, and a motif-transformation baseline.
- Controlled AMT versus raw AMT: mean objective-score gain of `+0.063907` with 95% CI `[0.061668, 0.066294]`.
- A0-A6 ablation: 63,000 rows, with mean objective score increasing from `0.560968` for raw AMT to `0.732610` for the full controlled system.
- Preload latency study: endpoint-to-first-MIDI mean latency decreases from `161.303 ms` to `85.534 ms`, and buffer underrun rate decreases from `20.744%` to `3.556%`.

These metrics are structural and engineering proxies. They support controllability and local runtime responsiveness, but they are not a substitute for formal listening tests or live-performance user studies.

## Repository Layout

```text
code/
  live_call_response.py              realtime MIDI engine
  midi_vad_endpoint.py               adaptive phrase endpoint detector
  interface_backend.py               local FastAPI/WebSocket studio backend
  static/                            browser UI for local performance
  offline_ab_test.py                 blind listening and objective sample generation
  evaluate_melody_metrics.py         objective symbolic-music metrics
  run_call100_objective_search.py    Call100 objective-search driver
  run_call100_ablation_latency.py    A0-A6 ablation and latency aggregation

online_blind_listening/
  Lightweight static interface for future listening studies.

paper/
  Paper PDF, LaTeX source, bibliography, and system overview figure.

results/
  call100_dataset/                   Call100 manifest and validation scripts
  call100_objective_search/          27,000-row comparison summaries
  call100_ablation_latency/          A0-A6 and preload latency summaries

docs/
  GitHub Pages project page.
```

## What Is Not Included

This archive intentionally excludes large or license-sensitive runtime artifacts:

- model weights and Hugging Face caches
- piano sample libraries, VST plugins, DAWs, and bundled audio software
- generated MIDI responses and raw per-run output directories
- private study answer keys and deployment endpoints

Install or download third-party models, datasets, and audio tools separately according to their licenses.

## Quick Start

Python 3.12 is recommended on Windows.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Create virtual MIDI ports such as `Python_IN` and `Python_OUT`, then run the local studio:

```powershell
python code/interface_backend.py --host 127.0.0.1 --port 8000
```

Open:

```text
http://127.0.0.1:8000
```

List MIDI ports:

```powershell
python code/live_call_response.py --list-ports
```

Run a realtime AMT session:

```powershell
python code/live_call_response.py `
  --backend amt `
  --model-id stanford-crfm/music-small-800k `
  --input-port "Python_IN" `
  --output-port "Python_OUT" `
  --monitor-input `
  --latency-mode fast `
  --musical-control `
  --live-stop-on-target-notes
```

## Reproducing The Reported Summaries

The release includes summary tables and validation outputs so the headline numbers can be checked without downloading model weights or raw generated MIDI.

Useful files:

- `results/call100_objective_search/summary_by_candidate.csv`
- `results/call100_objective_search/pairwise_bootstrap_tests.csv`
- `results/call100_objective_search/validation_summary.json`
- `results/call100_ablation_latency/ablation_summary_by_variant.csv`
- `results/call100_ablation_latency/latency_summary_by_condition.csv`
- `results/call100_ablation_latency/preload_on_off_comparison.csv`
- `results/call100_ablation_latency/ablation_validation_summary.json`

To rerun aggregation from available summaries:

```powershell
python code/run_call100_ablation_latency.py --help
python code/run_call100_objective_search.py --help
```

Full regeneration requires third-party model weights and the Call100 MIDI inputs. Generated responses and model caches are excluded from the DOI archive.

## GitHub Pages

The project page lives in `docs/index.md`. After the repository is pushed, enable GitHub Pages from the `main` branch and `/docs` folder.

Expected URL:

```text
https://mickeywzt.github.io/real-time-midi-call-response/
```

## Citation

Use `CITATION.cff` for GitHub citation metadata. Zenodo release metadata is defined in `.zenodo.json`.

Before the Zenodo DOI is minted:

```bibtex
@software{wang_hu_2026_realtime_midi_call_response,
  author = {Wang, Zitong and Hu, Sitong},
  title = {Real-Time MIDI Call-and-Response Generation Using Autoregressive Transformers},
  year = {2026},
  version = {1.0.0},
  url = {https://github.com/MickeyWzt/real-time-midi-call-response}
}
```

After Zenodo archives the GitHub release, replace the URL-only citation with the versioned Zenodo DOI.

## License

Project code and documentation in this repository are released under the MIT License. Third-party models, datasets, papers, plugins, DAWs, and audio assets remain under their respective licenses.
