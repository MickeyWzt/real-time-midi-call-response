# Real-Time MIDI Call-and-Response Generation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Release](https://img.shields.io/github/v/release/MickeyWzt/real-time-midi-call-response)](https://github.com/MickeyWzt/real-time-midi-call-response/releases)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21860065.svg)](https://doi.org/10.5281/zenodo.21860065)

Code and supporting materials for the paper **Real-Time MIDI Call-and-Response Generation Using Autoregressive Transformers** by Wang Zitong and Hu Sitong.

This repository wraps an offline autoregressive symbolic-music Transformer for live MIDI call-and-response performance. The system listens to a human MIDI phrase, detects a likely phrase endpoint, generates a response with an Anticipatory Music Transformer backend, applies phrase-level musical control, and schedules MIDI playback with latency-aware buffering.

The repository is published through GitHub Pages and archived on Zenodo. The v1.1.0 archive DOI is [10.5281/zenodo.21860065](https://doi.org/10.5281/zenodo.21860065). The stable all-versions concept DOI is [10.5281/zenodo.20838083](https://doi.org/10.5281/zenodo.20838083).

## Paper Summary

The paper asks whether an offline autoregressive Transformer can be made usable for real-time MIDI co-performance without retraining. The implementation combines:

- continuous-time MIDI event modeling
- adaptive MIDI-VAD phrase endpoint detection
- asynchronous decoding and micro-buffered playback
- phrase-level control for repetition, duration, fallback, and style constraints
- direct Call100 endpoint replay, structural evaluation, module ablation, scheduler replay, and blind listening

Key verified results included in this release:

- Harmonized Call100 comparison: 27,000 trial records, with raw A0 and controlled A6 drawn from the same seeded batch and every candidate re-scored by `structural_compliance_v1.1`.
- Controlled AMT versus raw AMT: mean structural-compliance gain `+0.171642`, 95% bootstrap CI `[0.168923, 0.174287]`; the motif baseline remains higher than controlled AMT by `0.020072`.
- A0-A6 ablation: 63,000 rows, with the composite increasing from `0.560968` to `0.732610` and the non-style component from `0.558971` to `0.676704`.
- Fallback audit: A3 produced no empty outputs, so the A4 phrase-level motif fallback activated `0/9000` times and all A3/A4 responses were byte-identical. The earlier `41.46%` value was an event-repair sample rate, not motif fallback.
- Endpoint replay: adaptive MIDI-VAD reaches F1 `0.712` at the stated two-second deadline, compared with `0.708` for fixed 800 ms, with 38 versus 68 premature commits. This uses file-end proxy boundaries, not human annotations.
- Scheduler-replay latency study: with a `150 ms` pre-commit overlap, endpoint-to-first-MIDI mean latency decreases from `161.303 ms` to `91.721 ms`, and buffer-underrun rate decreases from `20.744%` to `5.244%`.
- Blind listening: 44 complete submissions and 41 retained participants. A4 was not rated conclusively above raw AMT; the motif baseline was preferred to A6 on the eight fixed stimuli.

These layers support an engineering adaptation, structural control, and local scheduler responsiveness. They do not establish universal endpoint accuracy, faster intrinsic AMT decoding, or perceptual superiority.

## Repository Layout

```text
code/
  live_call_response.py              realtime MIDI engine
  midi_vad_endpoint.py               adaptive phrase endpoint detector
  interface_backend.py               local FastAPI/WebSocket studio backend
  static/                            browser UI for local performance
  offline_ab_test.py                 blind listening and objective sample generation
  evaluate_melody_metrics.py         objective symbolic-music metrics
  build_harmonized_call100_evaluation.py
                                      shared-batch candidate evaluation
  analyze_ablation_integrity.py      raw-label and fallback audit
  run_endpoint_benchmark.py          direct endpoint replay and baselines
  export_public_results.py           path-redacted, semantic public CSV export
  run_call100_objective_search.py    Call100 objective-search driver
  run_call100_ablation_latency.py    A0-A6 ablation and latency aggregation
  analyze_final_blind_study.py       frozen blind-study analysis
  test_evaluation_integrity.py       scoring/fallback regression tests
  test_endpoint_benchmark.py         endpoint benchmark regression tests

online_blind_listening/
  Static blind-listening interface retained for study transparency.

paper/
  Paper PDF, LaTeX source, bibliography, and system overview figure.

results/
  call100_dataset/                   Call100 manifest and validation scripts
  call100_harmonized_evaluation/     corrected 27,000-row trial metrics
  evaluation_integrity_audit/        raw-label and A3/A4 activation evidence
  endpoint_benchmark_call100/        100-call endpoint results
  call100_ablation_latency/          A0-A6 and preload latency summaries
  blind_listening_final/             privacy-preserving aggregate results

docs/
  GitHub Pages project page.
```

## What Is Not Included

This archive intentionally excludes large or license-sensitive runtime artifacts:

- model weights and Hugging Face caches
- piano sample libraries, VST plugins, DAWs, and bundled audio software
- generated MIDI responses and raw per-run output directories
- participant-level response exports, exclusion identifiers, private answer keys, and deployment credentials

Install or download third-party models, datasets, and audio tools separately according to their licenses.

## Quick Start

For a Windows collaborator setup, MIDI routing, model-file boundaries, and the
shared-branch workflow, see [COLLABORATION_WINDOWS.md](COLLABORATION_WINDOWS.md).

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

The legacy `results/call100_objective_search/` directory is retained only to make the v1.0.0 labeling error auditable; see its `LEGACY_NOTICE.md`. Manuscript claims use the harmonized files below.

Useful files:

- `results/call100_harmonized_evaluation/harmonized_trial_metrics.csv`
- `results/call100_harmonized_evaluation/paired_comparisons.csv`
- `results/call100_harmonized_evaluation/score_spec.json`
- `results/evaluation_integrity_audit/integrity_report.md`
- `results/endpoint_benchmark_call100/endpoint_condition_summary.csv`
- `results/call100_ablation_latency/ablation_summary_by_variant.csv`
- `results/call100_ablation_latency/ablation_trial_metrics.csv`
- `results/call100_ablation_latency/latency_log_all_trials.csv`
- `results/call100_ablation_latency/latency_summary_by_condition.csv`
- `results/call100_ablation_latency/preload_on_off_comparison.csv`
- `results/call100_ablation_latency/ablation_validation_summary.json`
- `results/blind_listening_final/FINAL_RESULTS.md`
- `results/blind_listening_final/provenance.json`
- `results/call100_dataset/call100_manifest_public.csv`

To rerun aggregation from available summaries:

```powershell
python -m compileall code
python code/run_call100_ablation_latency.py --help
python code/build_harmonized_call100_evaluation.py --help
python code/run_endpoint_benchmark.py --help
python code/test_evaluation_integrity.py
python code/test_endpoint_benchmark.py
```

Full regeneration requires third-party model weights and the Call100 MIDI inputs. Generated responses and model caches are excluded from the DOI archive.

## Lightweight Verification

These checks do not require model weights, MIDI hardware, or the excluded raw outputs:

```powershell
python -m pip install mido
python -m compileall code
python code/run_call100_objective_search.py --help
python code/run_call100_ablation_latency.py --help
python code/build_harmonized_call100_evaluation.py --help
python code/run_endpoint_benchmark.py --help
python code/test_evaluation_integrity.py
python code/test_endpoint_benchmark.py
```

GitHub Actions runs the same syntax, CLI, and regression checks on pushes and pull requests.

## GitHub Pages

The project page lives in `docs/index.md`. After the repository is pushed, enable GitHub Pages from the `main` branch and `/docs` folder.

Expected URL:

```text
https://mickeywzt.github.io/real-time-midi-call-response/
```

## Citation

Use `CITATION.cff` for GitHub citation metadata. Zenodo release metadata is defined in `.zenodo.json`.

```bibtex
@software{wang_hu_2026_realtime_midi_call_response,
  author = {Wang, Zitong and Hu, Sitong},
  title = {Real-Time MIDI Call-and-Response Generation Using Autoregressive Transformers},
  year = {2026},
  version = {1.1.0},
  doi = {10.5281/zenodo.21860065},
  url = {https://doi.org/10.5281/zenodo.21860065}
}
```

## License

Project code and documentation in this repository are released under the MIT License. Third-party models, datasets, papers, plugins, DAWs, and audio assets remain under their respective licenses.
