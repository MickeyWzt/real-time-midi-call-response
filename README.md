# Turn-Based MIDI Call-and-Response with a Frozen AMT

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Release](https://img.shields.io/github/v/release/MickeyWzt/real-time-midi-call-response)](https://github.com/MickeyWzt/real-time-midi-call-response/releases)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20838083.svg)](https://doi.org/10.5281/zenodo.20838083)

Code and supporting materials for the paper **Adapting a Frozen Anticipatory Music Transformer for Turn-Based MIDI Call-and-Response** by Wang Zitong and Hu Sitong.

This repository wraps an offline autoregressive symbolic-music Transformer for live MIDI call-and-response performance. The system listens to a human MIDI phrase, detects a likely phrase endpoint, generates a response with an Anticipatory Music Transformer backend, applies phrase-level musical control, and schedules MIDI playback with latency-aware buffering.

The repository is published through GitHub Pages and archived on Zenodo. GitHub release [v1.2.0](https://github.com/MickeyWzt/real-time-midi-call-response/releases/tag/v1.2.0) identifies this revision; the stable all-versions Zenodo concept DOI is [10.5281/zenodo.20838083](https://doi.org/10.5281/zenodo.20838083). The exact v1.2.0 version DOI is recorded on the GitHub release page after archival.

## Paper Summary

The paper asks what an inspectable deployment layer adds to a frozen autoregressive Transformer for turn-based MIDI interaction. The implementation combines:

- continuous-time MIDI event modeling
- adaptive MIDI-VAD phrase endpoint detection
- asynchronous decoding and micro-buffered playback
- phrase-level control for repetition, duration, fallback, and style constraints
- first-commit Call100 endpoint replay, structural evaluation, Call--response feature association, module ablation, and scheduler replay

Key verified results in v1.2.0:

- Harmonized Call100 comparison: 27,000 trial records, with raw A0 and controlled A6 drawn from the same seeded batch and every candidate re-scored by `structural_compliance_v1.1`.
- Controlled AMT versus raw AMT: mean structural-compliance gain `+0.171642`, 95% Call-cluster bootstrap CI `[0.157901, 0.185151]`; the motif baseline remains higher than controlled AMT by `0.020072`.
- Matched-versus-mismatched feature audit: the controlled-minus-raw onset-profile association contrast is `+0.020016` (bank-rebuilding Call-jackknife CI `[0.006927, 0.033105]`), whereas the interval-2gram contrast is `-0.096715` (`[-0.113257, -0.080172]`). Equal-source jackknife analyses preserve both directions. These are narrow symbolic associations, not musical-quality scores.
- Same-source-mismatch sensitivity: excluding off-diagonal responses from the focal Call's source cluster leaves the controlled-minus-raw directions unchanged (interval `-0.096742`, onset `+0.020266`, contour inconclusive).
- A0-A6 ablation: 63,000 rows, with the composite increasing from `0.560968` to `0.732610` and the remaining-components score from `0.558971` to `0.676704`.
- Fallback audit: A3 produced no empty outputs, so the A4 phrase-level motif fallback activated `0/9000` times and all A3/A4 responses were byte-identical. The `41.24%` value is an event-repair sample rate, not motif fallback.
- First-commit endpoint replay: adaptive MIDI-VAD reaches custom premature-sensitive first-commit score (PS-F1) `0.619` (95% CI `[0.523, 0.711]`) at the two-second deadline, compared with `0.500` (`[0.400, 0.590]`) for fixed 800 ms, with 34 versus 50 premature commits. PS-F1 uses `TP=success`, `FP=premature`, and `FN=premature+late+missed`; this is not conventional event-detection F1 and uses final-Note-Off proxy boundaries, not human annotations.
- Commit-anchored scheduler replay: with a `150 ms` pre-commit overlap and an `80 ms` readiness target, the modeled first-generated-event readiness lower bound decreases from `101.543 ms` to `86.545 ms`; the Call-cluster mean reduction is `14.999 ms` (`[10.152, 20.494]`). Readiness-target misses decrease from `20.744%` to `5.244%`. The replay does not measure MIDI transmission, host receipt, or audio onset.
- No human-participant data are analyzed or reported. Legacy aggregate listening artifacts are excluded from the manuscript evidence package; see their `LEGACY_NOTICE.md`.

These layers support an engineering adaptation, structural manipulation checks, narrow symbolic feature associations, and deterministic scheduler behavior. They do not establish universal endpoint accuracy, faster intrinsic AMT decoding, live co-performance usefulness, or perceptual superiority.

## Repository Layout

```text
code/
  live_call_response.py              realtime MIDI engine
  midi_vad_endpoint.py               adaptive phrase endpoint detector
  interface_backend.py               local FastAPI/WebSocket studio backend
  static/                            browser UI for local performance
  offline_ab_test.py                 objective sample and ablation generation
  evaluate_melody_metrics.py         objective symbolic-music metrics
  build_harmonized_call100_evaluation.py
                                      shared-batch candidate evaluation
  analyze_ablation_integrity.py      raw-label and fallback audit
  run_endpoint_benchmark.py          direct endpoint replay and baselines
  export_public_results.py           path-redacted, semantic public CSV export
  run_call100_objective_search.py    Call100 objective-search driver
  run_call100_ablation_latency.py    A0-A6 ablation and latency aggregation
  recompute_scheduler_replay.py      commit-anchored scheduler recomputation
  analyze_paper_clustered_statistics.py
                                      Call/source-cluster paper statistics
  analyze_call_response_association.py
                                      matched--mismatched symbolic association audit
  test_evaluation_integrity.py       scoring/fallback regression tests
  test_endpoint_benchmark.py         endpoint benchmark regression tests

paper/
  Paper PDF, LaTeX source, bibliography, and system overview figure.

results/
  call100_dataset/                   Call100 manifest and validation scripts
  call100_harmonized_evaluation/     corrected 27,000-row trial metrics
  evaluation_integrity_audit/        raw-label and A3/A4 activation evidence
  endpoint_benchmark_call100/        100-call endpoint results
  call100_ablation_latency/          A0-A6 and preload latency summaries
  call_response_association/         path-redacted feature-association outputs
  blind_listening_final/             ethics notice only; aggregate outputs removed
  paper_clustered_statistics/        Call/source-cluster analysis outputs

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
- `results/paper_clustered_statistics/candidate_clustered_comparisons.csv`
- `results/paper_clustered_statistics/ablation_clustered_stepwise.csv`
- `results/paper_clustered_statistics/latency_clustered_comparisons.csv`
- `results/call100_harmonized_evaluation/score_spec.json`
- `results/evaluation_integrity_audit/integrity_report.md`
- `results/evaluation_integrity_audit/runtime_revalidation_20260816.json`
- `results/endpoint_benchmark_call100/endpoint_condition_summary.csv`
- `results/call100_ablation_latency/ablation_trial_metrics.csv`
- `results/call100_ablation_latency/latency_log_all_trials.csv`
- `results/call100_ablation_latency/latency_summary_by_condition.csv`
- `results/call100_ablation_latency/preload_on_off_comparison.csv`
- `results/call100_ablation_latency/ablation_validation_summary.json`
- `results/call_response_association/candidate_association_summary.csv`
- `results/call_response_association/candidate_association_comparisons.csv`
- `results/call_response_association/provenance.json`
- `results/call_response_association/source_excluded_mismatch_sensitivity.csv`
- `results/paper_clustered_statistics/report.md`
- `results/call100_ablation_latency/scheduler_replay_provenance.json`
- `results/blind_listening_final/LEGACY_NOTICE.md`
- `results/call100_dataset/call100_manifest_public.csv`

To rerun aggregation from available summaries:

```powershell
python -m compileall code
python code/run_call100_ablation_latency.py --help
python code/build_harmonized_call100_evaluation.py --help
python code/run_endpoint_benchmark.py --help
python code/analyze_call_response_association.py --help
python code/test_evaluation_integrity.py
python code/test_endpoint_benchmark.py
python code/test_call_response_association.py
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

Use `CITATION.cff` for GitHub citation metadata. Zenodo release metadata is defined in `.zenodo.json`. The exact immutable v1.2.0 DOI is shown on the release page; the citation below uses the stable all-versions concept DOI.

```bibtex
@software{wang_hu_2026_realtime_midi_call_response,
  author = {Wang, Zitong and Hu, Sitong},
  title = {Adapting a Frozen Anticipatory Music Transformer for Turn-Based MIDI Call-and-Response},
  year = {2026},
  version = {1.2.0},
  doi = {10.5281/zenodo.20838083},
  url = {https://doi.org/10.5281/zenodo.20838083}
}
```

## License

Project code and documentation in this repository are released under the MIT License. Third-party models, datasets, papers, plugins, DAWs, and audio assets remain under their respective licenses.
