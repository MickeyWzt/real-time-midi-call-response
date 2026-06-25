# v1.0.0

Initial public research-code release for **Real-Time MIDI Call-and-Response Generation Using Autoregressive Transformers**.

Zenodo version DOI: https://doi.org/10.5281/zenodo.20838084

## Included

- realtime MIDI call-and-response engine
- local browser studio backend and frontend
- offline A/B and objective evaluation scripts
- Call100 dataset manifest and validation scripts
- Call100 objective-search summary tables
- A0-A6 ablation and latency summary tables
- paper PDF, LaTeX source, bibliography, and system overview figure
- GitHub Pages source under `docs/`
- `CITATION.cff` and `.zenodo.json` for citation and Zenodo archiving

## Excluded

- model weights and Hugging Face caches
- third-party audio tools, DAWs, VST plugins, and piano sample libraries
- generated MIDI response batches and raw per-run directories
- private answer keys, credentials, and deployment-local config

## Headline Results

- Controlled AMT improves over raw AMT by `+0.063907` mean objective score on the 27,000-trial Call100 comparison.
- Full A6 control increases mean objective score from `0.560968` to `0.732610` in the 63,000-row ablation.
- Speculative preload reduces mean endpoint-to-first-MIDI latency from `161.303 ms` to `85.534 ms` and underrun rate from `20.744%` to `3.556%`.
