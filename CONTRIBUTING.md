# Contributing

Thanks for improving this research code release.

## Scope

Useful contributions include:

- bug fixes for the realtime MIDI engine
- clearer setup instructions
- reproducibility checks for the Call100 summaries
- small test fixtures that do not require proprietary audio assets
- documentation for additional MIDI routing environments

Please do not commit model weights, Hugging Face caches, piano sample libraries, VST plugins, DAW installers, generated response batches, private answer keys, or credentials.

## Development

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Before opening a pull request, run the scripts you touched with `--help` or a small local smoke test where possible.

## Research Claims

Keep new claims tied to checked outputs. If a result comes from a new run, include the aggregation script, validation summary, and enough metadata for another reader to understand the run conditions.
