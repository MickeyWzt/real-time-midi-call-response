# Third-party Components

This project builds on several third-party tools and models. They are not
redistributed in this repository unless explicitly noted.

## Anticipatory Music Transformer

The AMT backend uses the Anticipatory Music Transformer code and pretrained
models by John Thickstun, David Hall, Chris Donahue, and Percy Liang.

- Paper: https://arxiv.org/abs/2306.08620
- Code: https://github.com/jthickstun/anticipation
- Models: https://huggingface.co/stanford-crfm
- License of the upstream code: Apache License 2.0

Install the upstream package separately or vendor it according to the upstream
license terms.

## Aria

The optional Aria backend expects locally downloaded model files. Model weights
are not included in this repository.

## MIDI and audio utilities

The realtime demo is designed for Windows with loopMIDI and an external or
virtual MIDI input. Optional piano plugins or sample libraries are not bundled.

## Apple AudioUnit SDK

The optional Logic MIDI FX source references Apple's AudioUnit SDK as a pinned
Git submodule at commit `bd98b31feff57a15989fcfab4cd86dc63382b1ac`:
https://github.com/apple/AudioUnitSDK.git. The upstream SDK is licensed under
Apache License 2.0. Its contents are fetched separately with
`git clone --recurse-submodules` and are not embedded in the repository archive.
Any future binary distribution must include the applicable upstream notices.

## Excluded assets

The repository intentionally excludes:

- Hugging Face caches and model weights
- VST plugins and standalone audio software
- Apple AudioUnit SDK contents (pinned submodule reference only)
- piano sample libraries
- thesis PDFs, private notes, and experiment logs
