# Windows collaborator setup

This document describes the current cross-platform collaboration path for the
MFP live MIDI system.

## What is shared

The repository contains the real-time MIDI engine, the paper-compatible AMT
path, an AMT-first streaming live mode, rhythm learning and playback, Loop Bank,
the browser control surface, tests, and the optional macOS Logic MIDI FX source.

The repository does not contain model caches, model weights, DAW content,
third-party plug-ins, Logic sound libraries, private recordings, or runtime
logs. Those files are either large, machine-specific, or license-sensitive.

## Windows requirements

- Python 3.12
- A virtual MIDI driver such as loopMIDI
- A Windows DAW or standalone VST3 host for the melody and drum sounds
- The AMT checkpoint downloaded separately, unless the project is run with a
  configured online model download

Create two virtual MIDI ports named `Python_IN` and `Python_OUT`. The engine
reads the performer input from `Python_IN`, sends melody and AI responses to
the selected melody output, and sends the rhythm guide on MIDI channel 10 to
`Python_OUT`.

## Install and run

```powershell
git clone <repository-url>
cd real-time-midi-call-response
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python code/interface_backend.py --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000` in a browser. Select the physical MIDI input in
the control surface, then route the two MIDI outputs to the Windows DAW.

## DAW routing

Use one melody instrument track on MIDI channel 1 and one drum instrument
track on MIDI channel 10. The performer input is monitored through the melody
route, while AI responses and Loop Bank playback use the same melody output.
The rhythm engine sends kick, snare, and hi-hat notes to the drum route.

The browser is the current portable control surface. The Logic MIDI FX target
is macOS-only; a Windows VST3 control-surface target can be added later
without changing the MIDI engine or response algorithm.

## Collaboration workflow

Keep the author's `main` branch protected. Work on a feature branch, push it
to the shared repository, and open a pull request. Do not commit `.venv`,
`hf_cache`, `model_weights`, `logs`, or `.env` files.

```powershell
git checkout -b mfp-live-studio
git add .
git commit -m "feat: add live MIDI performance studio"
git push -u origin mfp-live-studio
```

The author can review and merge the branch, after which both collaborators can
sync with:

```powershell
git checkout main
git pull --ff-only origin main
```
