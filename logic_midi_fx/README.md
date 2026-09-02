# MFP Logic MIDI FX

This folder contains the Logic-native Audio Unit v2 MIDI FX integration.

`MFP_Logic_MIDI_FX.js` is retained only as an early Scripter prototype. Normal use
is the installed `MFPNativeMIDIProcessor.component` shown in Logic as
`MFP: AI Response`.

The Audio Unit provides a custom Cocoa window containing the MFP live surface.
The pretrained model and rhythm analysis remain in the local Python companion
service. MIDI processing in the Audio Unit is clean pass-through so it cannot add
a duplicate transposed response or leave generated notes hanging.

## Native AU window

Start `start_mac_demo.command`, then insert **MIDI FX > Audio Units > MFP > AI
Response** on the Logic instrument track. The plug-in window loads
`http://127.0.0.1:8000/?embedded=logic`. If the service is unavailable, it shows
a retry panel instead of a blank plug-in.

The rhythm guide records tap timing and velocity, not drum notes. It infers BPM,
quantizes a 16-step pulse while retaining small timing variations, generates an
automatic kick/snare/hat MIDI guide, and synchronizes the AI response onset to
the inferred beat. A new learned rhythm replaces the current rhythm at the next
bar boundary.

The track instrument remains the sound source. To change the AI response tone,
change the Logic software instrument or patch on that track. To use a dedicated
Logic drum kit, put the drum kit on a separate software-instrument track and
route the selected drum channel there; the native AU does not replace Logic's
instrument library.

## Logic routing

```text
MiniLab 3 -> MFP companion service
              |-> Logic Pro Virtual In / channel 1: AI / Loop Bank -> current Logic instrument
              |-> Python_OUT / channel 10: rhythm guide -> selected Logic drum kit

Logic MIDI FX slot -> MFP: AI Response -> custom performance window + clean MIDI pass-through
```

The browser URL remains available for no-hardware testing. It exposes the same
state and controls as the Logic plug-in window.
