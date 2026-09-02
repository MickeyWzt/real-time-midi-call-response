/*
 * MFP Logic MIDI FX — Logic Pro Scripter bridge
 *
 * This is the first Logic-native integration layer. Put it in a Logic Pro
 * Scripter MIDI FX plug-in on the software-instrument track that should play
 * the response. The track's instrument remains the sound source, so changing
 * the Logic instrument immediately changes the sound of the MFP output.
 *
 * The script is intentionally self-contained. It provides a dependable
 * in-Logic fallback response and a rhythm recorder while the native Audio
 * Unit bridge is being built. It does not pretend to be the final native AU.
 */

var PluginParameters = [
  {name: "MFP Enabled", type: "checkbox", defaultValue: 1},
  {name: "Response Delay", type: "lin", minValue: 0.1, maxValue: 2.0, numberOfSteps: 19, defaultValue: 0.5, unit: "s"},
  {name: "Response Density", type: "lin", minValue: 0.25, maxValue: 1.0, numberOfSteps: 15, defaultValue: 0.7},
  {name: "Pad Rhythm", type: "checkbox", defaultValue: 0},
  {name: "Rhythm Replace", type: "checkbox", defaultValue: 0},
  {name: "Response Loop", type: "checkbox", defaultValue: 0}
];

var enabled = true;
var responseDelay = 0.5;
var responseDensity = 0.7;
var padRhythm = false;
var rhythmReplace = false;
var responseLoop = false;

var callNotes = [];
var lastCallBeat = -1;
var responseEvents = [];
var rhythmEvents = [];
var rhythmStartBeat = -1;
var rhythmLastBeat = -1;
var responseLoopStartBeat = -1;
var responseLoopLength = 0;

function ParameterChanged(param, value) {
  if (param === 0) enabled = value >= 0.5;
  if (param === 1) responseDelay = value;
  if (param === 2) responseDensity = value;
  if (param === 3) padRhythm = value >= 0.5;
  if (param === 4) rhythmReplace = value >= 0.5;
  if (param === 5) {
    responseLoop = value >= 0.5;
    if (!responseLoop) responseLoopStartBeat = -1;
  }
}
function Reset() {
  callNotes = [];
  lastCallBeat = -1;
  responseEvents = [];
  rhythmEvents = [];
  rhythmStartBeat = -1;
  rhythmLastBeat = -1;
  responseLoopStartBeat = -1;
  responseLoopLength = 0;
}

function isPad(note) {
  // MiniLab 3 pads are commonly mapped to the upper MIDI range. Keeping this
  // range narrow avoids stealing normal keyboard notes.
  return note.pitch >= 36 && note.pitch <= 51;
}

function rememberCall(event, beat) {
  callNotes.push({pitch: event.pitch, velocity: event.velocity, beat: beat});
  if (callNotes.length > 32) callNotes.shift();
  lastCallBeat = beat;
}

function makeResponse(nowBeat) {
  if (!enabled || callNotes.length === 0) return;

  var source = callNotes.slice(-Math.max(2, Math.floor(callNotes.length * responseDensity)));
  var start = nowBeat + responseDelay * 2.0;
  var previous = -1;
  for (var i = 0; i < source.length; i++) {
    var item = source[i];
    var pitch = item.pitch + (i % 2 === 0 ? 7 : 12);
    pitch = Math.max(36, Math.min(96, pitch));
    if (pitch === previous) pitch = Math.min(96, pitch + 2);
    var event = new Note({pitch: pitch, velocity: Math.max(35, Math.min(127, item.velocity - 8)), channel: 0});
    var beat = start + i * 0.25;
    event.sendAtBeat(beat);
    responseEvents.push({pitch: pitch, velocity: event.velocity, beat: i * 0.25});
    previous = pitch;
  }
  responseLoopLength = Math.max(1.0, source.length * 0.25);
  if (responseLoop && responseLoopStartBeat < 0) responseLoopStartBeat = start;
  callNotes = [];
}

function capturePad(event, beat) {
  if (!padRhythm || !isPad(event)) return false;
  if (rhythmStartBeat < 0 || rhythmReplace) {
    rhythmEvents = [];
    rhythmStartBeat = beat;
    rhythmLastBeat = beat;
    rhythmReplace = false;
  }
  rhythmEvents.push({pitch: event.pitch, velocity: event.velocity, beat: beat - rhythmStartBeat});
  rhythmLastBeat = beat;
  return true;
}

function HandleMIDI(event) {
  if (event instanceof Note) {
    var info = GetTimingInfo();
    var beat = info.blockStartBeat;
    if (capturePad(event, beat)) {
      event.send();
      return;
    }
    if (enabled) rememberCall(event, beat);
    event.send();
    return;
  }
  event.send();
}

function ProcessMIDI() {
  var info = GetTimingInfo();
  if (!info.playing) return;

  var beat = info.blockStartBeat;
  if (lastCallBeat >= 0 && beat - lastCallBeat >= responseDelay * 2.0 && callNotes.length > 0) {
    makeResponse(beat);
  }

  if (responseLoop && responseLoopStartBeat >= 0 && responseLoopEventsReady()) {
    var localBeat = beat - responseLoopStartBeat;
    if (localBeat >= responseLoopLength) {
      responseLoopStartBeat += responseLoopLength;
      for (var i = 0; i < responseEvents.length; i++) {
        var item = responseEvents[i];
        var note = new Note({pitch: item.pitch, velocity: item.velocity, channel: 0});
        note.sendAtBeat(responseLoopStartBeat + item.beat);
      }
    }
  }

  if (rhythmEvents.length > 0 && rhythmStartBeat >= 0 && beat - rhythmStartBeat > 8.0) {
    // Keep the recorded pad groove alive in Logic while Pad Rhythm is on.
    var rhythmLength = Math.max(1.0, Math.ceil((rhythmLastBeat - rhythmStartBeat) * 4.0) / 4.0);
    var cycle = Math.floor((beat - rhythmStartBeat) / rhythmLength);
    var cycleStart = rhythmStartBeat + cycle * rhythmLength;
    for (var r = 0; r < rhythmEvents.length; r++) {
      var hit = rhythmEvents[r];
      var drum = new Note({pitch: hit.pitch, velocity: hit.velocity, channel: 9});
      drum.sendAtBeat(cycleStart + hit.beat);
    }
  }
}

function responseLoopEventsReady() {
  return responseEvents.length > 0 && responseLoopLength > 0;
}
