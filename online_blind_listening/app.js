const config = {
  studyId: "mfp-call-response-12pair-stratified-nofallback-v3",
  sheetEndpoint: "",
  noCorsSubmit: true,
  ...(window.BLIND_TEST_CONFIG || {})
};

const state = {
  pairs: [],
  index: 0,
  startedAt: "",
  audioCtx: null,
  sampledPianoPromise: null,
  activeTimers: [],
  activeNodes: [],
  notes: new Map(),
  plays: new Map(),
  responses: [],
  pairStartedAt: "",
  pairStartedMs: 0,
  choiceLocked: false,
  sessionId: getOrCreateSessionId(),
  submissionId: crypto.randomUUID ? crypto.randomUUID() : makeId("submission")
};

const $ = (id) => document.getElementById(id);

function makeId(prefix) {
  return `${prefix}_${Math.random().toString(36).slice(2)}_${Date.now().toString(36)}`;
}

function getOrCreateSessionId() {
  const key = "mfp_blind_session_id";
  const existing = localStorage.getItem(key);
  if (existing) return existing;
  const id = crypto.randomUUID ? crypto.randomUUID() : makeId("session");
  localStorage.setItem(key, id);
  return id;
}

function recruitmentInfo() {
  const params = new URLSearchParams(window.location.search);
  return {
    recruitment_source: params.get("source") || params.get("utm_source") || "",
    utm_source: params.get("utm_source") || "",
    utm_medium: params.get("utm_medium") || "",
    utm_campaign: params.get("utm_campaign") || "",
    landing_referrer: document.referrer || ""
  };
}

function setPanel(panelId) {
  ["introPanel", "studyPanel", "finishPanel"].forEach((id) => $(id).classList.toggle("hidden", id !== panelId));
}

function setupStatus() {
  $("sheetStatus").textContent = config.sheetEndpoint ? "Sheet: configured" : "Sheet: local backup mode";
}

async function loadPairs() {
  const response = await fetch("./public/study-pairs.json", { cache: "no-store" });
  if (!response.ok) throw new Error("Cannot load study-pairs.json");
  state.pairs = await response.json();
}

function readStr(view, pos, len) {
  let out = "";
  for (let i = 0; i < len; i += 1) out += String.fromCharCode(view.getUint8(pos + i));
  return out;
}

function readVar(view, posRef) {
  let value = 0;
  while (true) {
    const byte = view.getUint8(posRef.pos);
    posRef.pos += 1;
    value = (value << 7) | (byte & 0x7f);
    if ((byte & 0x80) === 0) break;
  }
  return value;
}

function parseMidi(buffer) {
  const view = new DataView(buffer);
  let pos = 0;
  if (readStr(view, pos, 4) !== "MThd") throw new Error("Invalid MIDI header");
  pos += 4;
  const headerLen = view.getUint32(pos); pos += 4;
  const format = view.getUint16(pos); pos += 2;
  const trackCount = view.getUint16(pos); pos += 2;
  const division = view.getUint16(pos); pos += 2;
  pos = 8 + headerLen;
  if (division & 0x8000) throw new Error("SMPTE MIDI timing is not supported");
  const ticksPerBeat = division || 480;
  let tempo = 500000;
  const notes = [];

  for (let track = 0; track < trackCount; track += 1) {
    if (readStr(view, pos, 4) !== "MTrk") break;
    pos += 4;
    const trackLen = view.getUint32(pos); pos += 4;
    const end = pos + trackLen;
    const active = new Map();
    let ticks = 0;
    let runningStatus = null;

    while (pos < end) {
      const posRef = { pos };
      const delta = readVar(view, posRef);
      pos = posRef.pos;
      ticks += delta;
      let status = view.getUint8(pos);
      if (status < 0x80) {
        if (runningStatus === null) break;
        status = runningStatus;
      } else {
        pos += 1;
        if (status < 0xf0) runningStatus = status;
      }

      if (status === 0xff) {
        const type = view.getUint8(pos); pos += 1;
        const metaRef = { pos };
        const len = readVar(view, metaRef);
        pos = metaRef.pos;
        if (type === 0x51 && len === 3) {
          tempo = (view.getUint8(pos) << 16) | (view.getUint8(pos + 1) << 8) | view.getUint8(pos + 2);
        }
        pos += len;
        continue;
      }
      if (status === 0xf0 || status === 0xf7) {
        const sysRef = { pos };
        const len = readVar(view, sysRef);
        pos = sysRef.pos + len;
        continue;
      }

      const command = status & 0xf0;
      const channel = status & 0x0f;
      const data1 = view.getUint8(pos); pos += 1;
      let data2 = 0;
      if (command !== 0xc0 && command !== 0xd0) {
        data2 = view.getUint8(pos); pos += 1;
      }

      const key = `${channel}:${data1}`;
      const seconds = (ticks / ticksPerBeat) * (tempo / 1000000);
      if (command === 0x90 && data2 > 0) {
        active.set(key, { pitch: data1, velocity: data2, start: seconds });
      } else if ((command === 0x80 || (command === 0x90 && data2 === 0)) && active.has(key)) {
        const start = active.get(key);
        active.delete(key);
        notes.push({
          pitch: start.pitch,
          velocity: start.velocity,
          start: start.start,
          duration: Math.max(0.08, seconds - start.start)
        });
      }
    }
    pos = end;
  }

  if (format > 2) throw new Error("Unsupported MIDI format");
  return notes.sort((a, b) => a.start - b.start || a.pitch - b.pitch);
}

async function loadNotes(file) {
  if (state.notes.has(file)) return state.notes.get(file);
  const response = await fetch(`./public/${file}`);
  if (!response.ok) throw new Error(`Cannot load ${file}`);
  const notes = parseMidi(await response.arrayBuffer());
  state.notes.set(file, notes);
  return notes;
}

function midiToFrequency(pitch) {
  return 440 * Math.pow(2, (pitch - 69) / 12);
}

function midiToName(pitch) {
  const names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
  return `${names[pitch % 12]}${Math.floor(pitch / 12) - 1}`;
}

async function ensureAudio() {
  if (!state.audioCtx) {
    state.audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  }
  if (state.audioCtx.state === "suspended") await state.audioCtx.resume();
}

async function ensureSampledPiano() {
  if (state.sampledPianoPromise) return state.sampledPianoPromise;
  if (!window.Soundfont) throw new Error("Soundfont library unavailable");
  state.sampledPianoPromise = window.Soundfont.instrument(state.audioCtx, "acoustic_grand_piano", {
    soundfont: "FluidR3_GM"
  });
  return state.sampledPianoPromise;
}

function stopPlayback() {
  for (const timer of state.activeTimers) clearTimeout(timer);
  state.activeTimers = [];
  for (const node of state.activeNodes) {
    try {
      if (node.stop) node.stop();
      if (node.disconnect) node.disconnect();
    } catch {
      // Already stopped.
    }
  }
  state.activeNodes = [];
}

async function playClip(kind) {
  const pair = state.pairs[state.index];
  const file = kind === "call" ? pair.call_file : (kind === "A" ? pair.a_file : pair.b_file);
  const label = kind === "call" ? "listenCall" : (kind === "A" ? "listenA" : "listenB");
  stopPlayback();
  await ensureAudio();
  const notes = await loadNotes(file);
  const countKey = `${pair.pair_id}:${kind}`;
  state.plays.set(countKey, (state.plays.get(countKey) || 0) + 1);
  $(label).textContent = `Played ${state.plays.get(countKey)} time${state.plays.get(countKey) > 1 ? "s" : ""}`;
  $("modeLabel").textContent = "Playing...";

  try {
    const instrument = await ensureSampledPiano();
    const now = state.audioCtx.currentTime + 0.08;
    for (const note of notes) {
      const player = instrument.play(midiToName(note.pitch), now + note.start, {
        duration: Math.max(0.14, note.duration + 0.22),
        gain: Math.max(0.12, Math.min(0.78, note.velocity / 127)) * 0.45
      });
      if (player && player.stop) state.activeNodes.push(player);
    }
  } catch {
    playFallback(notes);
  }

  const totalMs = (Math.max(0, ...notes.map((n) => n.start + n.duration)) + 0.5) * 1000;
  state.activeTimers.push(setTimeout(() => {
    $("modeLabel").textContent = "Listen, then choose.";
  }, totalMs));
}

function playFallback(notes) {
  const now = state.audioCtx.currentTime + 0.08;
  for (const note of notes) {
    const osc = state.audioCtx.createOscillator();
    const gain = state.audioCtx.createGain();
    osc.type = "triangle";
    osc.frequency.value = midiToFrequency(note.pitch);
    gain.gain.setValueAtTime(0.0001, now + note.start);
    gain.gain.exponentialRampToValueAtTime(0.08 * Math.max(0.35, note.velocity / 127), now + note.start + 0.02);
    gain.gain.exponentialRampToValueAtTime(0.0001, now + note.start + Math.max(0.1, note.duration));
    osc.connect(gain).connect(state.audioCtx.destination);
    osc.start(now + note.start);
    osc.stop(now + note.start + Math.max(0.12, note.duration + 0.05));
    state.activeNodes.push(osc);
  }
}

function startPairTimer() {
  state.pairStartedAt = new Date().toISOString();
  state.pairStartedMs = performance.now();
}

function capturePairTiming() {
  const choiceAt = new Date().toISOString();
  const elapsedMs = state.pairStartedMs ? performance.now() - state.pairStartedMs : 0;
  return {
    pair_started_at: state.pairStartedAt,
    pair_choice_at: choiceAt,
    pair_duration_seconds: Math.max(0, elapsedMs / 1000).toFixed(3)
  };
}

async function renderPair() {
  stopPlayback();
  state.choiceLocked = false;
  const pair = state.pairs[state.index];
  startPairTimer();
  $("pairLabel").textContent = `Pair ${state.index + 1} of ${state.pairs.length}`;
  $("modeLabel").textContent = "Listen, then choose.";
  $("progressFill").style.width = `${((state.index + 1) / state.pairs.length) * 100}%`;
  $("listenCall").textContent = "Listen to the call first";
  $("listenA").textContent = "Not played yet";
  $("listenB").textContent = "Not played yet";
  document.querySelectorAll(".choice").forEach((button) => {
    button.classList.remove("selected");
    button.disabled = false;
  });
}

function choose(choice) {
  if (state.choiceLocked) return;
  state.choiceLocked = true;
  const timing = capturePairTiming();
  stopPlayback();
  document.querySelectorAll(".choice").forEach((button) => {
    button.classList.toggle("selected", button.dataset.choice === choice);
    button.disabled = true;
  });
  window.setTimeout(() => saveCurrentAndAdvance(choice, timing), 160);
}

function saveCurrentAndAdvance(choice, timing) {
  const pair = state.pairs[state.index];
  const existingIndex = state.responses.findIndex((item) => item.pair_id === pair.pair_id);
  const response = {
    pair_id: pair.pair_id,
    pair_index: state.index + 1,
    pair_started_at: timing.pair_started_at,
    pair_choice_at: timing.pair_choice_at,
    pair_duration_seconds: timing.pair_duration_seconds,
    choice,
    confidence: "",
    played_call_count: state.plays.get(`${pair.pair_id}:call`) || 0,
    played_a_count: state.plays.get(`${pair.pair_id}:A`) || 0,
    played_b_count: state.plays.get(`${pair.pair_id}:B`) || 0
  };
  if (existingIndex >= 0) state.responses[existingIndex] = response;
  else state.responses.push(response);

  if (state.index + 1 >= state.pairs.length) {
    stopPlayback();
    setPanel("finishPanel");
  } else {
    state.index += 1;
    renderPair();
  }
}

function buildSubmission() {
  const completedAt = new Date().toISOString();
  return {
    study_id: config.studyId,
    submission_id: state.submissionId,
    session_id: state.sessionId,
    started_at: state.startedAt,
    completed_at: completedAt,
    duration_seconds: Math.round((new Date(completedAt).getTime() - new Date(state.startedAt).getTime()) / 1000),
    ...recruitmentInfo(),
    music_background: $("musicBackground").value,
    listening_setup: $("headphones").value,
    comment: $("comment").value.trim(),
    responses: state.responses
  };
}

async function submitResults() {
  const payload = buildSubmission();
  localStorage.setItem("mfp_blind_last_submission", JSON.stringify(payload));
  if (!config.sheetEndpoint) {
    $("submitMessage").textContent = "Saved in this browser. Configure Google Apps Script endpoint to submit online.";
    $("submitMessage").classList.add("error");
    return;
  }

  $("submitBtn").disabled = true;
  $("submitMessage").classList.remove("error");
  $("submitMessage").textContent = "Submitting...";
  try {
    const body = JSON.stringify(payload);
    if (config.noCorsSubmit) {
      await fetch(config.sheetEndpoint, {
        method: "POST",
        mode: "no-cors",
        headers: { "Content-Type": "text/plain;charset=utf-8" },
        body
      });
      $("submitMessage").textContent = "Submitted. A local backup is also saved in this browser.";
    } else {
      const response = await fetch(config.sheetEndpoint, {
        method: "POST",
        headers: { "Content-Type": "text/plain;charset=utf-8" },
        body
      });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      $("submitMessage").textContent = "Submitted. Thank you.";
    }
  } catch (error) {
    $("submitBtn").disabled = false;
    $("submitMessage").textContent = `Submit failed: ${error.message}. Please download the backup JSON.`;
    $("submitMessage").classList.add("error");
  }
}

function downloadBackup() {
  const payload = buildSubmission();
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `${config.studyId}_${state.submissionId}.json`;
  link.click();
  URL.revokeObjectURL(url);
}

async function startStudy() {
  state.startedAt = new Date().toISOString();
  setPanel("studyPanel");
  await renderPair();
}

async function init() {
  setupStatus();
  await loadPairs();
  $("consentInput").addEventListener("change", (event) => {
    $("startBtn").disabled = !event.target.checked;
  });
  $("startBtn").addEventListener("click", startStudy);
  $("playCall").addEventListener("click", () => playClip("call"));
  $("playA").addEventListener("click", () => playClip("A"));
  $("playB").addEventListener("click", () => playClip("B"));
  $("stopBtn").addEventListener("click", stopPlayback);
  document.querySelectorAll(".choice").forEach((button) => {
    button.addEventListener("click", () => choose(button.dataset.choice));
  });
  $("submitBtn").addEventListener("click", submitResults);
  $("downloadBtn").addEventListener("click", downloadBackup);
}

init().catch((error) => {
  $("introPanel").innerHTML = `<h2>Unable to load the test</h2><p>${error.message}</p>`;
});
