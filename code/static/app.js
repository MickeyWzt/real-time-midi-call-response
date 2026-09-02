const $ = (id) => document.getElementById(id);
document.body.classList.toggle(
    "embedded-logic",
    new URLSearchParams(window.location.search).get("embedded") === "logic"
);

const state = {
    ws: null,
    connected: false,
    running: false,
    sessionStartedAt: null,
    sessionInput: null,
    devices: { inputs: [], outputs: [], selected_input: null, selected_output: null, virtual_mode: true },
    pianoHost: null,
    selectedInput: null,
    octave: 3,
    velocity: 96,
    keyboardMode: "standard",
    keyboardMinimized: false,
    activeKeys: new Set(),
    activeVisualNotes: new Map(),
    keyboardParticles: [],
    audio: {
        ctx: null,
        master: null,
        analyser: null,
        limiter: null,
        reverb: null,
        wet: null,
        active: new Map(),
        sampleManifest: null,
        sampleBuffers: new Map(),
        sampleLoading: new Map(),
        samplerReady: false,
        ready: false
    },
    round: {
        id: null,
        state: "idle",
        firstEvent: "--",
        underruns: 0
    },
    rhythm: {
        state: "idle",
        tap_count: 0,
        bpm: 100,
        confidence: 0,
        bars: 1,
        steps_per_bar: 16,
        pattern: [],
        loop_seconds: 0,
        learning: false,
        playing: false,
        replacing: false,
        stopping: false,
        saved_slots: [],
        current_slot: null,
        queued_slot: null
    },
    loopBank: {
        mode: "response",
        latest_ready: false,
        slots: ["A", "B", "C", "D"].map((name) => ({ name })),
        lastPlaybackAt: 0
    }
};

const selectMidi = $("midi-input");
const btnRefresh = $("refresh-devices");
const btnStart = $("start-btn");
const btnStop = $("stop-btn");
const btnTest = $("test-btn");
const backendSelect = $("backend-select");
const statusConnection = $("connection-status");
const statusEngine = $("engine-status");
const statusPianoHost = $("piano-host-status");
const statusWebMidi = $("web-midi-status");
const statusTestMode = $("test-mode-status");
const statusRhythm = $("rhythm-status");
const statusLoop = $("loop-status");
const logPanel = $("log-panel");
const canvas = $("visualizer-canvas");
const ctx = canvas.getContext("2d");
const keyboardCanvas = $("keyboard-particle-canvas");
const keyboardCtx = keyboardCanvas ? keyboardCanvas.getContext("2d") : null;

let keyMap = {};
let whiteOffsets = [];
let blackOffsets = [];
let keyboardLabels = {};

const KEYBOARD_SPANS = {
    compact: 36,
    standard: 48,
    expanded: 60,
};
const WHITE_SEMITONES = new Set([0, 2, 4, 5, 7, 9, 11]);
const TYPING_KEYS = ["a", "w", "s", "e", "d", "f", "t", "g", "y", "h", "u", "j", "k", "o", "l", "p", ";", "'"];
const NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
const MODEL_CONFIG = {
    amt_small: {
        backend: "amt",
        model_id: "stanford-crfm/music-small-800k",
        label: "AMT Small / stanford-crfm/music-small-800k"
    }
};

const ROUND_LABELS = {
    idle: "待命",
    generating: "AI 思考中",
    analysis: "分析乐句",
    plan: "组织回应",
    first_event: "回应已生成",
    generated: "准备播放",
    playback: "AI 回应中",
    done: "回应完成",
    ok: "完成",
    underrun: "播放等待"
};

const SESSION_LABELS = {
    starting: "正在加载",
    listening: "正在聆听",
    generating: "正在思考",
    playback_pending: "准备回应",
    buffering: "准备播放",
    playback: "正在回应",
    done: "回应完成"
};

function send(payload) {
    if (state.ws && state.ws.readyState === WebSocket.OPEN) {
        state.ws.send(JSON.stringify(payload));
    }
}

function setStatus(el, text, mode) {
    el.textContent = text;
    el.className = `status-pill ${mode}`;
}

function connect() {
    const protocol = location.protocol === "https:" ? "wss" : "ws";
    state.ws = new WebSocket(`${protocol}://${location.host}/ws`);

    state.ws.onopen = () => {
        state.connected = true;
        setStatus(statusConnection, "控制服务已连接", "connected");
        send({ type: "refresh_devices" });
    };

    state.ws.onclose = () => {
        state.connected = false;
        setStatus(statusConnection, "正在重新连接", "warning");
        setTimeout(connect, 1200);
    };

    state.ws.onmessage = (event) => {
        handleMessage(JSON.parse(event.data));
    };
}

function handleMessage(payload) {
    if (payload.type === "devices") {
        state.devices = payload;
        renderDevices();
    } else if (payload.type === "session_status") {
        renderSession(payload);
    } else if (payload.type === "piano_host_status") {
        renderPianoHost(payload);
    } else if (payload.type === "rhythm_status") {
        renderRhythm(payload);
    } else if (payload.type === "drum_status") {
        renderRhythm({
            state: payload.recording ? "learning" : payload.playing ? "running" : payload.state,
            tap_count: payload.event_count || 0,
            learning: Boolean(payload.recording),
            playing: Boolean(payload.playing),
            replacing: Boolean(payload.replacing),
            stopping: Boolean(payload.stopping)
        });
    } else if (payload.type === "loop_bank_status") {
        renderLoopBank(payload);
    } else if (payload.type === "visual_note") {
        pushVisualNote(payload);
    } else if (payload.type === "playback_note") {
        if (payload.bus === "loop" && payload.event === "note_on") {
            state.loopBank.lastPlaybackAt = Date.now();
            setStatus(statusLoop, "回应记忆正在发声", "connected");
        }
        handlePlaybackNote(payload);
    } else if (payload.type === "panic_status") {
        stopAllLocalAudio();
        log(payload.message || "所有声音已释放", payload.ok ? "info" : "error");
    } else if (payload.type === "round_state") {
        renderRoundState(payload);
    } else if (payload.type === "metrics") {
        renderMetrics(payload);
    } else if (payload.type === "log") {
        maybeVisualizeFromLog(payload.message);
        log(payload.message, payload.level || "live");
    } else if (payload.type === "error") {
        log(payload.message, "error");
    } else if (payload.type === "config") {
        renderConfig(payload);
    }
}

function renderConfig(payload) {
    if (payload.temperature !== undefined) {
        $("temperature").value = payload.temperature;
        $("temp-display").textContent = payload.temperature;
    }
    if (payload.backend && backendSelect) {
        backendSelect.value = "amt_small";
    }
    if (payload.response_strategy) {
        $("backend-display").textContent = "现场流式";
        $("response-engine-name").textContent = "Streaming AMT";
        $("response-engine-subtitle").textContent = "AMT 主体 + 动态调性 + 有限补齐";
        $("model-note").textContent = "AMT 音符生成后立即进入播放队列；Motif 只在空输出或 AMT 已形成主体后的有限尾部补齐中使用。";
    }
}

function renderSession(payload) {
    state.running = Boolean(payload.running);
    const startedAt = Number(payload.started_at || 0);
    if (state.running && startedAt && state.sessionStartedAt !== startedAt) {
        state.sessionStartedAt = startedAt;
        resetRoundDisplay();
    } else if (!state.running) {
        state.sessionStartedAt = null;
    }
    if (payload.input_port !== undefined) {
        state.sessionInput = payload.input_port;
        if (state.running && payload.input_port) {
            state.selectedInput = payload.input_port;
            if (Array.from(selectMidi.options).some((option) => option.value === payload.input_port)) {
                selectMidi.value = payload.input_port;
            }
        }
    }
    selectMidi.disabled = state.running;
    btnStart.classList.toggle("hidden", state.running);
    btnStop.classList.toggle("hidden", !state.running);

    if (state.running) {
        const label = SESSION_LABELS[payload.status] || "运行中";
        setStatus(statusEngine, `AI ${label}`, "connected");
        $("selected-source").textContent = payload.input_port || "Live input";
    } else if (payload.status === "error") {
        setStatus(statusEngine, "AI 发生错误", "error");
    } else {
        setStatus(statusEngine, "AI 尚未启动", "idle");
    }

    if (payload.round_id !== null && payload.round_id !== undefined) {
        state.round.id = payload.round_id;
        $("round-id").textContent = String(payload.round_id);
    }
    renderRhythm(state.rhythm);
    renderLoopBank(state.loopBank);
    renderPerformanceMode();
}

function resetRoundDisplay() {
    state.round = { id: null, state: "idle", firstEvent: "--", underruns: 0 };
    $("round-id").textContent = "--";
    $("round-state").textContent = "待命";
    $("first-event").textContent = "--";
    $("underruns").textContent = "0";
    $("stage-subtitle").textContent = "弹奏一句并自然停顿，AI 会接着回应。";
}

function isComputerTestMode() {
    const input = String(state.sessionInput || state.selectedInput || "").toLowerCase();
    return input.includes("python_in") || Boolean(state.devices.virtual_mode);
}

function renderPerformanceMode() {
    if (isComputerTestMode()) {
        setStatus(statusTestMode, "电脑键盘 → Logic", "connected");
        return;
    }
    setStatus(statusTestMode, "MiniLab 模式", "connected");
}

function renderPianoHost(payload) {
    state.pianoHost = payload;
    setStatus(statusPianoHost, "Logic 唯一音频输出", "connected");
}

function renderRhythm(payload) {
    state.rhythm = { ...state.rhythm, ...payload };
    const rhythm = state.rhythm;
    const labels = {
        idle: "待命",
        learning: "正在识别",
        running: "节奏运行中",
        replacing: "下一小节替换",
        stopping: "下一小节停止"
    };
    const label = labels[rhythm.state] || "待命";
    $("rhythm-state-label").textContent = label;
    setStatus(statusRhythm, `节奏 ${label}`, rhythm.learning || rhythm.playing ? "connected" : "idle");

    if (rhythm.learning) {
        $("rhythm-detail").textContent = rhythm.tap_count > 0
            ? `已收到 ${rhythm.tap_count} 次敲击。继续示范，停下约两秒会自动完成识别。`
            : "现在用空格、TAP 按钮或 MiniLab 的 Pad 示范节奏。键盘琴键仍可正常演奏旋律。";
    } else if (rhythm.replacing) {
        $("rhythm-detail").textContent = "新节奏已经识别，会在下一小节边界自然替换当前伴奏。";
    } else if (rhythm.stopping) {
        $("rhythm-detail").textContent = "伴奏会在当前小节结束时停止。";
    } else if (rhythm.playing) {
        $("rhythm-detail").textContent = "MFP DRUMS 正在接收稳定鼓组 MIDI；MFP MELODY 的输入和 AI 回应会同时保持可用。";
    } else {
        $("rhythm-detail").textContent = state.running
            ? "点“学习新节奏”，然后敲出你想要的速度和重音。"
            : "先启动引擎，再示范你想要的节奏。";
    }

    $("rhythm-bpm").textContent = rhythm.playing || rhythm.replacing ? Math.round(Number(rhythm.bpm || 0)) : "--";
    $("rhythm-meter-label").textContent = `4/4 · ${Math.max(1, Number(rhythm.bars || 1))} 小节`;
    $("rhythm-confidence").textContent = rhythm.learning
        ? `${rhythm.tap_count || 0} 次敲击`
        : rhythm.playing
            ? `识别 ${Math.round(Number(rhythm.confidence || 0) * 100)}%`
            : "等待示范";
    renderRhythmPattern(rhythm);

    $("rhythm-tap-btn").disabled = !state.running || !rhythm.learning;
    $("rhythm-learn-btn").disabled = !state.running || rhythm.learning;
    $("rhythm-finish-btn").disabled = !state.running || !rhythm.learning || rhythm.tap_count < 3;
    $("rhythm-stop-btn").disabled = !state.running || !rhythm.playing;
    $("rhythm-stop-now-btn").disabled = !state.running || (!rhythm.playing && !rhythm.learning);
    $("rhythm-variation-btn").disabled = !state.running || !rhythm.playing;
    renderRhythmBank(rhythm);
}

function renderRhythmBank(rhythm) {
    const saved = new Set(Array.isArray(rhythm.saved_slots) ? rhythm.saved_slots : []);
    for (const slot of ["A", "B", "C", "D"]) {
        const card = document.querySelector(`.rhythm-slot[data-slot="${slot}"]`);
        if (!card) continue;
        const isCurrent = rhythm.current_slot === slot;
        const isQueued = rhythm.queued_slot === slot;
        card.classList.toggle("has-content", saved.has(slot));
        card.classList.toggle("playing", isCurrent);
        card.classList.toggle("queued", isQueued);
        card.querySelector("span").textContent = isQueued
            ? "下一小节切换"
            : isCurrent
                ? `${Math.round(Number(rhythm.bpm || 0))} BPM · 使用中`
                : saved.has(slot)
                    ? `${Math.round(Number(rhythm.bpm || 0))} BPM · 已保存`
                    : "空";
        card.querySelector(".rhythm-save").disabled = !state.running || !rhythm.playing;
        card.querySelector(".rhythm-load").disabled = !state.running || !saved.has(slot);
    }
}

function renderRhythmPattern(rhythm) {
    const root = $("rhythm-pattern");
    const stepsPerBar = Math.max(1, Number(rhythm.steps_per_bar || 16));
    const totalSteps = Math.min(64, stepsPerBar * Math.max(1, Number(rhythm.bars || 1)));
    const active = new Set(Array.isArray(rhythm.pattern) ? rhythm.pattern.map(Number) : []);
    root.innerHTML = "";
    for (let step = 0; step < totalSteps; step += 1) {
        const cell = document.createElement("span");
        cell.className = "rhythm-step";
        if (step % 4 === 0) cell.classList.add("beat");
        if (active.has(step)) cell.classList.add("hit");
        root.appendChild(cell);
    }
}

function renderLoopBank(payload) {
    state.loopBank = {
        ...state.loopBank,
        ...payload,
        slots: Array.isArray(payload.slots) ? payload.slots : state.loopBank.slots
    };
    const bank = state.loopBank;
    const modeSelect = $("loop-save-mode");
    if (modeSelect && modeSelect.value !== bank.mode) modeSelect.value = bank.mode;
    const ready = Boolean(bank.latest_ready);
    const active = bank.slots.some((slot) => slot.playing);
    const recentlyHeard = Date.now() - Number(bank.lastPlaybackAt || 0) < 900;
    setStatus(
        statusLoop,
        active ? (recentlyHeard ? "回应记忆正在发声" : "回应记忆播放中") : ready ? "回应可以保存" : "回应记忆为空",
        active || ready ? "connected" : "idle"
    );
    $("loop-detail").textContent = ready
        ? (bank.mode === "call_response" ? "最近一轮已准备好：保存会记录你的句子和 AI 回应。" : "最近一轮已准备好：保存会只记录 AI 回应。")
        : "等 AI 完成一轮回应后，就可以保存到 A、B、C 或 D。";

    for (const slot of bank.slots) {
        const card = document.querySelector(`.loop-slot[data-slot="${slot.name}"]`);
        if (!card) continue;
        card.classList.toggle("has-content", Boolean(slot.has_content));
        card.classList.toggle("playing", Boolean(slot.playing));
        card.classList.toggle("stopping", Boolean(slot.stopping));
        const meta = card.querySelector("span");
        if (meta) {
            if (slot.playing && slot.stopping) meta.textContent = "本轮后停";
            else if (slot.playing) meta.textContent = "循环播放中 · 已发送到 Logic";
            else if (slot.has_content) meta.textContent = `${slot.event_count} 音符 · ${Number(slot.loop_seconds || 0).toFixed(1)}s`;
            else meta.textContent = "空";
        }
        const save = card.querySelector(".loop-save");
        const toggle = card.querySelector(".loop-toggle");
        if (save) save.disabled = !state.running || !ready;
        if (toggle) {
            toggle.disabled = !state.running || !slot.has_content;
            toggle.textContent = slot.playing ? "下一轮停止" : "播放";
        }
    }
    $("loop-stop-all-btn").disabled = !state.running || !active;
}

function renderDevices() {
    const previous = state.selectedInput || selectMidi.value;
    selectMidi.innerHTML = "";

    const inputs = state.devices.inputs || [];
    if (inputs.length === 0) {
        const option = document.createElement("option");
        option.value = "";
        option.textContent = "未找到 MIDI 输入";
        selectMidi.appendChild(option);
    } else {
        for (const name of inputs) {
            const option = document.createElement("option");
            option.value = name;
            option.textContent = name;
            selectMidi.appendChild(option);
        }
    }

    state.selectedInput = previous && inputs.includes(previous) ? previous : state.devices.selected_input;
    if (state.selectedInput) selectMidi.value = state.selectedInput;

    $("midi-count").textContent = String(inputs.length);
    $("output-count").textContent = String((state.devices.outputs || []).length);
    $("selected-source").textContent = state.selectedInput || "电脑键盘";

    if (state.devices.midi_error) {
        $("device-note").textContent = `CoreMIDI unavailable: ${state.devices.midi_error}`;
    } else if (state.devices.virtual_mode) {
        $("device-note").textContent = "电脑模拟模式：字母键弹旋律，空格键示范节奏；功能与连接 MiniLab 时一致。";
    } else {
        $("device-note").textContent = `已选择：${state.selectedInput || state.devices.selected_input}`;
    }
    renderPerformanceMode();
}

function renderRoundState(payload) {
    const data = payload.data || {};
    const rawState = payload.state || "idle";
    const label = ROUND_LABELS[rawState] || rawState.replaceAll("_", " ");
    state.round.state = rawState;
    $("round-state").textContent = label;

    if (payload.round_id !== null && payload.round_id !== undefined) {
        state.round.id = payload.round_id;
        $("round-id").textContent = String(payload.round_id);
    }
    if (Number.isFinite(data.first_event_latency)) {
        $("first-event").textContent = `${data.first_event_latency.toFixed(2)}s`;
    }
    if (Number.isFinite(data.buffer_underruns)) {
        state.round.underruns = data.buffer_underruns;
        $("underruns").textContent = String(data.buffer_underruns);
    }
    if (Number.isFinite(data.call_notes)) {
        $("stage-subtitle").textContent = `收到 ${data.call_notes} 个音符${data.target_notes ? `，计划回应 ${data.target_notes} 个音符` : ""}`;
    }
}

function renderMetrics(payload) {
    $("round-id").textContent = String(payload.round_id ?? "--");
    $("round-state").textContent = ROUND_LABELS[payload.status] || payload.status || "完成";
    $("first-event").textContent = payload.first_event || "--";
    $("underruns").textContent = String(payload.underruns ?? 0);
}

function log(message, level = "info") {
    if (!message) return;
    const line = document.createElement("div");
    line.className = `log-line ${level === "error" ? "error" : level === "live" ? "live" : "info"}`;
    line.textContent = message;
    logPanel.appendChild(line);
    while (logPanel.children.length > 80) logPanel.firstChild.remove();
    logPanel.scrollTop = logPanel.scrollHeight;
}

function initAudio() {
    if (state.audio.ctx) {
        if (state.audio.ctx.state === "suspended") state.audio.ctx.resume();
        state.audio.ready = true;
        $("audio-status").textContent = "LOGIC MIDI";
        $("sampler-detail").textContent = "MFP MELODY";
        renderPerformanceMode();
        return;
    }

    const AudioContext = window.AudioContext || window.webkitAudioContext;
    if (!AudioContext) {
        $("audio-status").textContent = "LOGIC MIDI";
        return;
    }

    const audioCtx = new AudioContext();
    const master = audioCtx.createGain();
    const analyser = audioCtx.createAnalyser();
    const limiter = audioCtx.createDynamicsCompressor();
    const reverb = audioCtx.createConvolver();
    const wet = audioCtx.createGain();
    analyser.fftSize = 1024;
    master.gain.value = 1.15;
    limiter.threshold.value = -8;
    limiter.knee.value = 10;
    limiter.ratio.value = 8;
    limiter.attack.value = 0.003;
    limiter.release.value = 0.18;
    reverb.buffer = createPianoRoomImpulse(audioCtx);
    wet.gain.value = 0.22;
    master.connect(limiter);
    master.connect(reverb);
    reverb.connect(wet);
    wet.connect(limiter);
    limiter.connect(analyser);
    analyser.connect(audioCtx.destination);

    state.audio.ctx = audioCtx;
    state.audio.master = master;
    state.audio.analyser = analyser;
    state.audio.limiter = limiter;
    state.audio.reverb = reverb;
    state.audio.wet = wet;
    state.audio.ready = true;
    $("audio-status").textContent = "LOGIC MIDI";
    $("sampler-detail").textContent = "MFP MELODY";
    renderPerformanceMode();
}

function midiToFreq(pitch) {
    return 440 * Math.pow(2, (pitch - 69) / 12);
}

function midiNoteName(pitch) {
    return `${NOTE_NAMES[pitch % 12]}${Math.floor(pitch / 12) - 1}`;
}

async function loadAutumnSampler() {
    const audio = state.audio;
    if (audio.sampleManifest || !audio.ctx) return;

    try {
        const response = await fetch("/static/piano_samples/autumn/manifest.json");
        if (!response.ok) throw new Error(`manifest ${response.status}`);
        audio.sampleManifest = await response.json();
        audio.samplerReady = true;
        $("audio-status").textContent = "LOADING AUTUMN";
        $("sampler-detail").textContent = "Loading Autumn";
        preloadVisibleAutumnSamples();
    } catch (error) {
        audio.samplerReady = false;
        $("audio-status").textContent = "SYNTH READY";
        $("sampler-detail").textContent = "Synth Piano";
        console.warn("Autumn Piano sampler unavailable", error);
    }
}

function preloadVisibleAutumnSamples() {
    const audio = state.audio;
    if (!audio.sampleManifest) return;

    const pitches = Array.from(new Set([...whiteOffsets, ...blackOffsets].map((offset) => pitchFor(offset))));
    const promises = pitches.map((pitch) => {
        const region = findAutumnRegion(pitch, state.velocity);
        return region ? getAutumnBuffer(region) : Promise.resolve(null);
    });

    Promise.allSettled(promises).then(() => {
        if (audio.sampleManifest) {
            $("audio-status").textContent = "AUTUMN READY";
            $("sampler-detail").textContent = "Autumn Ready";
        }
    });
}

function findAutumnRegion(pitch, velocity) {
    const manifest = state.audio.sampleManifest;
    if (!manifest || !Array.isArray(manifest.regions)) return null;

    const direct = manifest.regions.find((region) => (
        pitch >= region.loNote &&
        pitch <= region.hiNote &&
        velocity >= region.loVel &&
        velocity <= region.hiVel
    ));
    if (direct) return direct;

    let best = null;
    let bestDistance = Infinity;
    for (const region of manifest.regions) {
        const noteDistance = Math.abs(pitch - region.rootNote);
        const velocityCenter = (region.loVel + region.hiVel) / 2;
        const velocityDistance = Math.abs(velocity - velocityCenter) / 32;
        const distance = noteDistance * 4 + velocityDistance;
        if (distance < bestDistance) {
            best = region;
            bestDistance = distance;
        }
    }
    return best;
}

async function getAutumnBuffer(region) {
    const audio = state.audio;
    if (!region || !audio.ctx) return null;
    const url = `/static/piano_samples/autumn/${region.url}`;

    if (audio.sampleBuffers.has(url)) {
        return audio.sampleBuffers.get(url);
    }
    if (audio.sampleLoading.has(url)) {
        return audio.sampleLoading.get(url);
    }

    const loading = fetch(url)
        .then((response) => {
            if (!response.ok) throw new Error(`sample ${response.status}`);
            return response.arrayBuffer();
        })
        .then((data) => audio.ctx.decodeAudioData(data))
        .then((buffer) => {
            audio.sampleBuffers.set(url, buffer);
            audio.sampleLoading.delete(url);
            return buffer;
        })
        .catch((error) => {
            audio.sampleLoading.delete(url);
            console.warn("Autumn Piano sample load failed", url, error);
            return null;
        });

    audio.sampleLoading.set(url, loading);
    return loading;
}

function playAutumnBuffer(pitch, velocity, region, buffer, voiceKey = pitch) {
    const audio = state.audio;
    const now = audio.ctx.currentTime;
    const source = audio.ctx.createBufferSource();
    const gain = audio.ctx.createGain();
    const tone = audio.ctx.createBiquadFilter();
    const v = Math.max(0.12, Math.min(1, velocity / 127));

    source.buffer = buffer;
    source.playbackRate.value = Math.pow(2, (pitch - region.rootNote) / 12);
    tone.type = "lowpass";
    tone.frequency.value = 2600 + velocity * 36;
    tone.Q.value = 0.55;
    gain.gain.setValueAtTime(0.0001, now);
    gain.gain.exponentialRampToValueAtTime(0.95 + v * 0.85, now + 0.012);
    gain.gain.setTargetAtTime(1.18 + v * 0.42, now + 0.08, 0.9);

    source.connect(tone);
    tone.connect(gain);
    gain.connect(audio.master);
    source.start(now);
    source.onended = () => {
        if (audio.active.get(voiceKey)?.source === source) {
            audio.active.delete(voiceKey);
        }
    };
    audio.active.set(voiceKey, { kind: "sample", source, gain });
    return true;
}

function createPianoRoomImpulse(audioCtx) {
    const duration = 1.35;
    const length = Math.floor(audioCtx.sampleRate * duration);
    const impulse = audioCtx.createBuffer(2, length, audioCtx.sampleRate);

    for (let channel = 0; channel < 2; channel++) {
        const data = impulse.getChannelData(channel);
        for (let i = 0; i < length; i++) {
            const t = i / length;
            const decay = Math.pow(1 - t, 2.6);
            data[i] = (Math.random() * 2 - 1) * decay * 0.42;
        }
    }

    return impulse;
}

function schedulePianoPartial(audioCtx, output, freq, now, level, ratio, attack, decay, sustain, detune = 0, type = "sine") {
    const osc = audioCtx.createOscillator();
    const gain = audioCtx.createGain();
    osc.type = type;
    osc.frequency.setValueAtTime(freq * ratio, now);
    osc.detune.setValueAtTime(detune, now);

    gain.gain.setValueAtTime(0.0001, now);
    gain.gain.exponentialRampToValueAtTime(Math.max(0.0002, level), now + attack);
    gain.gain.exponentialRampToValueAtTime(Math.max(0.0002, level * sustain), now + decay);

    osc.connect(gain);
    gain.connect(output);
    osc.start(now);
    return { osc, gain };
}

function playLocalPiano(pitch, velocity = 96, voiceKey = pitch) {
    initAudio();
    const audio = state.audio;
    if (!audio.ctx || audio.active.has(voiceKey)) return;

    const region = findAutumnRegion(pitch, velocity);
    if (region) {
        const url = `/static/piano_samples/autumn/${region.url}`;
        if (audio.sampleBuffers.has(url)) {
            playAutumnBuffer(pitch, velocity, region, audio.sampleBuffers.get(url), voiceKey);
            return;
        }
        getAutumnBuffer(region);
    }

    const now = audio.ctx.currentTime;
    const freq = midiToFreq(pitch);
    const voiceGain = audio.ctx.createGain();
    const filter = audio.ctx.createBiquadFilter();
    const body = audio.ctx.createBiquadFilter();

    filter.type = "lowpass";
    filter.frequency.setValueAtTime(Math.min(7200, 2600 + velocity * 34), now);
    filter.frequency.exponentialRampToValueAtTime(Math.max(1200, 1900 + (pitch - 60) * 10), now + 0.58);
    filter.Q.value = 0.72;

    body.type = "peaking";
    body.frequency.value = Math.max(120, freq * 1.5);
    body.Q.value = 0.9;
    body.gain.value = 2.3;

    const v = Math.max(0.12, Math.min(1, velocity / 127));
    const level = 0.13 + v * 0.27;
    voiceGain.gain.setValueAtTime(0.9, now);

    const partials = [
        schedulePianoPartial(audio.ctx, filter, freq, now, level * 0.92, 1.000, 0.008, 1.55, 0.34, -3, "triangle"),
        schedulePianoPartial(audio.ctx, filter, freq, now, level * 0.36, 2.004, 0.006, 0.82, 0.13, 4, "sine"),
        schedulePianoPartial(audio.ctx, filter, freq, now, level * 0.20, 3.010, 0.005, 0.48, 0.06, -7, "sine"),
        schedulePianoPartial(audio.ctx, filter, freq, now, level * 0.13, 4.016, 0.004, 0.34, 0.035, 8, "sine")
    ];

    const hammer = audio.ctx.createOscillator();
    const hammerGain = audio.ctx.createGain();
    const hammerFilter = audio.ctx.createBiquadFilter();
    hammer.type = "triangle";
    hammer.frequency.setValueAtTime(Math.min(4200, freq * 7.5), now);
    hammerFilter.type = "highpass";
    hammerFilter.frequency.value = 1300;
    hammerGain.gain.setValueAtTime(0.0001, now);
    hammerGain.gain.exponentialRampToValueAtTime(0.026 + v * 0.055, now + 0.004);
    hammerGain.gain.exponentialRampToValueAtTime(0.0001, now + 0.055);
    hammer.connect(hammerFilter);
    hammerFilter.connect(hammerGain);
    hammerGain.connect(filter);
    hammer.start(now);
    hammer.stop(now + 0.075);

    filter.connect(body);
    body.connect(voiceGain);
    voiceGain.connect(audio.master);

    audio.active.set(voiceKey, { partials, hammer, voiceGain });
}

function stopLocalPiano(pitch, voiceKey = pitch) {
    const audio = state.audio;
    const voice = audio.active.get(voiceKey);
    if (!audio.ctx || !voice) return;

    const now = audio.ctx.currentTime;
    if (voice.kind === "sample") {
        voice.gain.gain.cancelScheduledValues(now);
        voice.gain.gain.setTargetAtTime(0.0001, now, 0.22);
        try {
            voice.source.stop(now + 0.9);
        } catch (error) {
            // The sample may have naturally ended before key release.
        }
        audio.active.delete(voiceKey);
        return;
    }

    voice.voiceGain.gain.cancelScheduledValues(now);
    voice.voiceGain.gain.setTargetAtTime(0.0001, now, 0.34);
    for (const partial of voice.partials) {
        partial.gain.gain.cancelScheduledValues(now);
        partial.gain.gain.setTargetAtTime(0.0001, now, 0.28);
        partial.osc.stop(now + 1.15);
    }
    audio.active.delete(voiceKey);
}

function makeNoiseBuffer(audioCtx, seconds) {
    const length = Math.max(1, Math.floor(audioCtx.sampleRate * seconds));
    const buffer = audioCtx.createBuffer(1, length, audioCtx.sampleRate);
    const data = buffer.getChannelData(0);
    for (let i = 0; i < length; i++) data[i] = Math.random() * 2 - 1;
    return buffer;
}

function playLocalDrum(note, velocity = 96) {
    initAudio();
    const audio = state.audio;
    if (!audio.ctx || !audio.master) return;

    const now = audio.ctx.currentTime;
    const level = Math.max(0.2, Math.min(1, velocity / 127));
    const output = audio.ctx.createGain();
    output.gain.value = 0.5 + level * 0.75;
    output.connect(audio.master);

    if (note === 36) {
        const osc = audio.ctx.createOscillator();
        const gain = audio.ctx.createGain();
        osc.type = "sine";
        osc.frequency.setValueAtTime(145, now);
        osc.frequency.exponentialRampToValueAtTime(46, now + 0.16);
        gain.gain.setValueAtTime(0.9, now);
        gain.gain.exponentialRampToValueAtTime(0.0001, now + 0.34);
        osc.connect(gain);
        gain.connect(output);
        osc.start(now);
        osc.stop(now + 0.36);
        return;
    }

    if ([45, 43].includes(note)) {
        const osc = audio.ctx.createOscillator();
        const gain = audio.ctx.createGain();
        osc.type = "triangle";
        osc.frequency.setValueAtTime(note === 45 ? 145 : 105, now);
        osc.frequency.exponentialRampToValueAtTime(note === 45 ? 82 : 64, now + 0.22);
        gain.gain.setValueAtTime(0.72, now);
        gain.gain.exponentialRampToValueAtTime(0.0001, now + 0.42);
        osc.connect(gain);
        gain.connect(output);
        osc.start(now);
        osc.stop(now + 0.44);
        return;
    }

    const duration = note === 46 || note === 49 || note === 51 ? 0.72 : note === 38 ? 0.24 : 0.12;
    const source = audio.ctx.createBufferSource();
    const filter = audio.ctx.createBiquadFilter();
    const gain = audio.ctx.createGain();
    source.buffer = makeNoiseBuffer(audio.ctx, duration);
    filter.type = note === 38 ? "bandpass" : "highpass";
    filter.frequency.value = note === 38 ? 1850 : note === 49 ? 5200 : 7200;
    filter.Q.value = note === 38 ? 0.7 : 0.3;
    gain.gain.setValueAtTime(note === 38 ? 0.72 : 0.38, now);
    gain.gain.exponentialRampToValueAtTime(0.0001, now + duration);
    source.connect(filter);
    filter.connect(gain);
    gain.connect(output);
    source.start(now);
}

function handlePlaybackNote(payload) {
    if (!isComputerTestMode()) return;
    const pitch = Number(payload.pitch);
    if (!Number.isFinite(pitch)) return;

    const event = payload.event || "note_on";
    const bus = payload.bus || "ai";
    if (bus === "drum") {
        pushVisualNote({ ...payload, source: "drum" });
        return;
    }

    pushVisualNote({ ...payload, source: bus === "loop" ? "ai" : bus });
}

function maybeVisualizeFromLog(message) {
    if (!message) return;

    const noteOn = message.match(/\[note_on\]\s+pitch=\s*(\d+)\s+velocity=\s*(\d+)/i);
    if (noteOn) {
        pushVisualNote({ pitch: Number(noteOn[1]), velocity: Number(noteOn[2]), source: "human", event: "note_on" });
        return;
    }

    const noteOff = message.match(/\[note_off\]\s+pitch=\s*(\d+)\s+velocity=\s*(\d+)/i);
    if (noteOff) {
        pushVisualNote({ pitch: Number(noteOff[1]), velocity: 0, source: "human", event: "note_off" });
        return;
    }

    const generated = message.match(/(?:queued|sampled) event #\s*(\d+):\s*tick=(-?\d+)\s+dur=(-?\d+)\s+pitch=(-?\d+)/i);
    if (generated) {
        pushVisualNote({
            pitch: Number(generated[4]),
            velocity: 88,
            source: "ai",
            event: "note_on",
            durationTicks: Number(generated[3])
        });
    }
}

function visualKey(source, pitch) {
    return `${source}:${pitch}`;
}

function normalizeVisualSource(source) {
    return source === "browser-midi" ? "human" : source;
}

function pushVisualNote(note) {
    const pitch = Number(note.pitch);
    if (!Number.isFinite(pitch)) return;

    const source = normalizeVisualSource(note.source || "human");
    const event = note.event || "note_on";
    const key = visualKey(source, pitch);

    if (event === "note_off") {
        state.activeVisualNotes.delete(key);
        markPianoKey(pitch, false);
        return;
    }

    if (state.activeVisualNotes.has(key)) {
        const active = state.activeVisualNotes.get(key);
        active.velocity = Math.max(active.velocity, Number(note.velocity || active.velocity || 90));
        markPianoKey(pitch, true);
        return;
    }

    const durationTicks = Number(note.duration_ticks ?? note.durationTicks ?? 0);
    const autoOffMs = source === "human"
        ? null
        : Math.max(180, Math.min(2800, durationTicks > 0 ? durationTicks * 10 : 520));

    state.activeVisualNotes.set(key, {
        pitch,
        velocity: Number(note.velocity || 90),
        source,
        born: performance.now(),
        autoOffAt: autoOffMs === null ? null : performance.now() + autoOffMs
    });
    spawnKeyboardParticle(pitch, source);
    markPianoKey(pitch, true);
}

function resizeCanvas() {
    const ratio = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    const w = Math.max(1, rect.width);
    const h = Math.max(1, rect.height);
    if (canvas.width !== Math.floor(w * ratio) || canvas.height !== Math.floor(h * ratio)) {
        canvas.width = Math.floor(w * ratio);
        canvas.height = Math.floor(h * ratio);
        ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    }
    return { w, h };
}

function resizeKeyboardCanvas() {
    if (!keyboardCanvas || !keyboardCtx) return { w: 0, h: 0 };
    const ratio = window.devicePixelRatio || 1;
    const rect = keyboardCanvas.getBoundingClientRect();
    const w = Math.max(1, rect.width);
    const h = Math.max(1, rect.height);
    if (keyboardCanvas.width !== Math.floor(w * ratio) || keyboardCanvas.height !== Math.floor(h * ratio)) {
        keyboardCanvas.width = Math.floor(w * ratio);
        keyboardCanvas.height = Math.floor(h * ratio);
        keyboardCtx.setTransform(ratio, 0, 0, ratio, 0, 0);
    }
    return { w, h };
}

function spawnKeyboardParticle(pitch, source) {
    if (!keyboardCanvas) return;
    const rect = keyboardCanvas.getBoundingClientRect();
    const w = Math.max(1, rect.width);
    const h = Math.max(1, rect.height);
    const colors = source === "ai"
        ? ["#ff4d63", "#ff8a9a", "#ffc857"]
        : ["#35d4ff", "#6aa8ff", "#3fe075"];
    const x = pitchToX(pitch, w);
    state.keyboardParticles.push({
        x: x + (Math.random() - 0.5) * 14,
        y: h - 14,
        vx: (Math.random() - 0.5) * 0.45,
        vy: -(0.65 + Math.random() * 0.85),
        life: 1,
        decay: 0.010 + Math.random() * 0.008,
        color: colors[Math.floor(Math.random() * colors.length)],
        size: 14 + Math.random() * 9,
        phase: Math.random() * Math.PI * 2
    });
    if (state.keyboardParticles.length > 80) {
        state.keyboardParticles.splice(0, state.keyboardParticles.length - 80);
    }
}

function draw() {
    const { w, h } = resizeCanvas();
    ctx.clearRect(0, 0, w, h);

    const now = performance.now();
    for (const [key, note] of Array.from(state.activeVisualNotes.entries())) {
        if (note.autoOffAt !== null && now >= note.autoOffAt) {
            state.activeVisualNotes.delete(key);
            markPianoKey(note.pitch, false);
        }
    }

    drawBackground(w, h);
    drawAudioWave(w, h);
    drawActiveBlocks(w, h);
    drawKeyboardParticles();

    requestAnimationFrame(draw);
}

function drawKeyboardParticles() {
    if (!keyboardCanvas || !keyboardCtx) return;
    const { w, h } = resizeKeyboardCanvas();
    keyboardCtx.clearRect(0, 0, w, h);

    const active = [];
    for (const p of state.keyboardParticles) {
        p.phase += 0.11;
        p.x += p.vx + Math.sin(p.phase) * 0.35;
        p.y += p.vy;
        p.life -= p.decay;

        if (p.life <= 0) continue;
        keyboardCtx.save();
        keyboardCtx.globalAlpha = Math.max(0, p.life);
        keyboardCtx.fillStyle = p.color;
        keyboardCtx.shadowColor = p.color;
        keyboardCtx.shadowBlur = 12;
        keyboardCtx.font = `${p.size}px Arial, sans-serif`;
        keyboardCtx.textAlign = "center";
        keyboardCtx.fillText(String.fromCharCode(9835), p.x, p.y);
        keyboardCtx.restore();
        active.push(p);
    }
    state.keyboardParticles = active;
}

function drawBackground(w, h) {
    const minPitch = 36;
    const maxPitch = 96;
    const top = 74;
    const mid = h * 0.54;

    const bg = ctx.createLinearGradient(0, 0, 0, h);
    bg.addColorStop(0, "#11141a");
    bg.addColorStop(0.52, "#0b0f15");
    bg.addColorStop(1, "#111218");
    ctx.fillStyle = bg;
    ctx.fillRect(0, 0, w, h);

    ctx.fillStyle = "rgba(255,255,255,0.035)";
    ctx.fillRect(0, top, w, 1);
    ctx.fillRect(0, mid, w, 1);

    for (let pitch = minPitch; pitch <= maxPitch; pitch++) {
        const x = pitchToX(pitch, w);
        const isC = pitch % 12 === 0;
        ctx.strokeStyle = isC ? "rgba(255,255,255,0.14)" : "rgba(255,255,255,0.045)";
        ctx.beginPath();
        ctx.moveTo(x, top);
        ctx.lineTo(x, h);
        ctx.stroke();
        if (isC) {
            ctx.fillStyle = "rgba(255,255,255,0.54)";
            ctx.font = "11px Inter, Segoe UI, sans-serif";
            ctx.fillText(`C${Math.floor(pitch / 12) - 1}`, x + 4, top + 18);
        }
    }

    drawLaneLabel("你的演奏", 18, top + 34, "#35d4ff");
    drawLaneLabel("AI 回应", 18, mid + 26, "#ff4d63");

    if (state.activeVisualNotes.size === 0) {
        ctx.fillStyle = "rgba(244,246,248,0.78)";
        ctx.font = "700 22px Inter, Segoe UI, sans-serif";
        ctx.textAlign = "center";
        const phase = ROUND_LABELS[state.round.state] || "等待演奏";
        ctx.fillText(phase, w / 2, h / 2 - 4);
        ctx.font = "13px Inter, Segoe UI, sans-serif";
        ctx.fillStyle = "rgba(154,163,173,0.9)";
        ctx.fillText(state.running ? "弹奏一句，然后自然停顿" : "启动 AI 后开始", w / 2, h / 2 + 24);
        ctx.textAlign = "left";
    }
}

function drawLaneLabel(text, x, y, color) {
    ctx.fillStyle = color;
    ctx.font = "800 12px Inter, Segoe UI, sans-serif";
    ctx.fillText(text, x, y);
}

function drawAudioWave(w, h) {
    const analyser = state.audio.analyser;
    if (!analyser) return;

    const buffer = new Uint8Array(analyser.fftSize);
    analyser.getByteTimeDomainData(buffer);
    ctx.save();
    ctx.strokeStyle = "rgba(63,224,117,0.9)";
    ctx.lineWidth = 2;
    ctx.shadowBlur = 14;
    ctx.shadowColor = "rgba(63,224,117,0.55)";
    ctx.beginPath();
    for (let i = 0; i < buffer.length; i++) {
        const x = (i / (buffer.length - 1)) * w;
        const y = 38 + ((buffer[i] - 128) / 128) * 24;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.restore();
}

function pitchToX(pitch, w) {
    const minPitch = 36;
    const maxPitch = 96;
    const clamped = Math.max(minPitch, Math.min(maxPitch, pitch));
    return 48 + ((clamped - minPitch) / (maxPitch - minPitch)) * Math.max(1, w - 96);
}

function drawRoundRect(x, y, width, height, radius) {
    if (typeof ctx.roundRect === "function") {
        ctx.roundRect(x, y, width, height, radius);
        return;
    }
    const r = Math.min(radius, width / 2, height / 2);
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + width, y, x + width, y + height, r);
    ctx.arcTo(x + width, y + height, x, y + height, r);
    ctx.arcTo(x, y + height, x, y, r);
    ctx.arcTo(x, y, x + width, y, r);
}

function drawActiveBlocks(w, h) {
    const now = performance.now();
    const mid = h * 0.54;
    for (const note of state.activeVisualNotes.values()) {
        const x = pitchToX(note.pitch, w);
        const isAi = note.source === "ai";
        const isTest = note.source === "test";
        const laneTop = isAi ? mid + 28 : 92;
        const laneBottom = isAi ? h - 34 : mid - 28;
        const laneHeight = Math.max(32, laneBottom - laneTop);
        const age = Math.max(0, now - note.born);
        const growPx = age * 0.17;
        const blockH = Math.max(24, Math.min(laneHeight, growPx));
        const y = laneBottom - blockH;
        const blockW = Math.max(28, Math.min(62, 20 + note.velocity * 0.30));
        const color = isAi ? "#ff4d63" : isTest ? "#3fe075" : "#35d4ff";
        const pulse = 0.86 + Math.sin(age / 120) * 0.14;

        const gradient = ctx.createLinearGradient(0, y, 0, y + blockH);
        gradient.addColorStop(0, hexToRgba(color, 0.16));
        gradient.addColorStop(0.72, hexToRgba(color, 0.78));
        gradient.addColorStop(1, color);

        ctx.save();
        ctx.shadowBlur = 22;
        ctx.shadowColor = color;
        ctx.globalAlpha = pulse;
        ctx.fillStyle = gradient;
        ctx.beginPath();
        drawRoundRect(x - blockW / 2, y, blockW, blockH, 6);
        ctx.fill();
        ctx.restore();

        ctx.fillStyle = color;
        ctx.beginPath();
        drawRoundRect(x - blockW / 2 - 3, laneBottom - 9, blockW + 6, 18, 5);
        ctx.fill();

        ctx.fillStyle = "rgba(255,255,255,0.96)";
        ctx.font = "700 11px Inter, Segoe UI, sans-serif";
        ctx.textAlign = "center";
        ctx.fillText(String(note.pitch), x, laneBottom + 4);
        ctx.textAlign = "left";
    }
}

function hexToRgba(hex, alpha) {
    const clean = hex.replace("#", "");
    const value = Number.parseInt(clean, 16);
    const r = (value >> 16) & 255;
    const g = (value >> 8) & 255;
    const b = value & 255;
    return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

function buildOffsets(span, wantWhite) {
    const offsets = [];
    for (let offset = 0; offset < span; offset++) {
        const isWhite = WHITE_SEMITONES.has(offset % 12);
        if (isWhite === wantWhite) offsets.push(offset);
    }
    return offsets;
}

function typingBaseOffset() {
    return state.keyboardMode === "compact" ? 12 : 24;
}

function rebuildKeyboardMaps() {
    const span = KEYBOARD_SPANS[state.keyboardMode] || KEYBOARD_SPANS.standard;
    whiteOffsets = buildOffsets(span, true);
    blackOffsets = buildOffsets(span, false);
    keyMap = {};
    keyboardLabels = {};

    const base = typingBaseOffset();
    TYPING_KEYS.forEach((key, index) => {
        const offset = base + index;
        if (offset < span) {
            keyMap[key] = offset;
            keyboardLabels[offset] = key.toUpperCase();
        }
    });
}

function syncKeyboardWindowState() {
    const deck = document.querySelector(".logic-keyboard-window");
    if (!deck) return;

    deck.classList.toggle("keyboard-compact", state.keyboardMode === "compact");
    deck.classList.toggle("keyboard-expanded", state.keyboardMode === "expanded");
    deck.classList.toggle("keyboard-minimized", state.keyboardMinimized);
    $("keyboard-compact")?.classList.toggle("is-active", state.keyboardMode === "compact");
    $("keyboard-expand")?.classList.toggle("is-active", state.keyboardMode === "expanded");
    $("keyboard-minimize")?.classList.toggle("is-active", state.keyboardMinimized);
}

function applyKeyboardMode(mode) {
    if (!KEYBOARD_SPANS[mode] || state.keyboardMode === mode) return;
    allNotesOff();
    state.keyboardMode = mode;
    rebuildKeyboardMaps();
    syncKeyboardWindowState();
    buildKeyboard();
    updateKeyboardRangeDetail();
    resizeKeyboardCanvas();
}

function toggleKeyboardMinimized() {
    allNotesOff();
    state.keyboardMinimized = !state.keyboardMinimized;
    syncKeyboardWindowState();
    resizeKeyboardCanvas();
}

function toggleKeyboardCompact() {
    state.keyboardMinimized = false;
    applyKeyboardMode(state.keyboardMode === "compact" ? "standard" : "compact");
}

function toggleKeyboardExpanded() {
    state.keyboardMinimized = false;
    applyKeyboardMode(state.keyboardMode === "expanded" ? "standard" : "expanded");
}

function buildKeyboard() {
    const root = $("keyboard");
    root.innerHTML = "";
    const whiteWidth = 100 / whiteOffsets.length;

    whiteOffsets.forEach((offset, index) => {
        const key = document.createElement("div");
        key.className = "white-key";
        key.dataset.offset = offset;
        key.style.left = `${index * whiteWidth}%`;
        key.style.width = `${whiteWidth}%`;
        key.innerHTML = `<span class="key-label">${keyboardLabels[offset] || ""}</span>`;
        attachPointerKey(key, offset);
        root.appendChild(key);
    });

    blackOffsets.forEach((offset) => {
        const previousWhite = whiteOffsets.findIndex((value) => value > offset) - 1;
        const x = (previousWhite + 0.72) * whiteWidth;
        const key = document.createElement("div");
        key.className = "black-key";
        key.dataset.offset = offset;
        key.style.left = `${x}%`;
        key.style.width = `${whiteWidth * 0.62}%`;
        key.innerHTML = `<span class="key-label">${keyboardLabels[offset] || ""}</span>`;
        attachPointerKey(key, offset);
        root.appendChild(key);
    });
}

function attachPointerKey(el, offset) {
    el.addEventListener("pointerdown", (e) => {
        e.preventDefault();
        el.setPointerCapture(e.pointerId);
        noteOn(offset, "virtual");
    });
    el.addEventListener("pointerup", () => noteOff(offset, "virtual"));
    el.addEventListener("pointercancel", () => noteOff(offset, "virtual"));
    el.addEventListener("pointerleave", () => noteOff(offset, "virtual"));
}

function pitchFor(offset) {
    return state.octave * 12 + offset;
}

function offsetForPitch(pitch) {
    return pitch - state.octave * 12;
}

function noteOn(offset, source = "virtual") {
    const pitch = pitchFor(offset);
    if (state.activeKeys.has(pitch)) return;
    state.activeKeys.add(pitch);
    // Melody audio must come from Logic Pro.  A browser-side piano here would
    // double the selected Logic instrument and make the performer's Call sound
    // different from the AI response.
    send({ type: "note_on", pitch, velocity: state.velocity });
    pushVisualNote({ pitch, velocity: state.velocity, source: "human", event: "note_on" });
    markPianoKey(pitch, true);
    $("selected-source").textContent = source === "virtual" ? "虚拟键盘" : "电脑键盘";
}

function noteOff(offset) {
    const pitch = pitchFor(offset);
    if (!state.activeKeys.has(pitch)) return;
    state.activeKeys.delete(pitch);
    send({ type: "note_off", pitch, velocity: 0 });
    pushVisualNote({ pitch, velocity: 0, source: "human", event: "note_off" });
    markPianoKey(pitch, false);
}

function rhythmTap() {
    if (!state.running || !state.rhythm.learning) return;
    const pitch = 37;
    send({ type: "rhythm_tap", velocity: state.velocity });
    pushVisualNote({ pitch, velocity: state.velocity, source: "human", event: "note_on" });
    $("rhythm-tap-btn").classList.add("active");
    $("selected-source").textContent = "节奏示范";
    setTimeout(() => {
        pushVisualNote({ pitch, velocity: 0, source: "human", event: "note_off" });
        $("rhythm-tap-btn").classList.remove("active");
    }, 72);
}

function markPianoKey(pitch, active) {
    const offset = offsetForPitch(pitch);
    const el = document.querySelector(`[data-offset="${offset}"]`);
    if (el) el.classList.toggle("active", active);
}

function allNotesOff() {
    Array.from(state.activeKeys).forEach((pitch) => {
        send({ type: "note_off", pitch, velocity: 0 });
        pushVisualNote({ pitch, velocity: 0, source: "human", event: "note_off" });
    });
    state.activeKeys.clear();
    document.querySelectorAll(".active").forEach((el) => el.classList.remove("active"));
}

function stopAllLocalAudio() {
    allNotesOff();
    for (const voiceKey of Array.from(state.audio.active.keys())) {
        stopLocalPiano(0, voiceKey);
    }
}

function setOctave(v) {
    allNotesOff();
    state.octave = Math.max(1, Math.min(6, v));
    $("octave-display").textContent = state.octave;
    updateKeyboardRangeDetail();
}

function setVelocity(v) {
    state.velocity = Math.max(20, Math.min(127, v));
    $("velocity-display").textContent = state.velocity;
}

function updateKeyboardRangeDetail() {
    const low = pitchFor(0);
    const highOffset = Math.max(...whiteOffsets, ...blackOffsets);
    const high = pitchFor(highOffset);
    $("range-detail").textContent = `${midiNoteName(low)} - ${midiNoteName(high)}`;
}

btnStart.onclick = () => {
    send({ type: "start_session", input_port: selectMidi.value || state.devices.selected_input });
};
btnStop.onclick = () => send({ type: "stop_session" });
$("panic-btn").onclick = () => {
    stopAllLocalAudio();
    send({ type: "panic_all" });
};
btnTest.onclick = () => {
    send({ type: "test_output" });
};
btnRefresh.onclick = () => {
    send({ type: "refresh_devices" });
};
$("rhythm-tap-btn").onclick = rhythmTap;
$("rhythm-learn-btn").onclick = () => send({ type: "rhythm_learn_start" });
$("rhythm-finish-btn").onclick = () => send({ type: "rhythm_learn_finish" });
$("rhythm-variation-btn").onclick = () => send({ type: "rhythm_variation" });
$("rhythm-stop-btn").onclick = () => send({ type: "rhythm_stop" });
$("rhythm-stop-now-btn").onclick = () => send({ type: "rhythm_stop_now" });
document.querySelectorAll(".rhythm-save").forEach((button) => {
    button.onclick = () => send({ type: `rhythm_save_${button.dataset.slot.toLowerCase()}` });
});
document.querySelectorAll(".rhythm-load").forEach((button) => {
    button.onclick = () => send({ type: `rhythm_load_${button.dataset.slot.toLowerCase()}` });
});
$("loop-save-mode").onchange = (event) => {
    const mode = event.target.value === "call_response" ? "call_response" : "response";
    send({ type: `loop_set_mode_${mode}` });
};
document.querySelectorAll(".loop-save").forEach((button) => {
    button.onclick = () => send({ type: `loop_save_${button.dataset.slot.toLowerCase()}` });
});
document.querySelectorAll(".loop-toggle").forEach((button) => {
    button.onclick = () => send({ type: `loop_toggle_${button.dataset.slot.toLowerCase()}` });
});
$("loop-stop-all-btn").onclick = () => send({ type: "loop_stop_all" });
selectMidi.onchange = (e) => {
    state.selectedInput = e.target.value;
    $("selected-source").textContent = e.target.value || "电脑键盘";
    $("device-note").textContent = e.target.value ? `已选择：${e.target.value}` : "未选择 MIDI 输入";
};

$("temperature").addEventListener("input", (e) => {
    $("temp-display").textContent = e.target.value;
});
$("temperature").addEventListener("change", (e) => {
    send({ type: "set_params", params: { temperature: Number(e.target.value) } });
});

backendSelect?.addEventListener("change", (e) => {
    const selected = MODEL_CONFIG[e.target.value] || MODEL_CONFIG.amt_small;
    $("backend-display").textContent = selected.backend.toUpperCase();
    $("model-note").textContent = state.running
        ? `Next start: ${selected.label}`
        : `Current: ${selected.label}`;
    send({
        type: "set_params",
        params: {
            backend: selected.backend,
            model_id: selected.model_id
        }
    });
});

$("octave-down").onclick = () => setOctave(state.octave - 1);
$("octave-up").onclick = () => setOctave(state.octave + 1);
$("velocity-down").onclick = () => setVelocity(state.velocity - 10);
$("velocity-up").onclick = () => setVelocity(state.velocity + 10);
$("keyboard-minimize").onclick = toggleKeyboardMinimized;
$("keyboard-compact").onclick = toggleKeyboardCompact;
$("keyboard-expand").onclick = toggleKeyboardExpanded;

document.addEventListener("keydown", (e) => {
    if (e.repeat || ["INPUT", "SELECT", "TEXTAREA"].includes(e.target.tagName)) return;
    const key = e.key.toLowerCase();
    if (e.code === "Space") {
        e.preventDefault();
        rhythmTap();
        return;
    }
    if (key === "z") return setOctave(state.octave - 1);
    if (key === "x") return setOctave(state.octave + 1);
    if (key === "c") return setVelocity(state.velocity - 10);
    if (key === "v") return setVelocity(state.velocity + 10);
    if (keyMap[key] !== undefined) {
        e.preventDefault();
        noteOn(keyMap[key], "computer");
    }
});

document.addEventListener("keyup", (e) => {
    const key = e.key.toLowerCase();
    if (keyMap[key] !== undefined) {
        e.preventDefault();
        noteOff(keyMap[key]);
    }
});

rebuildKeyboardMaps();
syncKeyboardWindowState();
buildKeyboard();
updateKeyboardRangeDetail();
connect();
setStatus(statusWebMidi, "后台独占 MIDI 输入", "connected");
draw();
window.addEventListener("resize", resizeCanvas);
window.addEventListener("blur", allNotesOff);
document.addEventListener("visibilitychange", () => {
    if (document.hidden) allNotesOff();
});
