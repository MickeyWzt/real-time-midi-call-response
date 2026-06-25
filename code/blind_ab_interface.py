"""
Local blind A/B listening interface for MIDI call-and-response experiments.

The server reads existing answer_key.csv files, builds same-call/same-trial
pairs, serves the MIDI files, and records blind choices to CSV. The browser
can use online SoundFont samples for a better piano timbre, with a local
Web Audio fallback when offline.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import parse_qs, urlparse


ROOT = Path(__file__).resolve().parents[1]
RAW_RUN = ROOT / "ab_tests" / "objective_search_aria_10call" / "pentatonic_low_temp_no_strongbeat"
STYLED_RUN = ROOT / "ab_tests" / "objective_search_aria_10call_styled" / "pentatonic_low_temp_no_strongbeat"
OUTPUT_DIR = ROOT / "ab_tests" / "blind_ab_interface"
RATINGS_CSV = OUTPUT_DIR / "ratings.csv"

FILE_REGISTRY: Dict[str, Path] = {}
PAIR_SETS: Dict[str, Dict[str, Any]] = {}


def read_answer_key(run_dir: Path, variant: str) -> List[Dict[str, str]]:
    path = run_dir / "answer_key.csv"
    if not path.exists():
        raise SystemExit(f"Missing answer_key.csv: {path}")
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["variant"] = variant
    return rows


def register_file(path_text: str) -> str:
    path = Path(path_text).resolve()
    try:
        path.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise SystemExit(f"Refusing to serve file outside workspace: {path}") from exc
    file_id = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:16]
    FILE_REGISTRY[file_id] = path
    return file_id


def side_from_row(row: Dict[str, str]) -> Dict[str, Any]:
    return {
        "sample_id": row["sample_id"],
        "candidate": row["candidate"],
        "model_id": row.get("model_id", ""),
        "variant": row["variant"],
        "response_notes": row.get("response_notes", ""),
        "combined_id": register_file(row["combined_midi"]),
        "response_id": register_file(row["response_only_midi"]),
    }


def index_rows(rows: Iterable[Dict[str, str]]) -> Dict[tuple[str, str, str], Dict[str, str]]:
    indexed: Dict[tuple[str, str, str], Dict[str, str]] = {}
    for row in rows:
        indexed[(row["call_id"], row["trial"], row["candidate"])] = row
    return indexed


def build_pair_set(
    set_id: str,
    title: str,
    left_rows: Dict[tuple[str, str, str], Dict[str, str]],
    left_candidate: str,
    right_rows: Dict[tuple[str, str, str], Dict[str, str]],
    right_candidate: str,
) -> Dict[str, Any]:
    pairs: List[Dict[str, Any]] = []
    left_keys = {
        (call_id, trial)
        for call_id, trial, candidate in left_rows
        if candidate == left_candidate
    }
    right_keys = {
        (call_id, trial)
        for call_id, trial, candidate in right_rows
        if candidate == right_candidate
    }
    for call_id, trial in sorted(left_keys & right_keys):
        left_row = left_rows[(call_id, trial, left_candidate)]
        right_row = right_rows[(call_id, trial, right_candidate)]
        sides = [side_from_row(left_row), side_from_row(right_row)]
        rng = random.Random(f"{set_id}:{call_id}:{trial}:20260530")
        rng.shuffle(sides)
        pair_id = hashlib.sha1(f"{set_id}:{call_id}:{trial}".encode("utf-8")).hexdigest()[:12]
        pairs.append(
            {
                "pair_id": pair_id,
                "call_id": call_id,
                "trial": trial,
                "left": sides[0],
                "right": sides[1],
            }
        )
    rng = random.Random(f"{set_id}:order:20260530")
    rng.shuffle(pairs)
    return {"set_id": set_id, "title": title, "pairs": pairs}


def build_pair_sets() -> None:
    raw_rows = read_answer_key(RAW_RUN, "raw")
    styled_rows = read_answer_key(STYLED_RUN, "styled")
    raw_index = index_rows(raw_rows)
    styled_index = index_rows(styled_rows)

    for pair_set in [
        build_pair_set(
            "aria_styled_vs_amt",
            "Aria styled vs AMT controlled",
            styled_index,
            "aria_medium_gen",
            raw_index,
            "amt_small_controlled",
        ),
        build_pair_set(
            "aria_raw_vs_amt",
            "Aria raw vs AMT controlled",
            raw_index,
            "aria_medium_gen",
            raw_index,
            "amt_small_controlled",
        ),
        build_pair_set(
            "aria_styled_vs_motif",
            "Aria styled vs motif baseline",
            styled_index,
            "aria_medium_gen",
            raw_index,
            "motif_transform_baseline",
        ),
    ]:
        PAIR_SETS[pair_set["set_id"]] = pair_set


def public_pair(pair: Dict[str, Any], rating: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    return {
        "pair_id": pair["pair_id"],
        "call_id": pair["call_id"],
        "trial": pair["trial"],
        "rated": rating is not None,
        "rated_choice": rating.get("choice", "") if rating else "",
        "rated_timestamp": rating.get("timestamp", "") if rating else "",
        "left": {
            "sample_id": pair["left"]["sample_id"],
            "combined_id": pair["left"]["combined_id"],
            "response_id": pair["left"]["response_id"],
            "response_notes": pair["left"]["response_notes"],
        },
        "right": {
            "sample_id": pair["right"]["sample_id"],
            "combined_id": pair["right"]["combined_id"],
            "response_id": pair["right"]["response_id"],
            "response_notes": pair["right"]["response_notes"],
        },
    }


def find_pair(set_id: str, pair_id: str) -> Optional[Dict[str, Any]]:
    pair_set = PAIR_SETS.get(set_id)
    if not pair_set:
        return None
    for pair in pair_set["pairs"]:
        if pair["pair_id"] == pair_id:
            return pair
    return None


def reveal_payload(pair: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "left": {
            "candidate": pair["left"]["candidate"],
            "model_id": pair["left"]["model_id"],
            "variant": pair["left"]["variant"],
        },
        "right": {
            "candidate": pair["right"]["candidate"],
            "model_id": pair["right"]["model_id"],
            "variant": pair["right"]["variant"],
        },
    }


def append_rating(payload: Dict[str, Any], pair: Dict[str, Any]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    choice = payload.get("choice", "")
    chosen = ""
    if choice == "left":
        chosen = pair["left"]["candidate"]
    elif choice == "right":
        chosen = pair["right"]["candidate"]
    elif choice == "tie":
        chosen = "tie"
    fields = [
        "timestamp",
        "set_id",
        "pair_id",
        "call_id",
        "trial",
        "choice",
        "chosen_candidate",
        "play_mode",
        "left_candidate",
        "left_model_id",
        "left_variant",
        "right_candidate",
        "right_model_id",
        "right_variant",
        "left_sample_id",
        "right_sample_id",
    ]
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "set_id": payload.get("set_id", ""),
        "pair_id": pair["pair_id"],
        "call_id": pair["call_id"],
        "trial": pair["trial"],
        "choice": choice,
        "chosen_candidate": chosen,
        "play_mode": payload.get("play_mode", ""),
        "left_candidate": pair["left"]["candidate"],
        "left_model_id": pair["left"]["model_id"],
        "left_variant": pair["left"]["variant"],
        "right_candidate": pair["right"]["candidate"],
        "right_model_id": pair["right"]["model_id"],
        "right_variant": pair["right"]["variant"],
        "left_sample_id": pair["left"]["sample_id"],
        "right_sample_id": pair["right"]["sample_id"],
    }
    write_header = not RATINGS_CSV.exists()
    with RATINGS_CSV.open("a", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def ratings_by_pair(set_id: str) -> Dict[str, Dict[str, str]]:
    ratings: Dict[str, Dict[str, str]] = {}
    if not RATINGS_CSV.exists():
        return ratings
    with RATINGS_CSV.open("r", newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if row.get("set_id") == set_id and row.get("pair_id") and row["pair_id"] not in ratings:
                ratings[row["pair_id"]] = row
    return ratings


def ratings_count(set_id: str) -> int:
    return len(ratings_by_pair(set_id))


INDEX_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>MFP Blind A/B</title>
  <script src="https://cdn.jsdelivr.net/npm/soundfont-player@0.12.0/dist/soundfont-player.min.js"></script>
  <style>
    :root {
      --bg: #f5f6f8;
      --panel: #ffffff;
      --ink: #17202a;
      --muted: #697386;
      --line: #dce2ea;
      --accent: #206b5f;
      --accent-2: #3949ab;
      --warn: #ad5b1d;
      --shadow: 0 14px 38px rgba(29, 41, 57, 0.10);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: "Segoe UI", "Microsoft YaHei", Arial, sans-serif;
      letter-spacing: 0;
    }
    .shell {
      min-height: 100vh;
      display: grid;
      grid-template-rows: auto 1fr auto;
    }
    header {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 18px;
      align-items: center;
      padding: 18px 28px 12px;
      border-bottom: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.82);
      backdrop-filter: blur(12px);
      position: sticky;
      top: 0;
      z-index: 5;
    }
    h1 {
      margin: 0;
      font-size: 22px;
      line-height: 1.25;
      font-weight: 720;
    }
    .sub {
      margin-top: 5px;
      color: var(--muted);
      font-size: 13px;
    }
    .top-controls {
      display: flex;
      gap: 10px;
      align-items: center;
      flex-wrap: wrap;
      justify-content: flex-end;
    }
    select, button {
      font: inherit;
      border: 1px solid var(--line);
      background: #fff;
      color: var(--ink);
      border-radius: 8px;
      height: 38px;
      padding: 0 12px;
    }
    button {
      cursor: pointer;
      font-weight: 650;
      transition: transform 120ms ease, background 120ms ease, border-color 120ms ease;
    }
    button:hover { transform: translateY(-1px); border-color: #b7c3d2; }
    button:disabled { cursor: not-allowed; opacity: .45; transform: none; }
    .primary { background: var(--accent); color: #fff; border-color: var(--accent); }
    .choice-left { background: #edf6ff; border-color: #9fc8f2; }
    .choice-right { background: #fff2e9; border-color: #f0b985; }
    .choice-tie { background: #eef2f6; }
    main {
      width: min(1440px, 100%);
      margin: 0 auto;
      padding: 18px 24px 18px;
    }
    .status-row {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 16px;
      align-items: center;
      margin-bottom: 14px;
    }
    .progress-wrap {
      display: grid;
      gap: 8px;
    }
    .progress-text {
      display: flex;
      gap: 14px;
      flex-wrap: wrap;
      color: var(--muted);
      font-size: 13px;
    }
    .bar {
      height: 8px;
      border-radius: 999px;
      background: #dfe5ee;
      overflow: hidden;
    }
    .bar > div {
      height: 100%;
      width: 0;
      background: linear-gradient(90deg, var(--accent), var(--accent-2));
      transition: width 180ms ease;
    }
    .ab-grid {
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
      gap: 18px;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 10px;
      box-shadow: var(--shadow);
      overflow: hidden;
      min-width: 0;
    }
    .panel-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 14px 16px;
      border-bottom: 1px solid var(--line);
    }
    .side-title {
      font-size: 18px;
      font-weight: 760;
    }
    .small-meta {
      color: var(--muted);
      font-size: 12px;
      white-space: nowrap;
    }
    canvas {
      display: block;
      width: 100%;
      height: 250px;
      background: #fbfcfe;
    }
    .panel-controls {
      display: flex;
      justify-content: space-between;
      gap: 10px;
      align-items: center;
      padding: 12px 16px 14px;
      border-top: 1px solid var(--line);
    }
    .reveal {
      min-height: 20px;
      color: var(--muted);
      font-size: 13px;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .transport {
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 10px;
      padding: 16px 0 12px;
      flex-wrap: wrap;
    }
    .transport button { min-width: 116px; }
    .choices {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
      margin-top: 12px;
    }
    .choices button {
      height: 48px;
      font-size: 15px;
    }
    .message {
      min-height: 24px;
      margin-top: 12px;
      color: var(--muted);
      font-size: 14px;
      text-align: center;
    }
    footer {
      color: var(--muted);
      font-size: 12px;
      padding: 0 28px 18px;
      text-align: right;
    }
    @media (max-width: 900px) {
      header { grid-template-columns: 1fr; }
      .top-controls { justify-content: flex-start; }
      main { padding: 14px; }
      .ab-grid { grid-template-columns: 1fr; }
      .choices { grid-template-columns: 1fr 1fr; }
    }
  </style>
</head>
<body>
  <div class="shell">
    <header>
      <div>
        <h1>MFP Blind A/B Listening</h1>
        <div class="sub">同一条 call，同一 trial，左右随机隐藏；选择会自动写入 CSV。</div>
      </div>
      <div class="top-controls">
        <select id="setSelect"></select>
        <select id="modeSelect">
          <option value="combined">Call + Response</option>
          <option value="response">Response only</option>
        </select>
        <select id="soundSelect">
          <option value="sampled">联网采样钢琴</option>
          <option value="soft">本地柔和钢琴</option>
        </select>
      </div>
    </header>

    <main>
      <div class="status-row">
        <div class="progress-wrap">
          <div class="progress-text">
            <span id="progressText">Loading</span>
            <span id="callText"></span>
            <span id="savedText"></span>
          </div>
          <div class="bar"><div id="barFill"></div></div>
        </div>
        <button id="nextBtn">下一组</button>
      </div>

      <div class="ab-grid">
        <section class="panel">
          <div class="panel-head">
            <div class="side-title">Left</div>
            <div id="leftMeta" class="small-meta"></div>
          </div>
          <canvas id="leftCanvas" width="900" height="330"></canvas>
          <div class="panel-controls">
            <button id="playLeftBtn">播放左边</button>
            <div id="leftReveal" class="reveal">Hidden</div>
          </div>
        </section>

        <section class="panel">
          <div class="panel-head">
            <div class="side-title">Right</div>
            <div id="rightMeta" class="small-meta"></div>
          </div>
          <canvas id="rightCanvas" width="900" height="330"></canvas>
          <div class="panel-controls">
            <div id="rightReveal" class="reveal">Hidden</div>
            <button id="playRightBtn">播放右边</button>
          </div>
        </section>
      </div>

      <div class="transport">
        <button id="playBothBtn" class="primary">同时播放</button>
        <button id="stopBtn">停止</button>
      </div>

      <div class="choices">
        <button id="chooseLeftBtn" class="choice-left">选左边</button>
        <button id="chooseTieBtn" class="choice-tie">差不多</button>
        <button id="chooseRightBtn" class="choice-right">选右边</button>
        <button id="revealBtn">揭晓</button>
      </div>
      <div id="message" class="message"></div>
    </main>

    <footer id="footerText"></footer>
  </div>

  <script>
    const state = {
      sets: [],
      setId: '',
      pairs: [],
      index: 0,
      pair: null,
      leftNotes: [],
      rightNotes: [],
      choiceLocked: false,
      ratingInFlight: false,
      audioCtx: null,
      activeNodes: [],
      pianoImpulse: null,
      sampledPiano: null,
      sampledPianoPromise: null,
      playbackStart: 0,
      playbackDuration: 0,
      playTimer: null,
      savedCount: 0
    };

    const $ = (id) => document.getElementById(id);

    async function api(path, options) {
      const response = await fetch(path, options);
      if (!response.ok) {
        throw new Error(await response.text());
      }
      return response.json();
    }

    function readStr(view, pos, len) {
      let out = '';
      for (let i = 0; i < len; i++) out += String.fromCharCode(view.getUint8(pos + i));
      return out;
    }

    function readVar(view, cursor) {
      let value = 0;
      while (true) {
        const b = view.getUint8(cursor.pos++);
        value = (value << 7) | (b & 0x7f);
        if ((b & 0x80) === 0) break;
      }
      return value;
    }

    function parseMidi(buffer) {
      const view = new DataView(buffer);
      let pos = 0;
      if (readStr(view, pos, 4) !== 'MThd') throw new Error('Invalid MIDI header');
      pos += 4;
      const headerLen = view.getUint32(pos); pos += 4;
      const format = view.getUint16(pos); pos += 2;
      const trackCount = view.getUint16(pos); pos += 2;
      const ticksPerBeat = view.getUint16(pos); pos += 2;
      pos = 8 + headerLen;
      const notes = [];

      for (let track = 0; track < trackCount; track++) {
        if (readStr(view, pos, 4) !== 'MTrk') break;
        pos += 4;
        const trackLen = view.getUint32(pos); pos += 4;
        const end = pos + trackLen;
        let absSec = 0;
        let tempo = 500000;
        let running = null;
        const active = new Map();
        const cursor = { pos };

        while (cursor.pos < end) {
          const delta = readVar(view, cursor);
          absSec += (delta * tempo) / 1000000 / ticksPerBeat;
          let status = view.getUint8(cursor.pos);
          if (status < 0x80) {
            if (running === null) break;
            status = running;
          } else {
            cursor.pos++;
            if (status < 0xf0) running = status;
          }

          if (status === 0xff) {
            const type = view.getUint8(cursor.pos++);
            const len = readVar(view, cursor);
            if (type === 0x51 && len === 3) {
              tempo = (view.getUint8(cursor.pos) << 16) | (view.getUint8(cursor.pos + 1) << 8) | view.getUint8(cursor.pos + 2);
            }
            cursor.pos += len;
            continue;
          }
          if (status === 0xf0 || status === 0xf7) {
            cursor.pos += readVar(view, cursor);
            continue;
          }

          const eventType = status & 0xf0;
          const channel = status & 0x0f;
          if (eventType === 0xc0 || eventType === 0xd0) {
            cursor.pos += 1;
            continue;
          }
          const d1 = view.getUint8(cursor.pos++);
          const d2 = view.getUint8(cursor.pos++);
          if (eventType === 0x90 && d2 > 0) {
            const key = channel + ':' + d1;
            if (!active.has(key)) active.set(key, []);
            active.get(key).push({ start: absSec, pitch: d1, velocity: d2 });
          } else if (eventType === 0x80 || (eventType === 0x90 && d2 === 0)) {
            const key = channel + ':' + d1;
            const stack = active.get(key);
            if (stack && stack.length) {
              const note = stack.shift();
              notes.push({
                start: note.start,
                duration: Math.max(0.05, absSec - note.start),
                pitch: note.pitch,
                velocity: note.velocity
              });
            }
          }
        }
        pos = end;
      }

      if (notes.length) {
        const minStart = Math.min(...notes.map(n => n.start));
        notes.forEach(n => n.start -= minStart);
      }
      notes.sort((a, b) => a.start - b.start || a.pitch - b.pitch);
      return notes;
    }

    async function loadNotes(fileId) {
      const response = await fetch('/midi/' + fileId);
      if (!response.ok) throw new Error('Cannot load MIDI');
      return parseMidi(await response.arrayBuffer());
    }

    function durationOf(notes) {
      return notes.length ? Math.max(...notes.map(n => n.start + n.duration)) : 1;
    }

    function drawPianoRoll(canvas, notes, side) {
      const ctx = canvas.getContext('2d');
      const w = canvas.width;
      const h = canvas.height;
      ctx.clearRect(0, 0, w, h);
      ctx.fillStyle = '#fbfcfe';
      ctx.fillRect(0, 0, w, h);
      ctx.strokeStyle = '#edf1f6';
      ctx.lineWidth = 1;
      for (let i = 0; i < 8; i++) {
        const y = (h / 8) * i;
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(w, y);
        ctx.stroke();
      }
      if (!notes.length) {
        ctx.fillStyle = '#697386';
        ctx.font = '20px Segoe UI';
        ctx.fillText('No MIDI notes', 28, 56);
        return;
      }
      const minPitch = Math.min(...notes.map(n => n.pitch));
      const maxPitch = Math.max(...notes.map(n => n.pitch));
      const span = Math.max(1, maxPitch - minPitch);
      const dur = durationOf(notes);
      const color = side === 'left' ? '#2a74b8' : '#c76a2a';
      for (const note of notes) {
        const x = 18 + (note.start / dur) * (w - 38);
        const noteW = Math.max(3, (note.duration / dur) * (w - 38));
        const y = 18 + (1 - (note.pitch - minPitch) / span) * (h - 44);
        ctx.fillStyle = color;
        ctx.globalAlpha = 0.82;
        ctx.fillRect(x, y, noteW, 8);
      }
      ctx.globalAlpha = 1;
      ctx.fillStyle = '#697386';
      ctx.font = '13px Segoe UI';
      ctx.fillText(notes.length + ' notes', 18, h - 14);
    }

    function stopPlayback() {
      for (const node of state.activeNodes) {
        try { node.stop(0); } catch (_) {}
      }
      state.activeNodes = [];
      clearInterval(state.playTimer);
      drawPianoRoll($('leftCanvas'), state.leftNotes, 'left');
      drawPianoRoll($('rightCanvas'), state.rightNotes, 'right');
    }

    function choiceButtons() {
      return [$('chooseLeftBtn'), $('chooseTieBtn'), $('chooseRightBtn')];
    }

    function setChoiceLocked(locked) {
      state.choiceLocked = locked;
      for (const button of choiceButtons()) {
        button.disabled = locked;
      }
    }

    function setNavigationLocked(locked) {
      $('nextBtn').disabled = locked;
      $('setSelect').disabled = locked;
      $('modeSelect').disabled = locked;
    }

    function clamp(value, min, max) {
      return Math.max(min, Math.min(max, value));
    }

    function midiToFrequency(pitch) {
      return 440 * Math.pow(2, (pitch - 69) / 12);
    }

    function midiToName(pitch) {
      const names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
      return names[pitch % 12] + (Math.floor(pitch / 12) - 1);
    }

    function pianoImpulse(ctx) {
      if (state.pianoImpulse) return state.pianoImpulse;
      const seconds = 2.6;
      const length = Math.floor(ctx.sampleRate * seconds);
      const buffer = ctx.createBuffer(2, length, ctx.sampleRate);
      for (let channel = 0; channel < 2; channel++) {
        const data = buffer.getChannelData(channel);
        for (let i = 0; i < length; i++) {
          const t = i / length;
          const shimmer = Math.sin(i * 0.019 + channel) * 0.18;
          data[i] = (Math.random() * 2 - 1 + shimmer) * Math.pow(1 - t, 2.8) * 0.42;
        }
      }
      state.pianoImpulse = buffer;
      return buffer;
    }

    function createNoiseBuffer(ctx, seconds) {
      const length = Math.max(1, Math.floor(ctx.sampleRate * seconds));
      const buffer = ctx.createBuffer(1, length, ctx.sampleRate);
      const data = buffer.getChannelData(0);
      for (let i = 0; i < length; i++) {
        const t = i / length;
        data[i] = (Math.random() * 2 - 1) * Math.pow(1 - t, 3.5);
      }
      return buffer;
    }

    function connectSpatialPianoBus(ctx, panValue, gainValue) {
      const dry = ctx.createGain();
      dry.gain.value = gainValue;
      const tone = ctx.createBiquadFilter();
      tone.type = 'lowpass';
      tone.frequency.value = 5400;
      tone.Q.value = 0.55;
      const pan = ctx.createStereoPanner ? ctx.createStereoPanner() : null;
      const compressor = ctx.createDynamicsCompressor();
      compressor.threshold.value = -20;
      compressor.knee.value = 22;
      compressor.ratio.value = 2.6;
      compressor.attack.value = 0.004;
      compressor.release.value = 0.22;
      const convolver = ctx.createConvolver();
      convolver.buffer = pianoImpulse(ctx);
      const wet = ctx.createGain();
      wet.gain.value = 0.18;

      dry.connect(tone);
      if (pan) {
        pan.pan.value = panValue;
        tone.connect(pan);
        pan.connect(compressor);
        tone.connect(convolver);
      } else {
        tone.connect(compressor);
        tone.connect(convolver);
      }
      convolver.connect(wet);
      wet.connect(compressor);
      compressor.connect(ctx.destination);
      return dry;
    }

    function schedulePianoNote(note, output, gainScale) {
      const ctx = state.audioCtx;
      const now = state.playbackStart;
      const freq = midiToFrequency(note.pitch);
      const velocity = clamp(note.velocity / 127, 0.18, 1.0);
      const start = now + note.start;
      const held = Math.max(0.08, note.duration);
      const tail = clamp(1.35 - Math.max(0, note.pitch - 60) * 0.018 + held * 0.18, 0.55, 1.8);
      const stopAt = start + held + tail + 0.12;
      const partials = [
        { ratio: 1.0, type: 'sine', gain: 0.34, detune: -3 },
        { ratio: 2.0, type: 'triangle', gain: 0.095, detune: 4 },
        { ratio: 3.01, type: 'sine', gain: 0.042, detune: -2 },
        { ratio: 4.98, type: 'sine', gain: 0.018, detune: 5 },
      ];

      for (const partial of partials) {
        const osc = ctx.createOscillator();
        const env = ctx.createGain();
        const filter = ctx.createBiquadFilter();
        osc.type = partial.type;
        osc.frequency.setValueAtTime(freq * partial.ratio, start);
        osc.detune.setValueAtTime(partial.detune, start);
        filter.type = 'lowpass';
        filter.frequency.setValueAtTime(clamp(1500 + velocity * 3600 - (note.pitch - 60) * 22, 900, 6500), start);
        filter.Q.value = 0.42;
        const peak = partial.gain * velocity * gainScale;
        env.gain.setValueAtTime(0.0001, start);
        env.gain.exponentialRampToValueAtTime(Math.max(0.0002, peak), start + 0.009);
        env.gain.exponentialRampToValueAtTime(Math.max(0.00018, peak * 0.42), start + 0.12);
        env.gain.exponentialRampToValueAtTime(0.0001, start + held + tail);
        osc.connect(filter).connect(env).connect(output);
        osc.start(start);
        osc.stop(stopAt);
        state.activeNodes.push(osc);
      }

      const hammer = ctx.createBufferSource();
      const hammerGain = ctx.createGain();
      const hammerFilter = ctx.createBiquadFilter();
      hammer.buffer = createNoiseBuffer(ctx, 0.036);
      hammerFilter.type = 'bandpass';
      hammerFilter.frequency.value = clamp(1800 + note.pitch * 24, 2200, 4700);
      hammerFilter.Q.value = 0.85;
      hammerGain.gain.setValueAtTime(0.028 * velocity * gainScale, start);
      hammerGain.gain.exponentialRampToValueAtTime(0.0001, start + 0.035);
      hammer.connect(hammerFilter).connect(hammerGain).connect(output);
      hammer.start(start);
      hammer.stop(start + 0.05);
      state.activeNodes.push(hammer);
    }

    function scheduleNotes(notes, panValue, gainValue) {
      const ctx = state.audioCtx;
      const now = state.playbackStart || ctx.currentTime + 0.10;
      const output = connectSpatialPianoBus(ctx, panValue, gainValue);
      const gainScale = notes.length > 45 ? 0.62 : notes.length > 28 ? 0.74 : 0.88;
      for (const note of notes) {
        schedulePianoNote(note, output, gainScale);
      }
      return now;
    }

    async function ensureSampledPiano() {
      if (state.sampledPiano) return state.sampledPiano;
      if (state.sampledPianoPromise) return state.sampledPianoPromise;
      if (!window.Soundfont) throw new Error('Soundfont library unavailable');
      $('message').textContent = '正在加载采样钢琴音色，第一次可能需要几秒...';
      state.sampledPianoPromise = window.Soundfont.instrument(
        state.audioCtx,
        'acoustic_grand_piano',
        { soundfont: 'FluidR3_GM' }
      ).then((instrument) => {
        state.sampledPiano = instrument;
        $('message').textContent = '';
        return instrument;
      }).catch((error) => {
        state.sampledPianoPromise = null;
        throw error;
      });
      return state.sampledPianoPromise;
    }

    function scheduleSampledNotes(instrument, notes, gainValue) {
      const ctx = state.audioCtx;
      const now = state.playbackStart || ctx.currentTime + 0.10;
      const gainScale = notes.length > 45 ? 0.58 : notes.length > 28 ? 0.70 : 0.84;
      for (const note of notes) {
        const velocity = clamp(note.velocity / 127, 0.20, 1.0);
        const duration = Math.max(0.12, note.duration + 0.42);
        const player = instrument.play(midiToName(note.pitch), now + note.start, {
          duration,
          gain: gainValue * gainScale * velocity
        });
        if (player && typeof player.stop === 'function') {
          state.activeNodes.push(player);
        }
      }
      return now;
    }

    async function play(which) {
      stopPlayback();
      if (!state.audioCtx) state.audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      await state.audioCtx.resume();
      state.playbackStart = state.audioCtx.currentTime + 0.10;
      let sampled = null;
      if ($('soundSelect').value === 'sampled') {
        try {
          sampled = await ensureSampledPiano();
        } catch (error) {
          sampled = null;
          $('message').textContent = '采样钢琴加载失败，已切到本地柔和钢琴。';
          console.warn(error);
        }
      } else {
        $('message').textContent = '';
      }

      if (sampled) {
        if (which === 'left') {
          state.playbackStart = scheduleSampledNotes(sampled, state.leftNotes, 0.92);
          state.playbackDuration = durationOf(state.leftNotes);
        } else if (which === 'right') {
          state.playbackStart = scheduleSampledNotes(sampled, state.rightNotes, 0.92);
          state.playbackDuration = durationOf(state.rightNotes);
        } else {
          state.playbackStart = Math.min(
            scheduleSampledNotes(sampled, state.leftNotes, 0.54),
            scheduleSampledNotes(sampled, state.rightNotes, 0.54)
          );
          state.playbackDuration = Math.max(durationOf(state.leftNotes), durationOf(state.rightNotes));
        }
      } else {
        if (which === 'left') {
          state.playbackStart = scheduleNotes(state.leftNotes, -0.45, 0.9);
          state.playbackDuration = durationOf(state.leftNotes);
        } else if (which === 'right') {
          state.playbackStart = scheduleNotes(state.rightNotes, 0.45, 0.9);
          state.playbackDuration = durationOf(state.rightNotes);
        } else {
          state.playbackStart = Math.min(
            scheduleNotes(state.leftNotes, -0.75, 0.62),
            scheduleNotes(state.rightNotes, 0.75, 0.62)
          );
          state.playbackDuration = Math.max(durationOf(state.leftNotes), durationOf(state.rightNotes));
        }
      }
      animatePlayhead();
    }

    function animatePlayhead() {
      const left = $('leftCanvas');
      const right = $('rightCanvas');
      state.playTimer = setInterval(() => {
        drawPianoRoll(left, state.leftNotes, 'left');
        drawPianoRoll(right, state.rightNotes, 'right');
        const elapsed = state.audioCtx.currentTime - state.playbackStart;
        const xL = 18 + Math.max(0, Math.min(1, elapsed / durationOf(state.leftNotes))) * (left.width - 38);
        const xR = 18 + Math.max(0, Math.min(1, elapsed / durationOf(state.rightNotes))) * (right.width - 38);
        for (const item of [[left, xL], [right, xR]]) {
          const ctx = item[0].getContext('2d');
          ctx.strokeStyle = '#17202a';
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.moveTo(item[1], 0);
          ctx.lineTo(item[1], item[0].height);
          ctx.stroke();
        }
        if (elapsed > state.playbackDuration + 0.3) stopPlayback();
      }, 33);
    }

    function currentFileId(side) {
      const mode = $('modeSelect').value;
      const key = mode === 'response' ? 'response_id' : 'combined_id';
      return state.pair[side][key];
    }

    async function loadPair(index) {
      if (state.ratingInFlight) return;
      stopPlayback();
      state.index = (index + state.pairs.length) % state.pairs.length;
      state.pair = state.pairs[state.index];
      setChoiceLocked(Boolean(state.pair.rated));
      $('message').textContent = state.pair.rated
        ? '本组已记录，不能重复选择。'
        : '';
      $('leftReveal').textContent = 'Hidden';
      $('rightReveal').textContent = 'Hidden';
      $('callText').textContent = state.pair.call_id + ' / trial ' + state.pair.trial;
      $('progressText').textContent = (state.index + 1) + ' / ' + state.pairs.length;
      $('barFill').style.width = ((state.index + 1) / state.pairs.length * 100).toFixed(1) + '%';
      $('leftMeta').textContent = state.pair.left.response_notes + ' response notes';
      $('rightMeta').textContent = state.pair.right.response_notes + ' response notes';
      const [left, right] = await Promise.all([
        loadNotes(currentFileId('left')),
        loadNotes(currentFileId('right'))
      ]);
      state.leftNotes = left;
      state.rightNotes = right;
      drawPianoRoll($('leftCanvas'), left, 'left');
      drawPianoRoll($('rightCanvas'), right, 'right');
    }

    async function reveal() {
      const data = await api('/api/reveal?set_id=' + encodeURIComponent(state.setId) + '&pair_id=' + encodeURIComponent(state.pair.pair_id));
      $('leftReveal').textContent = data.left.variant + ' · ' + data.left.candidate;
      $('rightReveal').textContent = data.right.variant + ' · ' + data.right.candidate;
    }

    async function choose(choice) {
      if (state.choiceLocked || state.ratingInFlight || !state.pair) {
        $('message').textContent = '本组已记录，不能重复选择。';
        return;
      }
      const selectedPair = state.pair;
      const selectedSetId = state.setId;
      setChoiceLocked(true);
      setNavigationLocked(true);
      state.ratingInFlight = true;
      try {
        const result = await api('/api/rate', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            set_id: selectedSetId,
            pair_id: selectedPair.pair_id,
            choice,
            play_mode: $('modeSelect').value + ' / ' + $('soundSelect').value
          })
        });
        state.savedCount = result.saved_count;
        $('savedText').textContent = 'Saved: ' + state.savedCount;
        const ratedPair = state.pairs.find((pair) => pair.pair_id === selectedPair.pair_id);
        if (ratedPair) {
          ratedPair.rated = true;
          ratedPair.rated_choice = choice;
        }
        if (state.pair && state.pair.pair_id === selectedPair.pair_id && state.setId === selectedSetId) {
          $('message').textContent = result.already_rated ? '本组之前已记录，没有重复保存。' : '已记录，本组已锁定。';
          $('leftReveal').textContent = result.reveal.left.variant + ' · ' + result.reveal.left.candidate;
          $('rightReveal').textContent = result.reveal.right.variant + ' · ' + result.reveal.right.candidate;
          setChoiceLocked(true);
        }
      } catch (error) {
        setChoiceLocked(false);
        $('message').textContent = error.message;
      } finally {
        state.ratingInFlight = false;
        setNavigationLocked(false);
      }
    }

    async function loadSet(setId) {
      const data = await api('/api/pairs?set_id=' + encodeURIComponent(setId));
      state.setId = setId;
      state.pairs = data.pairs;
      state.savedCount = data.saved_count;
      $('savedText').textContent = 'Saved: ' + state.savedCount;
      $('footerText').textContent = 'Ratings CSV: ' + data.ratings_csv;
      await loadPair(0);
    }

    async function init() {
      const data = await api('/api/sets');
      state.sets = data.sets;
      const select = $('setSelect');
      select.innerHTML = '';
      for (const item of state.sets) {
        const option = document.createElement('option');
        option.value = item.set_id;
        option.textContent = item.title + ' · ' + item.count + ' pairs';
        select.appendChild(option);
      }
      select.value = 'aria_styled_vs_amt';
      await loadSet(select.value);
    }

    $('setSelect').addEventListener('change', (event) => loadSet(event.target.value));
    $('modeSelect').addEventListener('change', () => loadPair(state.index));
    $('soundSelect').addEventListener('change', stopPlayback);
    $('playLeftBtn').addEventListener('click', () => play('left'));
    $('playRightBtn').addEventListener('click', () => play('right'));
    $('playBothBtn').addEventListener('click', () => play('both'));
    $('stopBtn').addEventListener('click', stopPlayback);
    $('chooseLeftBtn').addEventListener('click', () => choose('left'));
    $('chooseRightBtn').addEventListener('click', () => choose('right'));
    $('chooseTieBtn').addEventListener('click', () => choose('tie'));
    $('revealBtn').addEventListener('click', reveal);
    $('nextBtn').addEventListener('click', () => loadPair(state.index + 1));
    window.addEventListener('keydown', (event) => {
      if (event.target.tagName === 'SELECT') return;
      if (!state.choiceLocked && event.key === '1') choose('left');
      if (!state.choiceLocked && event.key === '2') choose('tie');
      if (!state.choiceLocked && event.key === '3') choose('right');
      if (event.key === ' ') { event.preventDefault(); play('both'); }
      if (!state.ratingInFlight && event.key === 'ArrowRight') loadPair(state.index + 1);
    });
    init().catch((error) => {
      $('message').textContent = error.message;
      console.error(error);
    });
  </script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:
        return

    def send_bytes(self, body: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, payload: Dict[str, Any], status: int = 200) -> None:
        self.send_bytes(json.dumps(payload, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8", status)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.send_bytes(INDEX_HTML.encode("utf-8"), "text/html; charset=utf-8")
            return
        if parsed.path == "/api/sets":
            self.send_json(
                {
                    "sets": [
                        {"set_id": item["set_id"], "title": item["title"], "count": len(item["pairs"])}
                        for item in PAIR_SETS.values()
                    ]
                }
            )
            return
        if parsed.path == "/api/pairs":
            query = parse_qs(parsed.query)
            set_id = query.get("set_id", ["aria_styled_vs_amt"])[0]
            pair_set = PAIR_SETS.get(set_id)
            if not pair_set:
                self.send_json({"error": f"Unknown set_id: {set_id}"}, 404)
                return
            ratings = ratings_by_pair(set_id)
            self.send_json(
                {
                    "set_id": set_id,
                    "title": pair_set["title"],
                    "pairs": [public_pair(pair, ratings.get(pair["pair_id"])) for pair in pair_set["pairs"]],
                    "saved_count": len(ratings),
                    "ratings_csv": str(RATINGS_CSV),
                }
            )
            return
        if parsed.path == "/api/reveal":
            query = parse_qs(parsed.query)
            pair = find_pair(query.get("set_id", [""])[0], query.get("pair_id", [""])[0])
            if not pair:
                self.send_json({"error": "Unknown pair"}, 404)
                return
            self.send_json(reveal_payload(pair))
            return
        if parsed.path.startswith("/midi/"):
            file_id = parsed.path.rsplit("/", 1)[-1]
            path = FILE_REGISTRY.get(file_id)
            if not path or not path.exists():
                self.send_json({"error": "MIDI not found"}, 404)
                return
            self.send_bytes(path.read_bytes(), "audio/midi")
            return
        self.send_json({"error": "Not found"}, 404)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/api/rate":
            self.send_json({"error": "Not found"}, 404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        set_id = payload.get("set_id", "")
        pair = find_pair(set_id, payload.get("pair_id", ""))
        if not pair:
            self.send_json({"error": "Unknown pair"}, 404)
            return
        if payload.get("choice") not in {"left", "right", "tie"}:
            self.send_json({"error": "choice must be left, right, or tie"}, 400)
            return
        if pair["pair_id"] in ratings_by_pair(set_id):
            self.send_json(
                {
                    "ok": True,
                    "already_rated": True,
                    "saved_count": ratings_count(set_id),
                    "reveal": reveal_payload(pair),
                }
            )
            return
        append_rating(payload, pair)
        self.send_json(
            {
                "ok": True,
                "already_rated": False,
                "saved_count": ratings_count(set_id),
                "reveal": reveal_payload(pair),
            }
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the local MIDI blind A/B listening interface")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_pair_sets()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[blind-ab] http://{args.host}:{args.port}")
    print(f"[blind-ab] ratings={RATINGS_CSV}")
    server.serve_forever()


if __name__ == "__main__":
    main()
