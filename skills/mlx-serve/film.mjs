#!/usr/bin/env node
// film.mjs: make a video on a local mlx-serve. A YuE2 song cut by its own sung words, pictures and clips from
// gen.mjs, and a timeline.json that film.tsx renders with Remotion. Node 18+. `node film.mjs help` for usage.
import { spawnSync } from "node:child_process";
import { copyFileSync, existsSync, mkdirSync, readdirSync, readFileSync, realpathSync, writeFileSync } from "node:fs";
import { createRequire } from "node:module";
import { basename, dirname, join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { BASE, encodePng, generate, listModels, pickModel, runFfmpeg, unload } from "./gen.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));
const REMOTION = "4.0.535";

const HELP = `film.mjs: plan, check and render a video made on mlx-serve (server: ${BASE}, env MLX_SERVE_URL). Read film.md.

  node film.mjs init <dir> [--no-install]     project: src/index.tsx (film.tsx), timeline.json, public/, Remotion
  node film.mjs song <dir> --lyrics lyrics.txt --style "<tags>" [--takes 3] [--seed 7] [--max 240] [--model id]
                                              YuE2 sings takes -> transcribed -> scored -> the best becomes public/song.wav;
                                              song.json = the tempo, the beats, and every sung line and word with its
                                              time when the server gives word times
  node film.mjs song <dir> --file mine.wav [--lyrics lyrics.txt]   the same analysis for a song you already have
  node film.mjs words <dir> [--words]         the sung lines (and words) with their song times
  node film.mjs beats <dir> [--from s] [--to s] [--div 2]   beat-grid times to cut on
  node film.mjs check <dir> [--max 140]       validate timeline.json against the files and the song
  node film.mjs stills <dir> <song seconds...>   frames as JPEGs + contact sheets (look at them!)
  node film.mjs render <dir> [--out film.mp4] [--max 140]   render, master to -14 LUFS / -1.5 dBFS peak, report
  node film.mjs ffmpeg <dir>                  print a usable ffmpeg path (for gen.mjs video: FFMPEG=...)

Times in timeline.json are SONG seconds. Pictures, clips, sound effects: node gen.mjs (next to this file).`;

// ---------- server ----------

/** OpenAI-style transcription with word times; null when the server has no transcription endpoint. */
async function transcribe(file, model) {
  const form = new FormData();
  form.append("file", new Blob([readFileSync(file)], { type: "audio/wav" }), basename(file));
  if (model) form.append("model", model);
  form.append("response_format", "verbose_json");
  form.append("timestamp_granularities[]", "word");
  form.append("timestamp_granularities[]", "segment");
  const res = await fetch(BASE + "/v1/audio/transcriptions", { method: "POST", body: form });
  if (res.status === 404) return null;
  if (!res.ok) throw new Error(`/v1/audio/transcriptions: HTTP ${res.status}: ${await res.text()}`);
  return transcriptWords(await res.json());
}

// A segment longer than this cannot place a word.
const SEGMENT_TIMING_MAX = 15;

/** Words from a verbose_json transcript: its word times, else each segment's words spread over the segment. A lone
 * segment (Qwen3-ASR answers one for the whole clip) or a long one leaves its words untimed (start null). */
export function transcriptWords(j) {
  if (Array.isArray(j.words) && j.words.length) return j.words.map((w) => ({ word: String(w.word).trim(), start: w.start, end: w.end }));
  const out = [];
  for (const s of j.segments || []) {
    const ws = String(s.text || "").trim().split(/\s+/).filter(Boolean);
    const step = (s.end - s.start) / ws.length, timed = j.segments.length > 1 && s.end - s.start <= SEGMENT_TIMING_MAX;
    ws.forEach((w, i) => out.push(timed ? { word: w, start: s.start + step * i, end: s.start + step * (i + 1) } : { word: w, start: null, end: null }));
  }
  return out;
}

// ---------- song analysis (pure) ----------

export const normWords = (s) => String(s).toLowerCase().replace(/[’`]/g, "'").match(/[a-z0-9']+/g) || [];

/** Sung lines of a lyrics file with YuE2 section tags ([Verse], [Chorus] ... on their own lines). */
export function parseLyrics(text) {
  const lines = [];
  let section = "";
  for (const raw of text.split(/\r?\n/)) {
    const l = raw.trim();
    if (!l) continue;
    const tag = /^\[([^\]]+)\]$/.exec(l);
    if (tag) { section = tag[1]; continue; }
    const words = normWords(l);
    if (words.length) lines.push({ text: l, section, words });
  }
  return lines;
}

/** Times for every lyric line from transcribed words, plus how well the take sang the lyrics. The alignment keeps the
 * most exact matches (a substitution costs a deletion plus an insertion), so a stray word never moves a line's start. */
export function alignLyrics(lines, heard) {
  const L = [];
  lines.forEach((ln, li) => ln.words.forEach((w) => L.push({ w, li })));
  const Hw = heard.map((h) => ({ ...h, w: normWords(h.word)[0] || "" }));
  const n = L.length, m = Hw.length;
  const D = Array.from({ length: n + 1 }, () => new Int32Array(m + 1));
  for (let i = 0; i <= n; i++) D[i][0] = i;
  for (let j = 0; j <= m; j++) D[0][j] = j;
  for (let i = 1; i <= n; i++) {
    for (let j = 1; j <= m; j++) {
      D[i][j] = Math.min(D[i - 1][j - 1] + (L[i - 1].w === Hw[j - 1].w ? 0 : 2), D[i - 1][j] + 1, D[i][j - 1] + 1);
    }
  }
  const times = lines.map(() => ({ start: Infinity, end: -Infinity, hits: 0 }));
  let exact = 0;
  for (let i = n, j = m; i > 0 && j > 0;) {
    const same = L[i - 1].w === Hw[j - 1].w;
    if (D[i][j] === D[i - 1][j - 1] + (same ? 0 : 2)) {
      const t = times[L[i - 1].li];
      t.start = Math.min(t.start, Hw[j - 1].start);
      t.end = Math.max(t.end, Hw[j - 1].end);
      if (same) { t.hits++; exact++; }
      i--; j--;
    } else if (D[i][j] === D[i - 1][j] + 1) i--;
    else j--;
  }
  // a line nothing matched sits between its neighbours
  for (let k = 0; k < times.length; k++) {
    if (Number.isFinite(times[k].start)) continue;
    const prev = times.slice(0, k).reverse().find((t) => Number.isFinite(t.end))?.end ?? 0;
    const next = times.slice(k + 1).find((t) => Number.isFinite(t.start))?.start ?? prev + 3;
    times[k].start = prev + 0.05;
    times[k].end = Math.max(prev + 0.1, next - 0.05);
  }
  return {
    similarity: Math.round(((2 * exact) / Math.max(1, n + m)) * 1000) / 1000,
    lines: lines.map((ln, k) => ({ text: ln.text, section: ln.section, start: round2(times[k].start), end: round2(times[k].end), sung: round2(times[k].hits / ln.words.length) })),
  };
}

/** Mono samples of a PCM16/float32 WAV. */
export function readWav(buf) {
  if (buf.toString("ascii", 0, 4) !== "RIFF" || buf.toString("ascii", 8, 12) !== "WAVE") throw new Error("not a WAV file");
  let fmt = null, data = null;
  for (let off = 12; off + 8 <= buf.length;) {
    const id = buf.toString("ascii", off, off + 4), size = buf.readUInt32LE(off + 4);
    if (id === "fmt ") fmt = { format: buf.readUInt16LE(off + 8), channels: buf.readUInt16LE(off + 10), rate: buf.readUInt32LE(off + 12), bits: buf.readUInt16LE(off + 22) };
    if (id === "data") data = buf.subarray(off + 8, Math.min(buf.length, off + 8 + size));
    off += 8 + size + (size & 1);
  }
  if (!fmt || !data) throw new Error("WAV without fmt/data chunks");
  const bytes = fmt.bits / 8, frames = Math.floor(data.length / (bytes * fmt.channels));
  const mono = new Float32Array(frames);
  for (let i = 0; i < frames; i++) {
    let s = 0;
    for (let c = 0; c < fmt.channels; c++) {
      const o = (i * fmt.channels + c) * bytes;
      s += fmt.format === 3 ? data.readFloatLE(o) : bytes === 2 ? data.readInt16LE(o) / 32768 : bytes === 3 ? data.readIntLE(o, 3) / 8388608 : data.readInt32LE(o) / 2147483648;
    }
    mono[i] = s / fmt.channels;
  }
  return { rate: fmt.rate, samples: mono, duration: frames / fmt.rate };
}

/** Loudness per window in dB below the loudest window. */
export function envelopeDb(wav, win = 0.5) {
  const n = Math.max(1, Math.floor(wav.rate * win)), rms = [];
  for (let i = 0; i + n <= wav.samples.length; i += n) {
    let s = 0;
    for (let k = i; k < i + n; k++) s += wav.samples[k] * wav.samples[k];
    rms.push(Math.sqrt(s / n));
  }
  const peak = Math.max(1e-9, ...rms);
  return rms.map((r) => round2(20 * Math.log10(Math.max(r, 1e-9) / peak)));
}

/** How far quiet parts sit below loud parts (p90 - p10 of the envelope): verses vs choruses. */
export function dynamics(env) {
  const s = [...env].sort((a, b) => a - b), q = (p) => s[Math.min(s.length - 1, Math.floor(p * s.length))];
  return round2(q(0.9) - q(0.1));
}

/** Quarter-note tempo from an ABC score's Q: line (Q:1/4=73, Q:1/8=120, Q:90). */
export function abcTempo(abc) {
  const m = /^Q:\s*(\d+)\/(\d+)\s*=\s*(\d+)/m.exec(abc || "");
  if (m) return (+m[3] * +m[1] * 4) / +m[2];
  const b = /^Q:\s*(\d+)\s*$/m.exec(abc || "");
  return b ? +b[1] : null;
}

/** Onset strength about 100 times a second: the rise in log energy of three bands (bass, mid, treble), a cheap
 * spectral flux. `fps` is exact (rate / whole-sample hop): 22.05 kHz hops 221 samples, not 220.5. */
export function onsetEnvelope(wav) {
  const hop = Math.round(wav.rate / 100), win = 2 * hop, x = wav.samples;
  const lo = new Float32Array(x.length), mid = new Float32Array(x.length), hi = new Float32Array(x.length);
  const a1 = Math.exp((-2 * Math.PI * 200) / wav.rate), a2 = Math.exp((-2 * Math.PI * 2000) / wav.rate);
  for (let i = 0, y1 = 0, y2 = 0; i < x.length; i++) {
    y1 = (1 - a1) * x[i] + a1 * y1;
    y2 = (1 - a2) * x[i] + a2 * y2;
    lo[i] = y1; mid[i] = y2 - y1; hi[i] = x[i] - y2;
  }
  const n = Math.max(0, Math.floor((x.length - win) / hop)), env = new Float32Array(n);
  for (const band of [lo, mid, hi]) {
    let prev = null;
    for (let i = 0; i < n; i++) {
      let e = 0;
      for (let k = i * hop; k < i * hop + win; k++) e += band[k] * band[k];
      const db = 10 * Math.log10(e / win + 1e-10);
      if (prev !== null) env[i] += Math.max(0, db - prev);
      prev = db;
    }
  }
  return { fps: wav.rate / hop, strength: env };
}

/** Tempo (bpm) from the onset envelope's autocorrelation, held near `hint` (YuE2's score tempo) when given. */
export function estimateTempo({ fps, strength: env }, hint = null) {
  const mean = env.reduce((a, v) => a + v, 0) / Math.max(1, env.length), z = env.map((v) => v - mean);
  let best = -Infinity, bpm = hint || 120;
  for (let b = 50; b <= 200; b += 0.1) {
    const lag = (60 / b) * fps, l0 = Math.floor(lag), f = lag - l0;
    let ac = 0;
    for (let i = 0; i + l0 + 1 < z.length; i++) ac += z[i] * (z[i + l0] * (1 - f) + z[i + l0 + 1] * f);
    const w = ac * Math.exp(-0.5 * (Math.log2(b / (hint || 120)) / (hint ? 0.15 : 1)) ** 2);
    if (w > best) { best = w; bpm = b; }
  }
  return Math.round(bpm * 10) / 10;
}

/** Beat times in seconds: dynamic programming over the onset envelope (Ellis 2007, as librosa), so the grid follows
 * the song's own drift; weak beats over a silent intro or tail are dropped. */
export function trackBeats({ fps, strength: env }, bpm, tightness = 100) {
  const period = (60 / bpm) * fps, sig = period / 32, half = Math.ceil(3 * sig);
  const sd = Math.sqrt(env.reduce((a, v) => a + v * v, 0) / Math.max(1, env.length)) || 1;
  const local = new Float32Array(env.length);
  for (let i = 0; i < env.length; i++) {
    let s = 0;
    for (let k = -half; k <= half; k++) if (i + k >= 0 && i + k < env.length) s += Math.exp(-0.5 * (k / sig) ** 2) * env[i + k];
    local[i] = s / sd;
  }
  const score = new Float32Array(env.length), back = new Int32Array(env.length).fill(-1);
  for (let t = 0; t < env.length; t++) {
    let best = -Infinity, arg = -1;
    for (let s = Math.max(0, Math.round(t - 2 * period)); s <= t - Math.round(period / 2); s++) {
      const v = score[s] - tightness * Math.log((t - s) / period) ** 2;
      if (v > best) { best = v; arg = s; }
    }
    score[t] = local[t] + (best > 0 ? best : 0);
    if (best > 0) back[t] = arg;
  }
  let end = env.length - 1;
  for (let t = Math.max(0, env.length - Math.round(period)); t < env.length; t++) if (score[t] > score[end]) end = t;
  const path = [];
  for (let t = end; t >= 0; t = back[t]) path.push(t);
  path.reverse();
  const floor = Math.sqrt(path.reduce((a, t) => a + local[t] ** 2, 0) / Math.max(1, path.length)) / 2;
  let a = 0, b = path.length;
  while (a < b && local[path[a]] < floor) a++;
  while (b > a && local[path[b - 1]] < floor) b--;
  // frame i's flux is the energy entering at (i + 1.5) hops
  return path.slice(a, b).map((t) => round2((t + 1.5) / fps));
}

/** Beats in [from, to], each beat split into `div` equal steps (div 2 = eighth notes in 4/4). */
export function beatsBetween(beats, from, to, div = 1) {
  const out = [];
  beats.forEach((b, k) => {
    for (let j = 0; j < div && (j === 0 || k + 1 < beats.length); j++) {
      const t = round2(j ? b + ((beats[k + 1] - b) * j) / div : b);
      if (t >= from - 1e-9 && t <= to + 1e-9) out.push(t);
    }
  });
  return out;
}

/** Higher is better: sung the lyrics, kept dynamics, and ended on its own before the duration bound. */
export function scoreTake({ similarity, duration, bound, spread }) {
  const hitBound = bound != null && duration >= bound - 1.5;
  return round2((similarity ?? 0) + 0.1 * Math.min(1, Math.max(0, (spread ?? 0) / 20)) - (hitBound ? 1 : 0));
}

const round2 = (x) => Math.round(x * 100) / 100;

// ---------- timeline ----------

const RANGED = new Set(["black", "backdrop", "still", "clip", "screen", "wall", "grid", "caption", "title", "words", "card", "terminal", "credits", "brand", "endcard", "letterbox", "lyrics"]);
const NEEDS = { still: ["src"], clip: ["src"], screen: ["src"], wall: ["items"], grid: ["items"], caption: ["text"], title: ["text"], words: ["words"], card: ["name"], credits: ["sections"], sfx: ["src", "at"], flash: ["at"], terminal: ["command"] };

/** Problems with a timeline before anything renders: unknown blocks, missing fields or files, times off the song. */
export function validateTimeline(t, { exists = () => true, songDuration = null, maxSeconds = 140, lines = null } = {}) {
  const errors = [], warnings = [];
  const song = t.song || {};
  if (!(song.end > (song.start ?? 0))) errors.push("song.end must be after song.start (both in song seconds)");
  if (song.file && !exists(song.file)) errors.push(`song.file ${song.file} is not in public/`);
  if (!song.file) warnings.push("no song.file: the video will be silent");
  if (songDuration && song.end > songDuration + 0.05) errors.push(`song.end ${song.end} is past the end of the song (${songDuration} s)`);
  const duration = round2((song.end ?? 0) - (song.start ?? 0));
  if (maxSeconds && duration > maxSeconds) warnings.push(`video is ${duration} s, over ${maxSeconds} s (X allows 140 s for most accounts)`);
  (t.blocks || []).forEach((b, i) => {
    const at = `blocks[${i}] (${b.type})`;
    if (!RANGED.has(b.type) && b.type !== "sfx" && b.type !== "flash") return errors.push(`${at}: unknown type; use ${[...RANGED, "sfx", "flash"].join(", ")}`);
    for (const k of NEEDS[b.type] || []) if (b[k] == null || (Array.isArray(b[k]) && !b[k].length)) errors.push(`${at}: needs "${k}"`);
    if (RANGED.has(b.type)) {
      if (!(typeof b.from === "number" && typeof b.to === "number")) errors.push(`${at}: needs numeric "from" and "to" (song seconds)`);
      else if (b.to <= b.from) errors.push(`${at}: "to" (${b.to}) must be after "from" (${b.from})`);
      else if (b.to < (song.start ?? 0) || b.from > song.end) warnings.push(`${at}: ${b.from}-${b.to} is outside the video (${song.start}-${song.end})`);
    }
    const files = [b.src, ...(b.items || []), b.icon, b.wordmark, typeof b.bg === "string" ? b.bg : b.bg?.src].filter((f) => typeof f === "string");
    for (const f of files) if (!exists(f)) errors.push(`${at}: ${f} is not in public/`);
    if (b.type === "lyrics" && lines && !lines.length) warnings.push(`${at}: song.json has no line times (the server gave no word times), so this block shows nothing`);
  });
  return { errors, warnings, duration };
}

// ---------- project ----------

const STARTER = {
  fps: 30,
  width: 1920,
  height: 1080,
  song: { file: null, start: 0, end: 8, fadeOut: 2 },
  blocks: [
    { type: "brand", from: 0, to: 2.5, title: "My Film", version: "made on mlx-serve" },
    { type: "title", from: 2.5, to: 5, text: "Your Mac is a studio", sub: "Edit timeline.json, then node film.mjs render ." },
    { type: "endcard", from: 5, to: 8, title: "My Film", line: "Every frame and every sound was made on one Mac." },
  ],
};

const FONT_FILES = [
  ["@fontsource/inter/files/inter-latin-400-normal.woff2", "inter-400.woff2"],
  ["@fontsource/inter/files/inter-latin-600-normal.woff2", "inter-600.woff2"],
  ["@fontsource/inter/files/inter-latin-700-normal.woff2", "inter-700.woff2"],
  ["@fontsource/inter/files/inter-latin-800-normal.woff2", "inter-800.woff2"],
  ["@fontsource/inter/files/inter-latin-900-normal.woff2", "inter-900.woff2"],
  ["@fontsource/instrument-serif/files/instrument-serif-latin-400-normal.woff2", "instrument-serif-400.woff2"],
  ["@fontsource/instrument-serif/files/instrument-serif-latin-400-italic.woff2", "instrument-serif-400-italic.woff2"],
  ["@fontsource/jetbrains-mono/files/jetbrains-mono-latin-400-normal.woff2", "jetbrains-mono-400.woff2"],
];

function init(dir, { install = true } = {}) {
  mkdirSync(join(dir, "src"), { recursive: true });
  mkdirSync(join(dir, "public", "_film"), { recursive: true });
  const pkg = {
    name: basename(resolve(dir)).toLowerCase().replace(/[^a-z0-9-]/g, "-") || "film",
    private: true,
    dependencies: {
      remotion: REMOTION, "@remotion/bundler": REMOTION, "@remotion/renderer": REMOTION, "@remotion/cli": REMOTION, "@remotion/fonts": REMOTION,
      react: "^19.0.0", "react-dom": "^19.0.0", "@fontsource/inter": "^5.0.0", "@fontsource/instrument-serif": "^5.0.0", "@fontsource/jetbrains-mono": "^5.0.0",
    },
  };
  if (!existsSync(join(dir, "package.json"))) writeFileSync(join(dir, "package.json"), JSON.stringify(pkg, null, 2) + "\n");
  if (!existsSync(join(dir, "src", "index.tsx"))) copyFileSync(join(HERE, "film.tsx"), join(dir, "src", "index.tsx"));
  if (!existsSync(join(dir, "tsconfig.json"))) writeFileSync(join(dir, "tsconfig.json"), JSON.stringify({ compilerOptions: { jsx: "react-jsx", module: "esnext", moduleResolution: "bundler", resolveJsonModule: true, esModuleInterop: true, strict: false, skipLibCheck: true } }, null, 2) + "\n");
  if (!existsSync(join(dir, "timeline.json"))) writeFileSync(join(dir, "timeline.json"), JSON.stringify(STARTER, null, 2) + "\n");
  if (!existsSync(join(dir, "song.json"))) writeFileSync(join(dir, "song.json"), JSON.stringify({ lines: [], words: [] }) + "\n");
  writeFileSync(join(dir, "public", "_film", "noise.png"), noisePng(512));
  if (install) {
    console.error("installing Remotion (first time: a few minutes)...");
    const r = spawnSync("npm", ["install", "--no-fund", "--no-audit"], { cwd: dir, stdio: "inherit" });
    if (r.status !== 0) throw new Error("npm install failed; fix it and run init again");
  }
  let copied = 0;
  for (const [from, to] of FONT_FILES) {
    const src = join(dir, "node_modules", from);
    if (existsSync(src)) { copyFileSync(src, join(dir, "public", "_film", to)); copied++; }
  }
  if (install && copied < FONT_FILES.length) console.error(`warning: only ${copied}/${FONT_FILES.length} font files found under node_modules/@fontsource`);
  console.log(`ready: ${resolve(dir)}\n  1. put pictures/clips/sounds under public/ (gen.mjs) and the song with: node film.mjs song ${dir} ...\n  2. edit timeline.json (song seconds), then node film.mjs check ${dir}\n  3. node film.mjs stills ${dir} <seconds...> to look, node film.mjs render ${dir}`);
}

/** A grayscale noise PNG for the film grain (deterministic). */
function noisePng(size) {
  const data = Buffer.alloc(size * size * 4);
  for (let i = 0, seed = 0x9e3779b9; i < size * size; i++) {
    seed = (seed * 1664525 + 1013904223) >>> 0;
    data[i * 4] = data[i * 4 + 1] = data[i * 4 + 2] = seed >>> 24;
    data[i * 4 + 3] = 255;
  }
  return encodePng({ w: size, h: size, data });
}

const readJson = (f, d = null) => (existsSync(f) ? JSON.parse(readFileSync(f, "utf8")) : d);

async function song(dir, opt) {
  if (!opt.file && !(opt.lyrics && opt.style)) {
    throw new Error('song needs --lyrics <file> --style "<tags>" (YuE2 sings it), or --file <wav> [--lyrics <file>] for a song you have');
  }
  const lyricsText = opt.lyrics ? readFileSync(opt.lyrics, "utf8") : "";
  const lines = parseLyrics(lyricsText);
  if (opt.lyrics && !lines.length) throw new Error(`${opt.lyrics} has no sung lines`);
  const pub = join(dir, "public"), tdir = join(pub, "takes"), bound = Number(opt.max || 240);
  mkdirSync(tdir, { recursive: true });
  let takes;
  if (opt.file) {
    const file = join(tdir, basename(opt.file));
    if (resolve(opt.file) !== resolve(file)) copyFileSync(opt.file, file);
    takes = [{ file, seed: null }];
  } else {
    const model = await pickModel("music", opt.model, /yue/i);
    const n = Math.max(1, Number(opt.takes || 3)), seed0 = Number(opt.seed || 7);
    takes = Array.from({ length: n }, (_, k) => ({ seed: seed0 + k * 101, file: join(tdir, `take-${seed0 + k * 101}.wav`) }));
    let made = 0;
    for (const t of takes.filter((t) => !existsSync(t.file))) {
      console.error(`take seed ${t.seed} (${model})`);
      const ev = await generate("music", { model, prompt: opt.style, lyrics: lyricsText, seed: t.seed, duration_seconds: bound });
      writeFileSync(t.file, Buffer.from(ev.data, "base64"));
      if (ev.abc) writeFileSync(t.file.replace(/\.wav$/, ".abc"), ev.abc);
      made++;
    }
    if (made) await unload(model);
  }
  const asr = (await listModels()).find((m) => (m.capabilities || []).includes("transcription"))?.id;
  const results = [];
  let heardAny = false;
  for (const t of takes) {
    const wav = readWav(readFileSync(t.file)), env = envelopeDb(wav), abc = t.file.replace(/\.wav$/, ".abc");
    const heard = heardAny || !results.length ? await transcribe(t.file, asr) : null;
    if (heard === null && !results.length) console.error("warning: this server has no /v1/audio/transcriptions: no word times; takes are scored by sound only");
    heardAny ||= heard !== null;
    // without word times the transcript still says which take sang the lyrics, but not when
    const timed = heard?.length > 0 && heard.every((w) => w.start != null);
    if (heard?.length && !timed && !results.length) console.error("warning: the transcription has no word times (one segment for the whole song): takes are scored on the lyrics, cut on beats. The server times words once mlx-community/Qwen3-ForcedAligner-0.6B-8bit is on disk beside the ASR");
    const aligned = heard && lines.length ? alignLyrics(lines, heard) : { similarity: null, lines: [] };
    const r = {
      file: `takes/${basename(t.file)}`, seed: t.seed, duration: round2(wav.duration), similarity: aligned.similarity, spread: dynamics(env),
      scoreBpm: existsSync(abc) ? abcTempo(readFileSync(abc, "utf8")) : null, lines: timed ? aligned.lines : [],
      words: timed ? heard.map((w) => ({ word: w.word, start: round2(w.start), end: round2(w.end) })) : [], env, wav,
    };
    r.score = scoreTake({ ...r, bound: opt.file ? null : bound });
    results.push(r);
    console.error(`  ${r.file}: ${r.duration} s, lyrics ${r.similarity ?? "?"}, dynamics ${r.spread} dB, score ${r.score}`);
  }
  if (heardAny && asr) await unload(asr);
  const best = [...results].sort((a, b) => b.score - a.score)[0];
  copyFileSync(join(pub, best.file), join(pub, "song.wav"));
  const onset = onsetEnvelope(best.wav), bpm = estimateTempo(onset, best.scoreBpm), beats = trackBeats(onset, bpm);
  const first = best.lines[0]?.start ?? best.words[0]?.start ?? null;
  // the song's own fade: the last half-second still within 20 dB of the peak
  const loudUntil = (best.env.length - 1 - [...best.env].reverse().findIndex((v) => v > -20)) * 0.5;
  writeFileSync(join(dir, "song.json"), JSON.stringify({
    file: "song.wav", from: best.file, duration: best.duration, bpm, firstVocal: first, loudUntil,
    lines: best.lines, words: best.words, beats, takes: results.map(({ lines, words, env, wav, ...r }) => r),
  }, null, 1) + "\n");
  const start = first != null ? Math.max(0, round2(first - 9)) : 0, end = Math.min(best.duration, round2(loudUntil + 8), round2(start + 139.9));
  console.log(`song.wav <- ${best.file} (score ${best.score}): ${best.duration} s, ${bpm} bpm, first vocal ${first ?? "?"} s, loud until ${loudUntil} s`);
  console.log(`timeline.json "song": {"file": "song.wav", "start": ${start}, "end": ${end}}; plan cuts with node film.mjs words ${dir} and beats ${dir}`);
}

function words(dir, opt) {
  const s = readJson(join(dir, "song.json"));
  if (!s) throw new Error("no song.json: run node film.mjs song first");
  if (!s.lines?.length && !s.words?.length) throw new Error(`song.json has no line times (the server gave no word times): cut on beats, node film.mjs beats ${dir}`);
  console.log(`${s.file} ${s.duration} s, ${s.bpm ?? "?"} bpm, first vocal ${s.firstVocal} s`);
  if (!s.lines?.length) return console.log(s.words.map((w) => `${w.word}@${w.start}`).join(" "));
  for (const l of s.lines) {
    console.log(`${l.start.toFixed(2).padStart(7)}-${l.end.toFixed(2).padEnd(7)} ${(l.section ? "[" + l.section + "]" : "").padEnd(14)} ${l.text}`);
    if (opt.words) console.log("          " + s.words.filter((w) => w.start >= l.start - 0.05 && w.start < l.end + 0.05).map((w) => `${w.word}@${w.start}`).join(" "));
  }
}

function beats(dir, opt) {
  const s = readJson(join(dir, "song.json"));
  if (!s?.beats?.length) throw new Error("song.json has no beats: run node film.mjs song first");
  console.log(beatsBetween(s.beats, Number(opt.from ?? 0), Number(opt.to ?? s.duration), Number(opt.div || 1)).join(" "));
}

function check(dir, opt, quiet = false) {
  const t = readJson(join(dir, "timeline.json"));
  if (!t) throw new Error(`${dir}/timeline.json not found: node film.mjs init ${dir}`);
  const s = readJson(join(dir, "song.json"), {});
  const r = validateTimeline(t, { exists: (f) => existsSync(join(dir, "public", f)), songDuration: s.duration ?? null, maxSeconds: Number(opt.max ?? 140), lines: s.lines ?? null });
  for (const w of r.warnings) console.error(`warning: ${w}`);
  for (const e of r.errors) console.error(`error: ${e}`);
  if (!quiet || r.errors.length) console.error(`${r.errors.length} errors, ${r.warnings.length} warnings; video ${r.duration} s`);
  if (r.errors.length) throw new Error("fix timeline.json first");
  return r;
}

// ---------- rendering ----------

async function remotion(dir) {
  const req = createRequire(join(resolve(dir), "package.json"));
  let bundler, renderer;
  try {
    bundler = await import(pathToFileURL(req.resolve("@remotion/bundler")).href);
    renderer = await import(pathToFileURL(req.resolve("@remotion/renderer")).href);
  } catch {
    throw new Error(`Remotion is not installed in ${dir}: node film.mjs init ${dir}`);
  }
  console.error("bundling...");
  const serveUrl = await bundler.bundle({ entryPoint: join(resolve(dir), "src", "index.tsx"), publicDir: join(resolve(dir), "public") });
  const composition = await renderer.selectComposition({ serveUrl, id: "Film" });
  return { renderer, serveUrl, composition };
}

/** The first ffmpeg (FFMPEG, then PATH, then Remotion's bundled one) that runs and has every filter in `need`:
 * Remotion's has no limiter, loudness meter or xstack. */
function ffmpegBin(dir, need = []) {
  const scope = join(dir, "node_modules", "@remotion");
  const bundled = existsSync(scope) ? readdirSync(scope).filter((d) => d.startsWith("compositor-")).map((d) => join(resolve(scope), d, process.platform === "win32" ? "ffmpeg.exe" : "ffmpeg")) : [];
  for (const bin of [process.env.FFMPEG, "ffmpeg", ...bundled].filter(Boolean)) {
    const r = runFfmpeg(bin, ["-hide_banner", "-filters"]);
    if (r.error || r.status !== 0) continue;
    if (need.every((f) => new RegExp("\\s" + f + "\\s").test(r.stdout))) return bin;
  }
  return null;
}

async function stills(dir, secs) {
  const t = readJson(join(dir, "timeline.json"));
  if (!secs.length) throw new Error("give song seconds: node film.mjs stills <dir> 12 30.5 42");
  check(dir, {}, true);
  const { renderer, serveUrl, composition } = await remotion(dir);
  const out = join(dir, "stills");
  mkdirSync(out, { recursive: true });
  const files = [];
  for (const s of secs.map(Number)) {
    const frame = Math.min(composition.durationInFrames - 1, Math.max(0, Math.round((s - (t.song?.start || 0)) * composition.fps)));
    const file = join(out, `t${s.toFixed(2)}.jpg`);
    await renderer.renderStill({ composition, serveUrl, frame, output: file, imageFormat: "jpeg", jpegQuality: 85, scale: 0.5 });
    files.push(file);
  }
  const ff = ffmpegBin(dir, ["xstack"]);
  if (!ff) return console.log(files.join("\n") + "\n(no contact sheets: install ffmpeg)");
  for (let k = 0; k < files.length; k += 9) {
    const group = files.slice(k, k + 9), sheet = join(out, `sheet-${k / 9 + 1}.jpg`);
    const lay = ["0_0", "w0_0", "w0+w1_0", "0_h0", "w0_h0", "w0+w1_h0", "0_h0+h1", "w0_h0+h1", "w0+w1_h0+h1"].slice(0, group.length).join("|");
    const args = ["-v", "error", "-y", ...group.flatMap((f) => ["-i", f])];
    args.push(...(group.length > 1 ? ["-filter_complex", `xstack=inputs=${group.length}:layout=${lay}:fill=black`] : []), sheet);
    runFfmpeg(ff, args);
    console.log(sheet + "   (" + group.map((f) => basename(f, ".jpg").slice(1)).join(", ") + " s)");
  }
}

/** Integrated loudness (LUFS) and true peak (dBFS) of a file's audio. */
function loudness(ff, file) {
  const out = runFfmpeg(ff, ["-hide_banner", "-nostats", "-i", file, "-vn", "-af", "ebur128=peak=true:framelog=quiet", "-f", "null", "-"]).stderr || "";
  const num = (re) => { const m = re.exec(out); return m ? Number(m[1]) : null; };
  return { lufs: num(/I:\s+(-?[\d.]+) LUFS/), peak: num(/Peak:\s+(-?[\d.]+) dBFS/) };
}

async function render(dir, opt) {
  const r = check(dir, opt);
  const { renderer, serveUrl, composition } = await remotion(dir);
  const outDir = join(dir, "out");
  mkdirSync(outDir, { recursive: true });
  const raw = join(outDir, "raw.mp4"), final = join(outDir, opt.out || "film.mp4");
  let last = -1;
  await renderer.renderMedia({
    composition, serveUrl, codec: "h264", outputLocation: raw, crf: 16, audioBitrate: "320k",
    onProgress: ({ progress }) => { const p = Math.floor(progress * 20); if (p !== last) { last = p; process.stderr.write(`\r  rendering ${Math.round(progress * 100)}%   `); } },
  });
  process.stderr.write("\n");
  const ff = ffmpegBin(dir, ["alimiter", "ebur128"]);
  if (!ff) { copyFileSync(raw, final); return console.log(`${final} (not mastered: install ffmpeg for the limiter and loudness check)`); }
  const max = Number(opt.max ?? 140);
  // streaming platforms play at -14 LUFS: bring the mix there, then an oversampled limiter keeps peaks under -1.5 dBFS
  const gain = Math.max(-20, Math.min(12, -14 - (loudness(ff, raw).lufs ?? -14)));
  const args = ["-v", "error", "-y", "-i", raw, "-c:v", "libx264", "-preset", "slow", "-crf", "17", "-profile:v", "high", "-pix_fmt", "yuv420p", "-maxrate", "20M", "-bufsize", "40M", "-g", "60",
    "-af", `volume=${gain.toFixed(2)}dB,aresample=192000,alimiter=limit=0.8414:attack=5:release=50:level=false,aresample=48000`, "-c:a", "aac", "-b:a", "320k", "-ar", "48000"];
  if (max && r.duration > max) args.push("-t", String(max - 0.1));
  const m = runFfmpeg(ff, [...args, "-movflags", "+faststart", final]);
  if (m.status !== 0) { copyFileSync(raw, final); return console.log(`${final} (not mastered: ${m.stderr.trim().split("\n").pop()})`); }
  const { lufs, peak } = loudness(ff, final);
  console.log(`${final}: ${Math.min(r.duration, max || Infinity)} s, ${lufs ?? "?"} LUFS, peak ${peak ?? "?"} dBFS`);
}

// ---------- CLI ----------

function parseArgs(argv) {
  const pos = [], opt = {};
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === "--no-install" || a === "--words") opt[a.slice(2)] = true;
    else if (a.startsWith("--")) opt[a.slice(2)] = argv[++i];
    else pos.push(a);
  }
  return { pos, opt };
}

async function main() {
  const [cmd, dir, ...rest] = process.argv.slice(2);
  const { pos, opt } = parseArgs(rest);
  if (cmd && cmd !== "help" && !dir) throw new Error(`missing <dir>; node film.mjs help`);
  switch (cmd) {
    case "init": return init(dir, { install: !opt["no-install"] });
    case "song": return song(dir, opt);
    case "words": return words(dir, opt);
    case "beats": return beats(dir, opt);
    case "check": return check(dir, opt);
    case "stills": return stills(dir, pos);
    case "render": return render(dir, opt);
    case "ffmpeg": { const ff = ffmpegBin(dir); if (!ff) throw new Error("no ffmpeg found: install one or set FFMPEG"); return console.log(ff); }
    default: console.log(HELP);
  }
}

if (process.argv[1] && realpathSync(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main().catch((e) => { console.error(`error: ${e.message}`); process.exitCode = 1; });
}
