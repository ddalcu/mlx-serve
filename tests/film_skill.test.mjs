// Tests for the film skill (skills/mlx-serve/film.mjs, gen.mjs video jobs): the song analysis that times
// every cut, the timeline checks, and the CLIs against a mock mlx-serve. No GPU, no real server.
//
//   node --test tests/film_skill.test.mjs

import { test } from "node:test";
import assert from "node:assert/strict";
import { execFile } from "node:child_process";
import { chmodSync, existsSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { createServer } from "node:http";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { promisify } from "node:util";

import { decodePng, encodePng, fixGlb, parseGlb, readAccessor, writeGlb } from "../skills/mlx-serve/gen.mjs";

import {
  abcTempo, alignLyrics, beatsBetween, dynamics, envelopeDb, estimateTempo, onsetEnvelope, parseLyrics,
  readWav, scoreTake, trackBeats, transcriptWords, validateTimeline,
} from "../skills/mlx-serve/film.mjs";

const SKILL = new URL("../skills/mlx-serve/", import.meta.url).pathname;
const run = promisify(execFile);
const tmp = () => mkdtempSync(join(tmpdir(), "film-skill-"));

// ---------- synthetic audio ----------

function wavBytes(samples, rate, { float = false, channels = 1 } = {}) {
  const bytes = float ? 4 : 2, data = Buffer.alloc(samples.length * bytes * channels);
  samples.forEach((v, i) => {
    for (let c = 0; c < channels; c++) {
      const o = (i * channels + c) * bytes;
      if (float) data.writeFloatLE(v, o);
      else data.writeInt16LE(Math.max(-32768, Math.min(32767, Math.round(v * 32767))), o);
    }
  });
  const h = Buffer.alloc(44);
  h.write("RIFF", 0); h.writeUInt32LE(36 + data.length, 4); h.write("WAVE", 8); h.write("fmt ", 12);
  h.writeUInt32LE(16, 16); h.writeUInt16LE(float ? 3 : 1, 20); h.writeUInt16LE(channels, 22); h.writeUInt32LE(rate, 24);
  h.writeUInt32LE(rate * channels * bytes, 28); h.writeUInt16LE(channels * bytes, 32); h.writeUInt16LE(bytes * 8, 34);
  h.write("data", 36); h.writeUInt32LE(data.length, 40);
  return Buffer.concat([h, data]);
}

/** Decaying noise bursts at `clicks` (seconds) over a quiet bed. */
function clickTrack(clicks, seconds, rate = 22050) {
  const x = new Float32Array(Math.round(seconds * rate));
  let seed = 12345;
  const noise = () => ((seed = (seed * 1103515245 + 12345) >>> 0) / 2 ** 31 - 1);
  for (let i = 0; i < x.length; i++) x[i] = 0.002 * noise();
  for (const c of clicks) {
    const at = Math.round(c * rate);
    for (let k = 0; k < rate * 0.08 && at + k < x.length; k++) x[at + k] += 0.8 * noise() * Math.exp(-k / (rate * 0.012));
  }
  return { rate, samples: x, duration: seconds };
}

const grid = (from, to, step) => { const out = []; for (let t = from; t <= to + 1e-9; t += step) out.push(Math.round(t * 1000) / 1000); return out; };
const nearest = (xs, t) => xs.reduce((m, x) => (Math.abs(x - t) < Math.abs(m - t) ? x : m), Infinity);

// ---------- song analysis ----------

test("parseLyrics keeps sung lines with their section and normalized words", () => {
  const lines = parseLyrics("[Verse]\nPencil sketches on a paper page\n\n[Chorus]\nTurn the lights on, I’ve got a studio\n[Outro]\n");
  assert.deepEqual(lines.map((l) => [l.section, l.words.join(" ")]), [
    ["Verse", "pencil sketches on a paper page"],
    ["Chorus", "turn the lights on i've got a studio"],
  ]);
});

test("alignLyrics times each line from the words heard, tolerating misheard, missing and extra words", () => {
  const lines = parseLyrics("[Verse]\nLight on the water\nStay with me\nTurn the lights on");
  const heard = [["Light", 2], ["on", 2.4], ["a", 2.7], ["water", 3], ["uh", 3.8], ["Stay", 4.2], ["me", 5], ["Turn", 7], ["the", 7.3], ["lights", 7.6], ["on", 8]]
    .map(([word, start]) => ({ word, start, end: start + 0.3 }));
  const r = alignLyrics(lines, heard);
  assert.deepEqual(r.lines.map((l) => [l.start, l.end]), [[2, 3.3], [4.2, 5.3], [7, 8.3]]);
  assert.equal(r.lines[0].sung, 0.75);
  assert.ok(r.similarity > 0.8 && r.similarity < 1, `${r.similarity}`);
});

test("alignLyrics places a line nobody heard between its neighbours", () => {
  const lines = parseLyrics("one two\nthree four\nfive six");
  const heard = [["one", 1], ["two", 1.5], ["five", 6], ["six", 6.5]].map(([word, start]) => ({ word, start, end: start + 0.4 }));
  const [a, b, c] = alignLyrics(lines, heard).lines;
  assert.ok(b.start > a.end && b.end < c.start, JSON.stringify([a, b, c]));
});

test("transcriptWords leaves words untimed when one segment spans the clip (Qwen3-ASR), however short", () => {
  for (const end of [148.4, 6]) {
    const words = transcriptWords({ text: "light on the water", segments: [{ id: 0, start: 0, end, text: "light on the water" }] });
    assert.deepEqual(words.map((w) => [w.word, w.start]), [["light", null], ["on", null], ["the", null], ["water", null]]);
  }
});

test("transcriptWords prefers word times and splits segments evenly without them", () => {
  assert.deepEqual(transcriptWords({ words: [{ word: " hi ", start: 1, end: 1.5 }] }), [{ word: "hi", start: 1, end: 1.5 }]);
  assert.deepEqual(transcriptWords({ segments: [{ start: 2, end: 4, text: " good night " }, { start: 5, end: 6, text: "sleep" }] }), [
    { word: "good", start: 2, end: 3 }, { word: "night", start: 3, end: 4 }, { word: "sleep", start: 5, end: 6 },
  ]);
});

test("readWav decodes 16-bit stereo and 32-bit float WAVs to mono", () => {
  const pcm = readWav(wavBytes([0.5, -0.5], 8000, { channels: 2 }));
  assert.equal(pcm.rate, 8000);
  assert.ok(Math.abs(pcm.samples[0] - 0.5) < 1e-3 && Math.abs(pcm.samples[1] + 0.5) < 1e-3);
  const fl = readWav(wavBytes([0.25, 1], 48000, { float: true }));
  assert.deepEqual([...fl.samples], [0.25, 1]);
  assert.throws(() => readWav(Buffer.from("not a wav file at all, sorry")), /not a WAV/);
});

test("dynamics measures how far the quiet part sits below the loud part", () => {
  const rate = 8000, x = new Float32Array(rate * 10);
  for (let i = 0; i < x.length; i++) x[i] = Math.sin(i / 3) * (i < x.length / 2 ? 0.05 : 0.5);
  assert.ok(Math.abs(dynamics(envelopeDb({ rate, samples: x })) - 20) < 0.5);
});

test("abcTempo reads the score's tempo as quarter notes per minute", () => {
  assert.equal(abcTempo("X:1\nQ:1/4=73\nK:C"), 73);
  assert.equal(abcTempo("Q:1/8=120"), 60);
  assert.equal(abcTempo("Q:3/8=60"), 90);
  assert.equal(abcTempo("Q:90"), 90);
  assert.equal(abcTempo("X:1\nK:C"), null);
});

test("scoreTake ranks a take that was cut off at the length bound below a weaker one that ended", () => {
  const ended = scoreTake({ similarity: 0.8, duration: 150, bound: 240, spread: 10 });
  const cutOff = scoreTake({ similarity: 0.97, duration: 239.5, bound: 240, spread: 16 });
  assert.ok(ended > cutOff);
  assert.ok(scoreTake({ similarity: 0.9, duration: 150, bound: 240, spread: 16 }) > scoreTake({ similarity: 0.9, duration: 150, bound: 240, spread: 4 }));
});

test("the beat tracker finds a click track's tempo and beats, and none over its silent tail", () => {
  const clicks = grid(1, 17.8, 0.6);
  const env = onsetEnvelope(clickTrack(clicks, 20));
  for (const hint of [100, null]) assert.ok(Math.abs(estimateTempo(env, hint) - 100) <= 1, `hint ${hint}: ${estimateTempo(env, hint)}`);
  const beats = trackBeats(env, estimateTempo(env, 100));
  assert.equal(beats.length, clicks.length);
  for (const b of beats) assert.ok(Math.abs(b - nearest(clicks, b)) <= 0.02, `beat ${b}`);
});

test("the beat tracker follows a song that speeds up", () => {
  const clicks = [];
  for (let t = 1, step = 0.62; t < 29; t += step, step *= 0.997) clicks.push(Math.round(t * 1000) / 1000);
  const env = onsetEnvelope(clickTrack(clicks, 30));
  const beats = trackBeats(env, estimateTempo(env, 98));
  const late = beats.filter((b) => Math.abs(b - nearest(clicks, b)) > 0.025);
  assert.ok(beats.length >= clicks.length - 1 && !late.length, `${beats.length}/${clicks.length} beats, off: ${late}`);
});

test("beatsBetween subdivides beats inside a range", () => {
  assert.deepEqual(beatsBetween([1, 2, 3.2], 0, 10, 2), [1, 1.5, 2, 2.6, 3.2]);
  assert.deepEqual(beatsBetween([1, 2, 3.2], 1.5, 3), [2]);
});

// ---------- timeline ----------

test("validateTimeline names unknown blocks, missing fields and files, and times off the song", () => {
  const t = {
    song: { file: "song.wav", start: 5, end: 150 },
    blocks: [
      { type: "still", src: "a.png", from: 5, to: 9 },
      { type: "spin", from: 9, to: 10 },
      { type: "clip", from: 10, to: 12 },
      { type: "still", src: "missing.png", from: 12, to: 14 },
      { type: "caption", text: "hi", from: 14, to: 13 },
      { type: "sfx", src: "boom.wav" },
    ],
  };
  const r = validateTimeline(t, { exists: (f) => f !== "missing.png", songDuration: 148.4 });
  assert.equal(r.errors.length, 6, r.errors.join("\n"));
  for (const needle of ["past the end of the song", "unknown type", 'needs "src"', "missing.png is not in public/", '"to" (13) must be after', 'needs "at"']) {
    assert.ok(r.errors.some((e) => e.includes(needle)), `no error mentions ${needle}`);
  }
  assert.ok(r.warnings.some((w) => w.includes("over 140 s")));
  const ok = validateTimeline({ song: { file: "song.wav", start: 0, end: 10 }, blocks: [{ type: "title", text: "Hi", from: 0, to: 2 }] }, { songDuration: 12 });
  assert.deepEqual([ok.errors, ok.warnings, ok.duration], [[], [], 10]);
});

// ---------- a mock mlx-serve ----------

async function mockServer(models, routes) {
  const calls = [];
  const server = createServer((req, res) => {
    const chunks = [];
    req.on("data", (c) => chunks.push(c));
    req.on("end", async () => {
      const body = Buffer.concat(chunks);
      calls.push({ method: req.method, url: req.url, body });
      if (req.url === "/v1/models") return res.end(JSON.stringify({ data: models }));
      if (req.url === "/v1/unload-model") return res.end("{}");
      const route = routes[req.url];
      if (!route) { res.statusCode = 404; return res.end(JSON.stringify({ error: { message: "not found" } })); }
      await route(req, res, body);
    });
  });
  await new Promise((r) => server.listen(0, "127.0.0.1", r));
  return { url: `http://127.0.0.1:${server.address().port}`, calls, close: () => new Promise((r) => server.close(r)) };
}

const sse = (res, ...events) => {
  res.setHeader("content-type", "text/event-stream");
  for (const e of events) res.write(`data: ${JSON.stringify(e)}\n\n`);
  res.end();
};

const LYRICS = "[Verse]\nLight on the water\nStay with me\n\n[Chorus]\nTurn the lights on\n";
const SUNG = [["Light", 2], ["on", 2.4], ["the", 2.7], ["water", 3], ["Stay", 4.2], ["with", 4.6], ["me", 5], ["Turn", 7], ["the", 7.3], ["lights", 7.6], ["on", 8]];

async function songServer({ transcription = true, wholeClip = false } = {}) {
  const tracks = { 7: clickTrack(grid(1, 11, 0.6), 12), 108: clickTrack(grid(1, 29, 0.6), 30) };
  return mockServer(
    [
      { id: "ace/step", capabilities: ["music"], state: "ready" },
      { id: "ddalcu/YuE2-3B-MLX-Serve-8bit", capabilities: ["music"], state: "unloaded" },
      ...(transcription ? [{ id: "org/whisper", capabilities: ["transcription"], state: "unloaded" }] : []),
    ],
    {
      "/v1/audio/music-generations": (req, res, body) => {
        const { seed } = JSON.parse(body);
        const t = tracks[seed];
        sse(res, { type: "progress", stage: "ar", step: 1, total: 2 },
          { type: "complete", data: wavBytes(t.samples, t.rate).toString("base64"), abc: "X:1\nQ:1/4=100\nK:C\n" });
      },
      ...(transcription ? {
        "/v1/audio/transcriptions": (req, res, body) => {
          const form = body.toString("latin1");
          assert.match(form, /name="timestamp_granularities\[\]"\r\n\r\nword/);
          const words = form.includes('filename="take-7.wav"') ? SUNG : [["Like", 2], ["a", 2.5], ["daughter", 3], ["turn", 7]];
          const text = words.map(([w]) => w).join(" ");
          if (wholeClip) return res.end(JSON.stringify({ task: "transcribe", language: "english", duration: 12, text, segments: [{ id: 0, seek: 0, start: 0, end: 12, text }] }));
          res.end(JSON.stringify({ text: "", words: words.map(([word, start]) => ({ word, start, end: start + 0.3 })) }));
        },
      } : {}),
    },
  );
}

test("film.mjs song sings takes with YuE2, keeps the one that sang the lyrics and ended, and times it", async () => {
  const srv = await songServer();
  try {
    const dir = tmp(), lyrics = join(dir, "lyrics.txt");
    writeFileSync(lyrics, LYRICS);
    const env = { ...process.env, MLX_SERVE_URL: srv.url };
    const { stdout } = await run("node", [join(SKILL, "film.mjs"), "song", join(dir, "film"), "--lyrics", lyrics, "--style", "pop", "--takes", "2", "--max", "30"], { env });
    const music = srv.calls.filter((c) => c.url === "/v1/audio/music-generations").map((c) => JSON.parse(c.body));
    assert.deepEqual(music.map((m) => [m.model, m.seed, m.duration_seconds]), [["ddalcu/YuE2-3B-MLX-Serve-8bit", 7, 30], ["ddalcu/YuE2-3B-MLX-Serve-8bit", 108, 30]]);
    assert.deepEqual(srv.calls.filter((c) => c.url === "/v1/unload-model").map((c) => JSON.parse(c.body).model), ["ddalcu/YuE2-3B-MLX-Serve-8bit", "org/whisper"]);

    const song = JSON.parse(readFileSync(join(dir, "film", "song.json"), "utf8"));
    assert.equal(song.from, "takes/take-7.wav");
    assert.ok(readFileSync(join(dir, "film", "public", "song.wav")).equals(readFileSync(join(dir, "film", "public", "takes", "take-7.wav"))));
    assert.deepEqual(song.lines.map((l) => [l.section, l.text, l.start, l.end]), [
      ["Verse", "Light on the water", 2, 3.3], ["Verse", "Stay with me", 4.2, 5.3], ["Chorus", "Turn the lights on", 7, 8.3],
    ]);
    assert.equal(song.words.length, SUNG.length);
    assert.ok(Math.abs(song.bpm - 100) <= 1, `${song.bpm}`);
    for (const b of song.beats) assert.ok(Math.abs(b - nearest(grid(1, 11, 0.6), b)) <= 0.02, `beat ${b}`);
    assert.equal(song.firstVocal, 2);
    assert.match(stdout, /"start": 0, "end": 12/);

    // a second run reuses the takes on disk
    const before = srv.calls.length;
    await run("node", [join(SKILL, "film.mjs"), "song", join(dir, "film"), "--lyrics", lyrics, "--style", "pop", "--takes", "2", "--max", "30"], { env });
    assert.equal(srv.calls.slice(before).filter((c) => c.url === "/v1/audio/music-generations").length, 0);
    const { stdout: printed } = await run("node", [join(SKILL, "film.mjs"), "words", join(dir, "film")], { env });
    assert.match(printed, /7\.00-8\.30\s+\[Chorus\]\s+Turn the lights on/);
  } finally {
    await srv.close();
  }
});

test("film.mjs song without a transcription endpoint still picks a song and its beats", async () => {
  const srv = await songServer({ transcription: false });
  try {
    const dir = tmp(), lyrics = join(dir, "lyrics.txt");
    writeFileSync(lyrics, LYRICS);
    const { stderr } = await run("node", [join(SKILL, "film.mjs"), "song", join(dir, "film"), "--lyrics", lyrics, "--style", "pop", "--takes", "1", "--max", "30"], { env: { ...process.env, MLX_SERVE_URL: srv.url } });
    assert.match(stderr, /no \/v1\/audio\/transcriptions/);
    const song = JSON.parse(readFileSync(join(dir, "film", "song.json"), "utf8"));
    assert.deepEqual([song.lines, song.words], [[], []]);
    assert.ok(song.beats.length > 10);
  } finally {
    await srv.close();
  }
});

test("film.mjs song on a transcript without word times scores the takes but writes no line times", async () => {
  const srv = await songServer({ wholeClip: true });
  try {
    const dir = tmp(), lyrics = join(dir, "lyrics.txt");
    writeFileSync(lyrics, LYRICS);
    const { stderr } = await run("node", [join(SKILL, "film.mjs"), "song", join(dir, "film"), "--lyrics", lyrics, "--style", "pop", "--takes", "2", "--max", "30"], { env: { ...process.env, MLX_SERVE_URL: srv.url } });
    assert.match(stderr, /no word times/);
    const song = JSON.parse(readFileSync(join(dir, "film", "song.json"), "utf8"));
    assert.equal(song.from, "takes/take-7.wav");
    assert.ok(song.takes.find((t) => t.seed === 7).similarity > song.takes.find((t) => t.seed === 108).similarity);
    assert.deepEqual([song.lines, song.words], [[], []]);
    assert.ok(song.beats.length > 10);
    await assert.rejects(run("node", [join(SKILL, "film.mjs"), "words", join(dir, "film")]), (e) => /no line times/.test(e.stderr) && /film\.mjs beats/.test(e.stderr));
  } finally {
    await srv.close();
  }
});

// ---------- gen.mjs video ----------

// Stands in for ffmpeg: keeps its arguments and the raw inputs gen.mjs streamed to disk.
const FAKE_FFMPEG = `#!/usr/bin/env node
const fs = require("node:fs");
const a = process.argv.slice(2), out = a[a.length - 1];
const inputs = a.flatMap((v, i) => (v === "-i" ? [a[i + 1]] : []));
fs.writeFileSync(out + ".args.json", JSON.stringify(a));
inputs.forEach((f, i) => fs.copyFileSync(f, out + ".in" + i));
fs.writeFileSync(out, "mp4");
`;

const H3 = [{ id: "org/MiniMax-H3", capabilities: ["video"], state: "ready" }];

async function videoRun({ frames, escapeSlashes, args, models = H3 }) {
  const w = 64, h = 48, rgb = Buffer.alloc(w * h * 3 * frames), pcm = Buffer.alloc(32000 * 2 * 2 / 2);
  for (let i = 0; i < rgb.length; i++) rgb[i] = (i * 7 + (i >> 9)) & 255;
  for (let i = 0; i < pcm.length; i++) pcm[i] = (i * 13) & 255;
  const srv = await mockServer(models, {
    "/v1/video/generations": (req, res) => {
      res.setHeader("content-type", "text/event-stream");
      res.write(`data: ${JSON.stringify({ type: "progress", stage: "denoise", step: 1, total: 4 })}\n\n`);
      let data = rgb.toString("base64");
      if (escapeSlashes) data = data.replace(/\//g, "\\/");
      const ev = `data: {"type":"complete","frames":${frames},"height":${h},"width":${w},"fps":24,"format":"rgb8","data":"${data}",` +
        `"audio_sample_rate":32000,"audio_channels":2,"audio_format":"pcm_s16le","audio_data":"${pcm.toString("base64")}"}\n\n`;
      // in pieces, the way a big body arrives
      for (let i = 0; i < ev.length; i += 50000) res.write(ev.slice(i, i + 50000));
      res.end();
    },
  });
  try {
    const dir = tmp(), ff = join(dir, "ffmpeg"), out = join(dir, "clip.mp4"), first = join(dir, "first.png");
    writeFileSync(ff, FAKE_FFMPEG);
    chmodSync(ff, 0o755);
    writeFileSync(first, "PNG!");
    await run("node", [join(SKILL, "gen.mjs"), "video", "a paper boat in a storm", "-o", out, "--first-frame", first, "--turbo", ...args],
      { env: { ...process.env, MLX_SERVE_URL: srv.url, FFMPEG: ff }, maxBuffer: 1 << 24 });
    const body = JSON.parse(srv.calls.find((c) => c.url === "/v1/video/generations").body);
    return { dir, out, body, rgb, pcm, ffArgs: JSON.parse(readFileSync(out + ".args.json", "utf8")) };
  } finally {
    await srv.close();
  }
}

test("gen.mjs video streams a big result to disk and muxes frames with the model's sound", async () => {
  const r = await videoRun({ frames: 120, escapeSlashes: true, args: ["--size", "64x48", "--frames", "120", "--seed", "3"] });
  assert.ok(r.rgb.length * 4 / 3 > 1 << 20, "the event must be past the streaming threshold");
  assert.ok(readFileSync(r.out + ".in0").equals(r.rgb), "the frames reach ffmpeg as sent");
  assert.ok(readFileSync(r.out + ".in1").equals(r.pcm), "the sound reaches ffmpeg as sent");
  assert.deepEqual([r.body.model, r.body.width, r.body.height, r.body.num_frames, r.body.turbo, r.body.seed, r.body.first_frame_image],
    ["org/MiniMax-H3", 64, 48, 120, true, 3, Buffer.from("PNG!").toString("base64")]);
  const a = r.ffArgs.join(" ");
  for (const s of ["-s 64x48", "-r 24", "-f s16le -ar 32000 -ac 2", "-pix_fmt yuv420p"]) assert.ok(a.includes(s), `${s} in ${a}`);
  assert.ok(!existsSync(r.out + ".rgb") && !existsSync(r.out + ".pcm"), "raw intermediates are removed");
});

test("gen.mjs video asks the first-frame H3 pack for a 16:9 five-second clip by default and reads a small result whole", async () => {
  const models = ["ddalcu/LTX-2.5-MLX-Serve-8bit", "ddalcu/MiniMax-H3-REF2VA-MLX-Serve-8bit", "ddalcu/MiniMax-H3-FL2VA-MLX-Serve-8bit"]
    .map((id) => ({ id, capabilities: ["video"], state: "unloaded" }));
  const r = await videoRun({ frames: 5, escapeSlashes: false, args: [], models });
  assert.deepEqual([r.body.model, r.body.width, r.body.height, r.body.num_frames], ["ddalcu/MiniMax-H3-FL2VA-MLX-Serve-8bit", 1312, 736, 124]);
  assert.ok(readFileSync(r.out + ".in0").equals(r.rgb), "the frames reach ffmpeg as sent");
});

test("gen.mjs video reports the server's error and leaves no partial frames on disk", async () => {
  const srv = await mockServer([{ id: "org/MiniMax-H3", capabilities: ["video"], state: "ready" }], {
    "/v1/video/generations": (req, res) => sse(res, { type: "progress", stage: "denoise", step: 1, total: 4 }, { type: "error", message: "out of memory" }),
  });
  try {
    const out = join(tmp(), "clip.mp4");
    await assert.rejects(run("node", [join(SKILL, "gen.mjs"), "video", "a storm", "-o", out], { env: { ...process.env, MLX_SERVE_URL: srv.url } }),
      (e) => /out of memory/.test(e.stderr));
    assert.ok(!existsSync(out + ".rgb") && !existsSync(out));
  } finally {
    await srv.close();
  }
});

// ---------- init ----------

// ---------- gen.mjs 3d ----------

const MOSS_AND_TRELLIS = [{ id: "org/moss", capabilities: ["image"], state: "ready" }, { id: "org/trellis", capabilities: ["3d"], state: "ready" }];

async function threeDRun({ ref, input = "a red car", args = [], glb = Buffer.from("glb"), models = MOSS_AND_TRELLIS }) {
  const srv = await mockServer(
    models,
    {
      "/v1/images/generations": (req, res) => sse(res, { type: "complete", data: [{ b64_json: ref.toString("base64") }] }),
      "/v1/3d/generations": (req, res) => sse(res, { type: "complete", format: "glb", data: glb.toString("base64") }),
    },
  );
  try {
    const dir = tmp(), out = join(dir, "car.glb");
    await run("node", [join(SKILL, "gen.mjs"), "3d", input, "-o", out, ...args], { env: { ...process.env, MLX_SERVE_URL: srv.url } });
    const body = JSON.parse(srv.calls.find((c) => c.url === "/v1/3d/generations").body);
    let sent;
    try { sent = decodePng(Buffer.from(body.image, "base64")); } catch { /* a reference sent as it was may be a PNG this decoder refuses */ }
    return { sent, raw: body.image, body, dir, out };
  } finally {
    await srv.close();
  }
}

const pixelAt = ({ w, data }, x, y) => Array.from(data.subarray((y * w + x) * 4, (y * w + x) * 4 + 4));

test("gen.mjs 3d sends the finish only when asked, so the model keeps its own roughness otherwise", async () => {
  const png = encodePng({ w: 8, h: 8, data: new Uint8Array(8 * 8 * 4).fill(255) });
  const plain = await threeDRun({ ref: png });
  assert.equal(plain.body.roughness_scale, undefined);
  assert.equal(plain.body.clearcoat, undefined);
  const painted = await threeDRun({ ref: png, args: ["--roughness-scale", "0.4", "--clearcoat", "1"] });
  assert.equal(painted.body.roughness_scale, 0.4);
  assert.equal(painted.body.clearcoat, 1);
});

test("gen.mjs 3d sends --resolution, so a GPU short of the 1024 peak can ask for 512", async () => {
  const png = encodePng({ w: 8, h: 8, data: new Uint8Array(8 * 8 * 4).fill(255) });
  assert.equal((await threeDRun({ ref: png })).body.resolution, undefined);
  assert.equal((await threeDRun({ ref: png, args: ["--resolution", "512"] })).body.resolution, 512);
});

test("gen.mjs 3d cuts the plain background off the reference image, because TRELLIS.2 and Pixal3D read the alpha as the subject", async () => {
  const w = 32, h = 32, data = new Uint8Array(w * h * 4).fill(255);
  for (let y = 12; y < 20; y++) for (let x = 12; x < 20; x++) data.set([200, 30, 30, 255], (y * w + x) * 4);
  const r = await threeDRun({ ref: encodePng({ w, h, data }) });
  assert.equal(pixelAt(r.sent, 1, 1)[3], 0, "the white background is transparent");
  assert.deepEqual(pixelAt(r.sent, 16, 16), [200, 30, 30, 255], "the object keeps its colour and is opaque");
  assert.equal(r.body.model, "org/trellis");
});

test("gen.mjs 3d sends a reference it cannot decode (16-bit PNG) as it is instead of failing the job", async () => {
  const png = encodePng({ w: 8, h: 8, data: new Uint8Array(8 * 8 * 4).fill(255) });
  png[24] = 16; // IHDR bit depth: the decoder here takes 8-bit only
  const r = await threeDRun({ ref: png });
  assert.equal(r.raw, png.toString("base64"));
});

test("gen.mjs 3d sends a reference as it is when the cutout would erase the whole subject (white on white)", async () => {
  const png = encodePng({ w: 32, h: 32, data: new Uint8Array(32 * 32 * 4).fill(255) });
  const r = await threeDRun({ ref: png });
  assert.equal(r.raw, png.toString("base64"));
});

test("gen.mjs 3d leaves a reference that already has transparency alone", async () => {
  const w = 16, h = 16, data = new Uint8Array(w * h * 4); // all transparent black, one dark opaque object
  for (let y = 6; y < 10; y++) for (let x = 6; x < 10; x++) data.set([10, 10, 10, 255], (y * w + x) * 4);
  const png = encodePng({ w, h, data });
  const dir = tmp(), file = join(dir, "own.png");
  writeFileSync(file, png);
  const r = await threeDRun({ ref: png, input: file });
  assert.deepEqual(pixelAt(r.sent, 7, 7), [10, 10, 10, 255], "a dark object is not mistaken for background");
  assert.equal(pixelAt(r.sent, 0, 0)[3], 0);
});

/** A car-ish wedge, 1.8 wide and 4 long, nose (lower, 0.9 high) at local +z, tail 1.4 high, turned `yawDeg` about Y. */
function wedgeGlb(yawDeg) {
  const a = (yawDeg * Math.PI) / 180, turn = ([x, y, z]) => [x * Math.cos(a) + z * Math.sin(a), y, -x * Math.sin(a) + z * Math.cos(a)];
  const pos = [], nrm = [];
  for (const z of [-2, 2]) for (const x of [-0.9, 0.9]) for (const top of [0, 1]) {
    pos.push(...turn([x, top ? (z > 0 ? 0.9 : 1.4) : 0, z]));
    nrm.push(...turn([0, 0, 1]));
  }
  const idx = [0, 1, 2, 2, 1, 3, 4, 6, 5, 5, 6, 7];
  const f = new Float32Array([...pos, ...nrm]), u = new Uint16Array(idx);
  const bin = Buffer.concat([Buffer.from(f.buffer), Buffer.from(u.buffer)]);
  const xs = [0, 1, 2].map((c) => pos.filter((_, i) => i % 3 === c));
  const json = {
    asset: { version: "2.0" }, scene: 0, scenes: [{ nodes: [0] }], nodes: [{ mesh: 0 }],
    meshes: [{ primitives: [{ attributes: { POSITION: 0, NORMAL: 1 }, indices: 2 }] }],
    buffers: [{ byteLength: bin.length }],
    bufferViews: [{ buffer: 0, byteOffset: 0, byteLength: pos.length * 4 }, { buffer: 0, byteOffset: pos.length * 4, byteLength: nrm.length * 4 }, { buffer: 0, byteOffset: pos.length * 8, byteLength: u.byteLength }],
    accessors: [
      { bufferView: 0, componentType: 5126, count: 8, type: "VEC3", min: xs.map((v) => Math.min(...v)), max: xs.map((v) => Math.max(...v)) },
      { bufferView: 1, componentType: 5126, count: 8, type: "VEC3" },
      { bufferView: 2, componentType: 5123, count: 12, type: "SCALAR" },
    ],
  };
  return writeGlb(json, bin);
}

test("gen.mjs glb --vehicle turns a yawed car so its length runs along Z with the nose on +Z, normals and bounds following", () => {
  for (const yaw of [0, 33, 147, 270]) {
    const out = parseGlb(fixGlb(wedgeGlb(yaw), { vehicle: true }));
    const P = readAccessor(out.json, out.bin, 0), N = readAccessor(out.json, out.bin, 1);
    const lo = [0, 1, 2].map((c) => Math.min(...Array.from({ length: 8 }, (_, i) => P[i * 3 + c])));
    const hi = [0, 1, 2].map((c) => Math.max(...Array.from({ length: 8 }, (_, i) => P[i * 3 + c])));
    assert.ok(Math.abs(hi[0] - lo[0] - 1.8) < 1e-3 && Math.abs(hi[2] - lo[2] - 4) < 1e-3, `yaw ${yaw}: extent ${hi.map((h, c) => (h - lo[c]).toFixed(3))}`);
    for (let i = 0; i < 8; i++) if (P[i * 3 + 2] > 0) assert.ok(Math.max(P[i * 3 + 1], 0) <= 0.9 + 1e-3, `yaw ${yaw}: the +Z end is the low nose`);
    assert.ok(Array.from({ length: 8 }, (_, i) => N[i * 3 + 2]).every((z) => Math.abs(z - 1) < 1e-3), `yaw ${yaw}: normals turned with the mesh`);
    assert.deepEqual(out.json.accessors[0].min.map((v) => +v.toFixed(3)), lo.map((v) => +v.toFixed(3)), "accessor bounds follow the vertices");
  }
});

test("gen.mjs glb --vehicle with --height keeps the pivot at the base centre of the straightened car", () => {
  const out = parseGlb(fixGlb(wedgeGlb(33), { vehicle: true, height: 1.2 }));
  const node = out.json.nodes[out.json.scenes[0].nodes[0]], s = node.scale[0], P = readAccessor(out.json, out.bin, 0);
  const lo = [0, 1, 2].map((c) => Math.min(...Array.from({ length: 8 }, (_, i) => P[i * 3 + c])));
  const hi = [0, 1, 2].map((c) => Math.max(...Array.from({ length: 8 }, (_, i) => P[i * 3 + c])));
  assert.ok(Math.abs(node.translation[0] + ((lo[0] + hi[0]) / 2) * s) < 1e-4 && Math.abs(node.translation[2] + ((lo[2] + hi[2]) / 2) * s) < 1e-4, "centred on the car, not on its old yawed box");
  assert.ok(Math.abs((hi[1] - lo[1]) * s - 1.2) < 1e-4 && Math.abs(hi[2] - lo[2] - 4) < 1e-3, "straightened, scaled to the asked height");
});

/** A textured plane of n x n quads (uint32 indices, UVs 0..uvScale) with two `side`-pixel checkerboard PNGs
 *  (base colour, metallic-roughness) and `orphans` vertices no triangle uses. */
function texturedGlb({ n = 24, side = 64, orphans = 0, uvScale = 1, node = {} } = {}) {
  const pos = [], nrm = [], uv = [], idx = [];
  for (let y = 0; y <= n; y++) for (let x = 0; x <= n; x++) {
    pos.push(x * 0.1, y * 0.1, 0.03 * Math.sin(x * 0.7) * Math.cos(y * 0.5));
    nrm.push(0, 0, 1);
    uv.push((x / n) * uvScale, (y / n) * uvScale);
  }
  for (let o = 0; o < orphans; o++) { pos.push(50 + o, 50, 50); nrm.push(0, 0, 1); uv.push(0.5, 0.5); }
  const v = (x, y) => y * (n + 1) + x;
  for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) idx.push(v(x, y), v(x + 1, y), v(x + 1, y + 1), v(x, y), v(x + 1, y + 1), v(x, y + 1));
  const checker = () => { const d = new Uint8Array(side * side * 4); for (let i = 0; i < side * side; i++) { const on = ((i % side) + Math.floor(i / side)) % 2 ? 255 : 0; d.set([on, on, on, 255], i * 4); } return encodePng({ w: side, h: side, data: d }); };
  const parts = [Buffer.from(new Float32Array(pos).buffer), Buffer.from(new Float32Array(nrm).buffer), Buffer.from(new Float32Array(uv).buffer), Buffer.from(new Uint32Array(idx).buffer), checker(), checker()];
  const views = [], accessors = [];
  let off = 0;
  parts.forEach((b, i) => { views.push({ buffer: 0, byteOffset: off, byteLength: b.length }); off += b.length + ((4 - (b.length % 4)) % 4); });
  const bin = Buffer.concat(parts.map((b) => Buffer.concat([b, Buffer.alloc((4 - (b.length % 4)) % 4)])));
  const cnt = pos.length / 3, mm = [0, 1, 2].map((c) => pos.filter((_, i) => i % 3 === c));
  accessors.push({ bufferView: 0, componentType: 5126, count: cnt, type: "VEC3", min: mm.map((a) => a.reduce((x, y) => Math.min(x, y))), max: mm.map((a) => a.reduce((x, y) => Math.max(x, y))) },
    { bufferView: 1, componentType: 5126, count: cnt, type: "VEC3" }, { bufferView: 2, componentType: 5126, count: cnt, type: "VEC2" },
    { bufferView: 3, componentType: 5125, count: idx.length, type: "SCALAR" });
  return writeGlb({
    asset: { version: "2.0" }, scene: 0, scenes: [{ nodes: [0] }], nodes: [{ mesh: 0, ...node }],
    meshes: [{ primitives: [{ attributes: { POSITION: 0, NORMAL: 1, TEXCOORD_0: 2 }, indices: 3, material: 0 }] }],
    materials: [{ pbrMetallicRoughness: { baseColorTexture: { index: 0 }, metallicRoughnessTexture: { index: 1 }, metallicFactor: 1, roughnessFactor: 0.4 } }],
    textures: [{ source: 0 }, { source: 1 }], images: [{ bufferView: 4, mimeType: "image/png" }, { bufferView: 5, mimeType: "image/png" }],
    buffers: [{ byteLength: bin.length }], bufferViews: views, accessors,
  }, bin);
}

const pngOf = (g, i) => { const v = g.json.bufferViews[g.json.images[i].bufferView]; return decodePng(g.bin.subarray(v.byteOffset || 0, (v.byteOffset || 0) + v.byteLength)); };
const trianglesOf = (g) => g.json.accessors[g.json.meshes[0].primitives[0].indices].count / 3;

test("gen.mjs glb --max-tris leaves no old indices or unused vertices behind, so the file shrinks with the mesh", () => {
  const src = texturedGlb({ n: 24, orphans: 5 });
  const out = fixGlb(src, { maxTris: 400 });
  const g = parseGlb(out), p = g.json.meshes[0].primitives[0];
  assert.ok(trianglesOf(g) < 1152 * 0.7, `simplified: ${trianglesOf(g)} triangles`);
  assert.ok(out.length < src.length, `file ${src.length} -> ${out.length} bytes`);
  const idx = readAccessor(g.json, g.bin, p.indices), count = g.json.accessors[p.attributes.POSITION].count;
  assert.equal(new Set(idx).size, count, "every vertex is used by a triangle, so none were left behind");
  assert.ok(Math.max(...idx) < count);
  for (const a of ["POSITION", "NORMAL", "TEXCOORD_0"]) assert.equal(g.json.accessors[p.attributes[a]].count, count, `${a} follows the vertices`);
  assert.deepEqual([pngOf(g, 0).w, pngOf(g, 1).w], [64, 64], "the textures ride through untouched");
  assert.ok(g.bin.length - g.json.buffers[0].byteLength < 4, "the buffer's length is its own, short of the chunk's padding by under 4 bytes");
});

test("gen.mjs glb --max-texture shrinks each texture by area, converting colour through linear light and data maps as they are", () => {
  const g = parseGlb(fixGlb(texturedGlb({ side: 64 }), { maxTexture: 32 }));
  assert.deepEqual([pngOf(g, 0).w, pngOf(g, 0).h, pngOf(g, 1).w], [32, 32, 32]);
  // A one-pixel black/white checkerboard averages to half the light: sRGB 188 for colour, plain 128 for data.
  const colour = pngOf(g, 0).data, data = pngOf(g, 1).data;
  assert.ok(Math.abs(colour[4 * 100] - 188) <= 2, `base colour ${colour[4 * 100]}`);
  assert.ok(Math.abs(data[4 * 100] - 128) <= 1, `metallic-roughness ${data[4 * 100]}`);
  const same = parseGlb(fixGlb(texturedGlb({ side: 64 }), { maxTexture: 128 }));
  assert.equal(pngOf(same, 0).w, 64, "a texture already under the limit is left alone");
});

test("gen.mjs glb --quantize writes 16-bit positions, 8-bit normals and 16-bit UVs that dequantize to the original mesh", () => {
  const src = texturedGlb({ n: 24 }), plain = parseGlb(src);
  const out = fixGlb(src, { quantize: true }), g = parseGlb(out), p = g.json.meshes[0].primitives[0];
  assert.ok(out.length < src.length, `file ${src.length} -> ${out.length} bytes`);
  assert.ok(g.json.extensionsRequired.includes("KHR_mesh_quantization") && g.json.extensionsUsed.includes("KHR_mesh_quantization"));
  const [pa, na, ua] = ["POSITION", "NORMAL", "TEXCOORD_0"].map((a) => g.json.accessors[p.attributes[a]]);
  assert.deepEqual([pa.componentType, pa.normalized, na.componentType, na.normalized, ua.componentType, ua.normalized], [5122, true, 5120, true, 5123, true]);
  assert.equal(g.json.bufferViews[pa.bufferView].byteStride % 4, 0, "positions pad to 4 bytes");
  assert.equal(g.json.bufferViews[na.bufferView].byteStride % 4, 0, "normals pad to 4 bytes");
  // Vertices are renumbered, so compare what each triangle corner points at, old file against new.
  const node = g.json.nodes[0], idxNew = readAccessor(g.json, g.bin, p.indices), idxOld = readAccessor(plain.json, plain.bin, 3);
  const Pq = readAccessor(g.json, g.bin, p.attributes.POSITION), Po = readAccessor(plain.json, plain.bin, 0);
  const Nq = readAccessor(g.json, g.bin, p.attributes.NORMAL), Uq = readAccessor(g.json, g.bin, p.attributes.TEXCOORD_0), Uo = readAccessor(plain.json, plain.bin, 2);
  assert.equal(idxNew.length, idxOld.length, "the same triangles");
  let worst = 0, worstUv = 0;
  for (let k = 0; k < idxNew.length; k++) {
    for (let c = 0; c < 3; c++) worst = Math.max(worst, Math.abs((Pq[idxNew[k] * 3 + c] / 32767) * node.scale[c] + node.translation[c] - Po[idxOld[k] * 3 + c]));
    for (let c = 0; c < 2; c++) worstUv = Math.max(worstUv, Math.abs(Uq[idxNew[k] * 2 + c] / 65535 - Uo[idxOld[k] * 2 + c]));
    assert.ok(Math.abs(Nq[idxNew[k] * 3 + 2] / 127 - 1) < 1e-6 && Nq[idxNew[k] * 3] === 0, "the normal (0,0,1) survives");
  }
  assert.ok(worst < 1e-4, `largest position error ${worst}`);
  assert.ok(worstUv < 1.6e-5, `largest UV error ${worstUv}`);
});

test("gen.mjs glb --quantize keeps UVs that tile past 1 as floats, and skips a mesh whose node already has a transform", () => {
  const tiled = parseGlb(fixGlb(texturedGlb({ uvScale: 4 }), { quantize: true }));
  const p = tiled.json.meshes[0].primitives[0];
  assert.equal(tiled.json.accessors[p.attributes.TEXCOORD_0].componentType, 5126, "tiling UVs cannot be normalized 16-bit");
  assert.equal(tiled.json.accessors[p.attributes.POSITION].componentType, 5122, "the rest still quantizes");
  const moved = parseGlb(fixGlb(texturedGlb({ node: { translation: [1, 2, 3] } }), { quantize: true }));
  assert.equal(moved.json.accessors[moved.json.meshes[0].primitives[0].attributes.POSITION].componentType, 5126, "a node with its own transform is left in floats");
  assert.ok(!(moved.json.extensionsRequired || []).includes("KHR_mesh_quantization"));
});

test("gen.mjs glb repacks a mesh of 200k vertices, quantized or not (a spread of that many values overflows the stack)", () => {
  const big = texturedGlb({ n: 450, side: 8 });
  for (const opts of [{ quantize: true }, { maxTexture: 4 }]) {
    const g = parseGlb(fixGlb(big, opts));
    const p = g.json.meshes[0].primitives[0];
    assert.equal(g.json.accessors[p.attributes.POSITION].count, 451 * 451);
    assert.equal(trianglesOf(g), 450 * 450 * 2);
    assert.equal(g.json.accessors[p.attributes.POSITION].min.length, 3);
  }
});

test("gen.mjs glb takes --max-texture and --quantize from the command line", async () => {
  const dir = tmp(), src = join(dir, "in.glb"), out = join(dir, "out.glb");
  writeFileSync(src, texturedGlb({ side: 64 }));
  await run("node", [join(SKILL, "gen.mjs"), "glb", src, "-o", out, "--max-texture", "32", "--quantize"], { env: process.env });
  const g = parseGlb(readFileSync(out));
  assert.equal(pngOf(g, 0).w, 32);
  assert.equal(g.json.accessors[g.json.meshes[0].primitives[0].attributes.POSITION].componentType, 5122);
});

test("gen.mjs 3d applies --max-texture and --quantize to the model it gets back, and keeps both off the request", async () => {
  const ref = encodePng({ w: 8, h: 8, data: new Uint8Array(8 * 8 * 4).fill(255) });
  const r = await threeDRun({ ref, args: ["--max-texture", "32", "--quantize"], glb: texturedGlb({ side: 64 }) });
  const g = parseGlb(readFileSync(r.out));
  assert.equal(pngOf(g, 0).w, 32);
  assert.ok(g.json.extensionsRequired.includes("KHR_mesh_quantization"));
  assert.equal(r.body.max_texture, undefined);
  assert.equal(r.body.quantize, undefined);
});

test("gen.mjs 3d prefers TRELLIS.2 over another 3D model, even when the other one is the loaded one", async () => {
  const ref = encodePng({ w: 8, h: 8, data: new Uint8Array(8 * 8 * 4).fill(255) });
  const models = [
    { id: "org/moss", capabilities: ["image"], state: "ready" },
    { id: "local/Pixal3D-local", capabilities: ["3d"], state: "ready" },
    { id: "org/TRELLIS.2-4B-MLX-Serve-8bit", capabilities: ["3d"], state: "unloaded" },
  ];
  assert.equal((await threeDRun({ ref, models })).body.model, "org/TRELLIS.2-4B-MLX-Serve-8bit");
  const only = models.filter((m) => !/TRELLIS/.test(m.id));
  assert.equal((await threeDRun({ ref, models: only })).body.model, "local/Pixal3D-local", "without TRELLIS.2 the other model serves");
});

test("gen.mjs 3d asks the server for about three times max_tris faces, so the UV unwrap is not run on 300k faces", async () => {
  const ref = encodePng({ w: 8, h: 8, data: new Uint8Array(8 * 8 * 4).fill(255) });
  const r = await threeDRun({ ref, args: ["--max-tris", "8000"], glb: wedgeGlb(0) });
  assert.equal(r.body.decimation_target, 24000);
  const small = await threeDRun({ ref, args: ["--max-tris", "1000"], glb: wedgeGlb(0) });
  assert.equal(small.body.decimation_target, 10000, "the server's lowest accepted target");
  const big = await threeDRun({ ref, args: ["--max-tris", "500000"], glb: wedgeGlb(0) });
  assert.equal(big.body.decimation_target, 300000, "never above the server's default");
  const none = await threeDRun({ ref });
  assert.equal(none.body.decimation_target, undefined, "no max_tris, nothing sent");
});

test("film.mjs init writes a project whose starter timeline checks clean", async () => {
  const dir = join(tmp(), "proj");
  await run("node", [join(SKILL, "film.mjs"), "init", dir, "--no-install"]);
  assert.equal(readFileSync(join(dir, "src", "index.tsx"), "utf8"), readFileSync(join(SKILL, "film.tsx"), "utf8"));
  const pkg = JSON.parse(readFileSync(join(dir, "package.json"), "utf8"));
  assert.equal(pkg.dependencies.remotion, pkg.dependencies["@remotion/renderer"]);
  const noise = readFileSync(join(dir, "public", "_film", "noise.png"));
  assert.deepEqual([...noise.subarray(1, 4)], [...Buffer.from("PNG")]);
  assert.equal(noise.readUInt32BE(16), 512);
  const { stderr } = await run("node", [join(SKILL, "film.mjs"), "check", dir]);
  assert.match(stderr, /0 errors/);
  // init never overwrites the user's timeline or their edited renderer
  writeFileSync(join(dir, "timeline.json"), "{}");
  writeFileSync(join(dir, "src", "index.tsx"), "// edited");
  await run("node", [join(SKILL, "film.mjs"), "init", dir, "--no-install"]);
  assert.equal(readFileSync(join(dir, "timeline.json"), "utf8"), "{}");
  assert.equal(readFileSync(join(dir, "src", "index.tsx"), "utf8"), "// edited");
});
