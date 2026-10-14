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
