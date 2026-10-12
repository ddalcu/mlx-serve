#!/usr/bin/env node
// gen.mjs: generate game assets on a local mlx-serve and post-process them.
// Node 18+, no dependencies. `node gen.mjs help` for usage.
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { basename, dirname, extname, join, resolve } from "node:path";
import { deflateSync, inflateSync } from "node:zlib";

const BASE = (process.env.MLX_SERVE_URL || "http://127.0.0.1:11234").replace(/\/+$/, "");

const HELP = `gen.mjs: mlx-serve asset generator and post-processor (server: ${BASE}, env MLX_SERVE_URL)

Generate (picks the model by capability from /v1/models; --model <id> overrides):
  node gen.mjs image "<prompt>" -o roof.png [--size 1024x1024] [--seed 7] [--tile] [--maps] [--cutout]
  node gen.mjs speech "<text>" -o line.wav [--voice af_heart]
  node gen.mjs music "<prompt>" -o theme.wav [--lyrics lyrics.txt] [--duration 90]
  node gen.mjs sound "<prompt>" -o door.wav [--duration 2]
  node gen.mjs 3d "<object prompt>" -o lantern.glb [--height 1.6] [--max-tris 8000]
  node gen.mjs 3d ref.png -o lantern.glb          (from an image of one object)
  node gen.mjs batch assets.json [--force] [--keep-loaded]
  node gen.mjs models

Post-process (PNG and GLB in, PNG and GLB out; no server needed):
  node gen.mjs maps albedo.png [--strength 2] [--roughness 0.75]  -> albedo_normal/_roughness/_ao.png
  node gen.mjs tile in.png -o out.png [--band 0.25]                -> seamless when repeated
  node gen.mjs cutout in.png -o out.png [--tolerance 40] [--trim] [--holes]
                                    -> plain background made transparent (--holes: enclosed pockets too)
  node gen.mjs glb in.glb -o out.glb [--height 1.6 | --size 2] [--max-tris 8000]
                                    -> pivot at the base centre, scaled to metres, fewer triangles

batch file:
  {"out": "public/assets", "jobs": [
    {"type": "image", "id": "roof_tiles", "prompt": "...", "tile": true, "maps": true},
    {"type": "image", "id": "maple_leaves", "prompt": "a cluster of red maple leaves, flat, plain white background", "cutout": true, "holes": true},
    {"type": "speech", "id": "narration", "input": "..."},
    {"type": "music", "id": "theme", "prompt": "...", "lyrics": "[verse]\\n..."},
    {"type": "sound", "id": "door", "prompt": "...", "duration_seconds": 2},
    {"type": "3d", "id": "stone_lantern", "prompt": "a granite Japanese stone lantern", "height_m": 1.6, "max_tris": 8000}]}
  Jobs run one at a time. Other job fields go to the endpoint as is (steps, voice, lyrics, ...).
  Writes <out>/<id>.png|wav|glb plus <out>/manifest.json; a job whose spec and model did not
  change is skipped (--force regenerates). Media models are unloaded at the end (--keep-loaded).`;

// ---------- server ----------

const ENDPOINT = {
  image: "/v1/images/generations",
  speech: "/v1/audio/speech",
  music: "/v1/audio/music-generations",
  sound: "/v1/audio/sound-generations",
  "3d": "/v1/3d/generations",
};
const has = (m, c) => (m.capabilities || []).includes(c);
const WANTS = {
  image: (m) => has(m, "image") && !has(m, "3d"),
  speech: (m) => has(m, "audio") && !has(m, "music") && !has(m, "sound"),
  music: (m) => has(m, "music"),
  sound: (m) => has(m, "sound"),
  "3d": (m) => has(m, "3d"),
};

let modelList = null;
async function listModels() {
  if (modelList) return modelList;
  const res = await fetch(BASE + "/v1/models").catch((e) => {
    throw new Error(`mlx-serve is not reachable at ${BASE} (${e.cause?.code || e.message}); start it or set MLX_SERVE_URL`);
  });
  if (!res.ok) throw new Error(`GET /v1/models: HTTP ${res.status}`);
  return (modelList = (await res.json()).data || []);
}

/** A local model that is already loaded wins, then a local one, then a LAN peer's. */
async function pickModel(kind, explicit) {
  if (explicit) return explicit;
  const rank = (m) => (m.state === "ready" ? 0 : m.state === "remote" ? 2 : 1);
  const found = (await listModels()).filter(WANTS[kind]).sort((a, b) => rank(a) - rank(b));
  if (!found.length) throw new Error(`no model with the ${kind} capability: download one in the MLX Core app (Models)`);
  return found[0].id;
}

/** POST a streamed media job; returns the SSE `complete` event. */
async function generate(kind, body) {
  const res = await fetch(BASE + ENDPOINT[kind], {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ ...body, stream: true }),
  });
  if (!res.ok) {
    const text = await res.text();
    let msg = text;
    try { msg = JSON.parse(text).error?.message ?? text; } catch {}
    throw new Error(`${kind}: HTTP ${res.status}: ${msg}`);
  }
  const reader = res.body.getReader();
  const dec = new TextDecoder();
  let buf = "";
  const t0 = Date.now();
  for (;;) {
    const { value, done } = await reader.read();
    if (done) throw new Error(`${kind}: stream ended without a result`);
    buf += dec.decode(value, { stream: true }).replace(/\r\n/g, "\n");
    let i;
    while ((i = buf.indexOf("\n\n")) >= 0) {
      const block = buf.slice(0, i);
      buf = buf.slice(i + 2);
      const data = block.split("\n").filter((l) => l.startsWith("data:")).map((l) => l.slice(5).trim()).join("");
      if (!data || data === "[DONE]") continue;
      const ev = JSON.parse(data);
      if (ev.type === "progress") {
        const step = ev.total ? `${ev.step}/${ev.total}` : `${ev.step ?? ""}`;
        process.stderr.write(`\r  ${kind} ${ev.stage ?? ""} ${step}  ${((Date.now() - t0) / 1000).toFixed(0)}s   `);
      } else if (ev.type === "error") {
        throw new Error(`${kind}: ${ev.message}`);
      } else if (ev.type === "complete") {
        process.stderr.write("\n");
        return ev;
      }
    }
  }
}

/** The file bytes inside a `complete` event: images carry `data[0].b64_json`, the rest `data`. */
function payload(kind, ev) {
  const b64 = kind === "image" ? ev.data?.[0]?.b64_json : ev.data;
  if (typeof b64 !== "string") throw new Error(`${kind}: the complete event carried no data`);
  return Buffer.from(b64, "base64");
}

async function unload(model) {
  await fetch(BASE + "/v1/unload-model", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ model }),
  }).catch(() => {});
}

// ---------- jobs ----------

// Job fields that steer this script; everything else is forwarded to the endpoint.
const LOCAL = new Set(["type", "id", "file", "model", "tile", "maps", "cutout", "trim", "holes", "tolerance", "band",
  "strength", "roughness", "height_m", "size_m", "max_tris", "image_file", "ref_prompt"]);

const REF_SUFFIX = ", a single object, centered, whole object in frame, three-quarter view, plain white background, soft even studio light";

async function runJob(job, out, used) {
  const kind = job.type;
  if (!ENDPOINT[kind]) throw new Error(`job ${job.id}: unknown type "${kind}" (image, speech, music, sound, 3d)`);
  const model = await pickModel(kind, job.model);
  used.add(model);
  const ext = { image: ".png", speech: ".wav", music: ".wav", sound: ".wav", "3d": ".glb" }[kind];
  const file = job.file || join(out, job.id + ext);
  mkdirSync(dirname(file), { recursive: true });
  const body = { model };
  for (const [k, v] of Object.entries(job)) if (!LOCAL.has(k)) body[k] = v;
  const files = { main: file };
  console.error(`- ${job.id} (${kind}, ${model})`);

  if (kind === "3d") {
    let ref = job.image_file;
    if (!ref) {
      const imageModel = await pickModel("image");
      used.add(imageModel);
      ref = file.replace(/\.glb$/, "") + "_ref.png";
      const ev = await generate("image", { model: imageModel, prompt: (job.ref_prompt || job.prompt) + REF_SUFFIX, seed: job.seed ?? 42 });
      writeFileSync(ref, payload("image", ev));
      files.ref = ref;
    }
    delete body.prompt;
    body.image = readFileSync(ref).toString("base64");
    body.texture ??= true;
    let glb = payload("3d", await generate("3d", body));
    if (job.height_m || job.size_m || job.max_tris) glb = fixGlb(glb, { height: job.height_m, size: job.size_m, maxTris: job.max_tris });
    writeFileSync(file, glb);
    return { model, files };
  }

  const bytes = payload(kind, await generate(kind, body));
  if (kind !== "image") {
    writeFileSync(file, bytes);
    return { model, files };
  }
  let img = decodePng(bytes);
  if (job.tile) img = makeTileable(img, job.band);
  if (job.cutout) img = cutout(img, { tolerance: job.tolerance, trim: job.trim, holes: job.holes });
  writeFileSync(file, encodePng(img));
  if (job.maps) Object.assign(files, writeMaps(img, file, { strength: job.strength, roughness: job.roughness }));
  return { model, files };
}

async function runBatch(specFile, opts) {
  const spec = JSON.parse(readFileSync(specFile, "utf8"));
  const out = resolve(dirname(specFile), spec.out || "assets");
  const manifestFile = join(out, "manifest.json");
  const manifest = existsSync(manifestFile) ? JSON.parse(readFileSync(manifestFile, "utf8")) : { assets: {} };
  manifest.generated_by = "mlx-serve gen.mjs";
  const used = new Set();
  let failed = 0;
  for (const job of spec.jobs || []) {
    if (!job.id || !job.type) throw new Error(`every job needs "id" and "type": ${JSON.stringify(job)}`);
    const model = await pickModel(job.type, job.model);
    const hash = createHash("sha256").update(JSON.stringify({ job, model })).digest("hex").slice(0, 16);
    const prev = manifest.assets[job.id];
    if (!opts.force && prev?.hash === hash && Object.values(prev.files).every((f) => existsSync(join(out, f)))) {
      console.error(`- ${job.id}: unchanged, skipped`);
      continue;
    }
    try {
      const r = await runJob({ ...job, model }, out, used);
      const files = Object.fromEntries(Object.entries(r.files).map(([k, f]) => [k, f.startsWith(out) ? f.slice(out.length + 1) : f]));
      manifest.assets[job.id] = { type: job.type, files, model: r.model, prompt: job.prompt ?? job.input, seed: job.seed, hash };
      writeFileSync(manifestFile, JSON.stringify(manifest, null, 2) + "\n");
    } catch (e) {
      failed++;
      console.error(`  FAILED ${job.id}: ${e.message}`);
    }
  }
  if (!opts.keepLoaded) for (const m of used) await unload(m);
  console.error(`${(spec.jobs || []).length - failed}/${(spec.jobs || []).length} jobs ok; manifest: ${manifestFile}`);
  if (failed) process.exitCode = 1;
}

// ---------- PNG (8-bit, non-interlaced) ----------

const CRC = new Int32Array(256).map((_, n) => {
  let c = n;
  for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
  return c;
});
function crc32(buf) {
  let c = -1;
  for (const b of buf) c = CRC[(c ^ b) & 0xff] ^ (c >>> 8);
  return (c ^ -1) >>> 0;
}

/** RGBA image {w, h, data: Uint8Array}. */
function decodePng(buf) {
  if (buf.readUInt32BE(0) !== 0x89504e47) throw new Error("not a PNG (convert JPEG/WebP to PNG first)");
  let pos = 8, w = 0, h = 0, depth = 0, ctype = 0, interlace = 0, palette = null, trns = null;
  const idat = [];
  while (pos < buf.length) {
    const len = buf.readUInt32BE(pos), type = buf.toString("ascii", pos + 4, pos + 8), d = buf.subarray(pos + 8, pos + 8 + len);
    if (type === "IHDR") { w = d.readUInt32BE(0); h = d.readUInt32BE(4); depth = d[8]; ctype = d[9]; interlace = d[12]; }
    else if (type === "PLTE") palette = d;
    else if (type === "tRNS") trns = d;
    else if (type === "IDAT") idat.push(d);
    else if (type === "IEND") break;
    pos += 12 + len;
  }
  if (depth !== 8 || interlace) throw new Error("only 8-bit, non-interlaced PNGs are supported");
  const ch = { 0: 1, 2: 3, 3: 1, 4: 2, 6: 4 }[ctype];
  const raw = inflateSync(Buffer.concat(idat)), stride = w * ch, px = new Uint8Array(stride * h);
  for (let y = 0; y < h; y++) {
    const f = raw[y * (stride + 1)], src = y * (stride + 1) + 1, row = y * stride, up = row - stride;
    for (let x = 0; x < stride; x++) {
      const a = x >= ch ? px[row + x - ch] : 0, b = y ? px[up + x] : 0, c = x >= ch && y ? px[up + x - ch] : 0;
      let v = raw[src + x];
      if (f === 1) v += a;
      else if (f === 2) v += b;
      else if (f === 3) v += (a + b) >> 1;
      else if (f === 4) { const p = a + b - c, pa = Math.abs(p - a), pb = Math.abs(p - b), pc = Math.abs(p - c); v += pa <= pb && pa <= pc ? a : pb <= pc ? b : c; }
      px[row + x] = v;
    }
  }
  const data = new Uint8Array(w * h * 4);
  for (let i = 0; i < w * h; i++) {
    const s = i * ch;
    if (ctype === 6) data.set(px.subarray(s, s + 4), i * 4);
    else if (ctype === 2) { data.set(px.subarray(s, s + 3), i * 4); data[i * 4 + 3] = 255; }
    else if (ctype === 3) { const p = px[s]; data.set(palette.subarray(p * 3, p * 3 + 3), i * 4); data[i * 4 + 3] = trns && p < trns.length ? trns[p] : 255; }
    else { data[i * 4] = data[i * 4 + 1] = data[i * 4 + 2] = px[s]; data[i * 4 + 3] = ctype === 4 ? px[s + 1] : 255; }
  }
  return { w, h, data };
}

function encodePng({ w, h, data }) {
  const raw = Buffer.alloc((w * 4 + 1) * h);
  for (let y = 0; y < h; y++) raw.set(data.subarray(y * w * 4, (y + 1) * w * 4), y * (w * 4 + 1) + 1);
  const chunk = (type, d) => {
    const head = Buffer.alloc(8);
    head.writeUInt32BE(d.length, 0);
    head.write(type, 4, "ascii");
    const crc = Buffer.alloc(4);
    crc.writeUInt32BE(crc32(Buffer.concat([head.subarray(4), d])), 0);
    return Buffer.concat([head, d, crc]);
  };
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(w, 0);
  ihdr.writeUInt32BE(h, 4);
  ihdr.set([8, 6, 0, 0, 0], 8);
  return Buffer.concat([Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]), chunk("IHDR", ihdr),
    chunk("IDAT", deflateSync(raw, { level: 9 })), chunk("IEND", Buffer.alloc(0))]);
}

// ---------- image post-processing ----------

const smoothstep = (a, b, x) => { const t = Math.min(1, Math.max(0, (x - a) / (b - a))); return t * t * (3 - 2 * t); };
const luma = ({ w, h, data }) => Float32Array.from({ length: w * h }, (_, i) => (0.299 * data[i * 4] + 0.587 * data[i * 4 + 1] + 0.114 * data[i * 4 + 2]) / 255);

/** Separable box blur with wrap-around edges, run twice (close to a Gaussian). */
function blur(src, w, h, r) {
  let a = src, b = new Float32Array(src.length);
  for (let pass = 0; pass < 4; pass++) {
    const horiz = pass % 2 === 0;
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      let s = 0;
      for (let k = -r; k <= r; k++) s += horiz ? a[y * w + ((x + k + w) % w)] : a[((y + k + h) % h) * w + x];
      b[y * w + x] = s / (2 * r + 1);
    }
    [a, b] = [b, new Float32Array(src.length)];
  }
  return a;
}

/** Normal (OpenGL/three.js convention, green up), roughness and ambient-occlusion maps from an albedo. */
function writeMaps(img, file, { strength = 2, roughness = 0.75 } = {}) {
  const { w, h } = img, L = luma(img), H = blur(L, w, h, 1), cav = blur(H, w, h, Math.max(2, Math.round(w / 128)));
  let mean = 0;
  for (const v of L) mean += v;
  mean /= L.length;
  let sd = 0;
  for (const v of L) sd += (v - mean) ** 2;
  sd = Math.max(0.05, Math.sqrt(sd / L.length));
  const at = (x, y) => H[((y + h) % h) * w + ((x + w) % w)];
  const nrm = new Uint8Array(w * h * 4), rough = new Uint8Array(w * h * 4), ao = new Uint8Array(w * h * 4);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const gx = at(x + 1, y - 1) + 2 * at(x + 1, y) + at(x + 1, y + 1) - at(x - 1, y - 1) - 2 * at(x - 1, y) - at(x - 1, y + 1);
    const gy = at(x - 1, y + 1) + 2 * at(x, y + 1) + at(x + 1, y + 1) - at(x - 1, y - 1) - 2 * at(x, y - 1) - at(x + 1, y - 1);
    let nx = -gx * strength, ny = gy * strength, nz = 1;
    const len = Math.hypot(nx, ny, nz);
    const i = y * w + x, o = i * 4;
    nrm.set([(nx / len) * 127.5 + 127.5, (ny / len) * 127.5 + 127.5, (nz / len) * 127.5 + 127.5, 255], o);
    const r = Math.min(1, Math.max(0.05, roughness + 0.15 * (mean - L[i]) / sd)) * 255;
    rough.set([r, r, r, 255], o);
    const a = Math.min(1, Math.max(0, 1 - 4 * Math.max(0, cav[i] - H[i]))) * 255;
    ao.set([a, a, a, 255], o);
  }
  const stem = file.replace(/\.png$/i, "");
  const files = { normal: stem + "_normal.png", roughness: stem + "_roughness.png", ao: stem + "_ao.png" };
  writeFileSync(files.normal, encodePng({ w, h, data: nrm }));
  writeFileSync(files.roughness, encodePng({ w, h, data: rough }));
  writeFileSync(files.ao, encodePng({ w, h, data: ao }));
  return files;
}

/** Seamless when repeated: blend a half-offset copy in near each edge, one axis at a time,
 *  so each copy's own seam sits where the other is fully weighted. */
function makeTileable({ w, h, data }, band = 0.25) {
  const pass = (src, horiz) => {
    const out = new Uint8Array(src.length);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const t = horiz ? smoothstep(0, band * w, Math.min(x, w - 1 - x)) : smoothstep(0, band * h, Math.min(y, h - 1 - y));
      const sx = horiz ? (x + (w >> 1)) % w : x, sy = horiz ? y : (y + (h >> 1)) % h;
      for (let c = 0; c < 4; c++) out[(y * w + x) * 4 + c] = src[(y * w + x) * 4 + c] * t + src[(sy * w + sx) * 4 + c] * (1 - t);
    }
    return out;
  };
  return { w, h, data: pass(pass(data, true), false) };
}

/** Plain background to alpha: flood-fill from the border over pixels near the border colour
 *  (`holes`: enclosed pockets of it too), soften a two-pixel rim and pull the background colour out of it. */
function cutout({ w, h, data }, { tolerance = 40, trim = false, holes = false } = {}) {
  const border = [];
  for (let x = 0; x < w; x++) border.push(x, (h - 1) * w + x);
  for (let y = 0; y < h; y++) border.push(y * w, y * w + w - 1);
  const bg = [0, 1, 2].map((c) => { const v = border.map((i) => data[i * 4 + c]).sort((a, b) => a - b); return v[v.length >> 1]; });
  const dist = (i) => Math.max(Math.abs(data[i * 4] - bg[0]), Math.abs(data[i * 4 + 1] - bg[1]), Math.abs(data[i * 4 + 2] - bg[2]));
  const isBg = new Uint8Array(w * h), stack = border.filter((i) => dist(i) < tolerance);
  for (const i of stack) isBg[i] = 1;
  while (stack.length) {
    const i = stack.pop(), x = i % w, y = (i / w) | 0;
    for (const j of [x > 0 ? i - 1 : -1, x < w - 1 ? i + 1 : -1, y > 0 ? i - w : -1, y < h - 1 ? i + w : -1]) {
      if (j >= 0 && !isBg[j] && dist(j) < tolerance) { isBg[j] = 1; stack.push(j); }
    }
  }
  // Pockets of background enclosed by the object (between leaves, through a window); off by default
  // because a white object would lose its own white.
  if (holes) for (let i = 0; i < w * h; i++) if (dist(i) < tolerance) isBg[i] = 1;
  const out = new Uint8Array(data);
  const nearBg = (x, y) => { for (let dy = -2; dy <= 2; dy++) for (let dx = -2; dx <= 2; dx++) { const xx = x + dx, yy = y + dy; if (xx >= 0 && yy >= 0 && xx < w && yy < h && isBg[yy * w + xx]) return true; } return false; };
  let x0 = w, y0 = h, x1 = -1, y1 = -1;
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const i = y * w + x, o = i * 4;
    let a = 1;
    if (isBg[i]) a = 0;
    else if (nearBg(x, y)) a = smoothstep(tolerance * 0.5, tolerance * 2, dist(i));
    if (a > 0 && a < 1) for (let c = 0; c < 3; c++) out[o + c] = Math.min(255, Math.max(0, (data[o + c] - bg[c] * (1 - a)) / a));
    out[o + 3] = Math.round(a * data[o + 3]);
    if (out[o + 3] > 8) { x0 = Math.min(x0, x); y0 = Math.min(y0, y); x1 = Math.max(x1, x); y1 = Math.max(y1, y); }
  }
  if (!trim || x1 < 0) return { w, h, data: out };
  x0 = Math.max(0, x0 - 2); y0 = Math.max(0, y0 - 2); x1 = Math.min(w - 1, x1 + 2); y1 = Math.min(h - 1, y1 + 2);
  const tw = x1 - x0 + 1, th = y1 - y0 + 1, cropped = new Uint8Array(tw * th * 4);
  for (let y = 0; y < th; y++) cropped.set(out.subarray(((y + y0) * w + x0) * 4, ((y + y0) * w + x1 + 1) * 4), y * tw * 4);
  return { w: tw, h: th, data: cropped };
}

// ---------- GLB ----------

function parseGlb(buf) {
  if (buf.readUInt32LE(0) !== 0x46546c67) throw new Error("not a GLB file");
  let pos = 12, json = null, bin = Buffer.alloc(0);
  while (pos < buf.length) {
    const len = buf.readUInt32LE(pos), type = buf.readUInt32LE(pos + 4);
    if (type === 0x4e4f534a) json = JSON.parse(buf.toString("utf8", pos + 8, pos + 8 + len));
    else if (type === 0x004e4942) bin = buf.subarray(pos + 8, pos + 8 + len);
    pos += 8 + len;
  }
  return { json, bin };
}

function writeGlb(json, bin) {
  const pad = (b, fill) => Buffer.concat([b, Buffer.alloc((4 - (b.length % 4)) % 4, fill)]);
  const j = pad(Buffer.from(JSON.stringify(json)), 0x20), b = pad(bin, 0);
  const head = Buffer.alloc(12), jh = Buffer.alloc(8), bh = Buffer.alloc(8);
  head.writeUInt32LE(0x46546c67, 0); head.writeUInt32LE(2, 4); head.writeUInt32LE(12 + 8 + j.length + 8 + b.length, 8);
  jh.writeUInt32LE(j.length, 0); jh.writeUInt32LE(0x4e4f534a, 4);
  bh.writeUInt32LE(b.length, 0); bh.writeUInt32LE(0x004e4942, 4);
  return Buffer.concat([head, jh, j, bh, b]);
}

/** An accessor's elements as plain numbers, honouring byteStride. */
function readAccessor(json, bin, index) {
  const acc = json.accessors[index], view = json.bufferViews[acc.bufferView];
  const n = { SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4 }[acc.type];
  const size = { 5126: 4, 5125: 4, 5123: 2, 5121: 1 }[acc.componentType];
  const read = { 5126: "readFloatLE", 5125: "readUInt32LE", 5123: "readUInt16LE", 5121: "readUInt8" }[acc.componentType];
  const stride = view.byteStride || n * size, base = (view.byteOffset || 0) + (acc.byteOffset || 0);
  const out = new Float64Array(acc.count * n);
  for (let i = 0; i < acc.count; i++) for (let c = 0; c < n; c++) out[i * n + c] = bin[read](base + i * stride + c * size);
  return out;
}

function nodeMatrix(node) {
  if (node.matrix) return node.matrix;
  const [x, y, z, w] = node.rotation || [0, 0, 0, 1], [sx, sy, sz] = node.scale || [1, 1, 1], t = node.translation || [0, 0, 0];
  return [(1 - 2 * (y * y + z * z)) * sx, 2 * (x * y + z * w) * sx, 2 * (x * z - y * w) * sx, 0,
    2 * (x * y - z * w) * sy, (1 - 2 * (x * x + z * z)) * sy, 2 * (y * z + x * w) * sy, 0,
    2 * (x * z + y * w) * sz, 2 * (y * z - x * w) * sz, (1 - 2 * (x * x + y * y)) * sz, 0, t[0], t[1], t[2], 1];
}
const mul = (a, b) => Array.from({ length: 16 }, (_, i) => { const c = i >> 2, r = i & 3; let s = 0; for (let k = 0; k < 4; k++) s += a[k * 4 + r] * b[c * 4 + k]; return s; });

/** Scene bounds in world space from each POSITION accessor's min/max. */
function sceneBounds(json, bin) {
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  const visit = (ni, parent) => {
    const node = json.nodes[ni], m = mul(parent, nodeMatrix(node));
    if (node.mesh !== undefined) for (const p of json.meshes[node.mesh].primitives) {
      const acc = json.accessors[p.attributes.POSITION];
      let mn = acc.min, mx = acc.max;
      if (!mn || !mx) {
        const v = readAccessor(json, bin, p.attributes.POSITION);
        mn = [0, 1, 2].map((c) => { let s = Infinity; for (let i = c; i < v.length; i += 3) s = Math.min(s, v[i]); return s; });
        mx = [0, 1, 2].map((c) => { let s = -Infinity; for (let i = c; i < v.length; i += 3) s = Math.max(s, v[i]); return s; });
      }
      for (let k = 0; k < 8; k++) {
        const p3 = [k & 1 ? mx[0] : mn[0], k & 2 ? mx[1] : mn[1], k & 4 ? mx[2] : mn[2]];
        for (let c = 0; c < 3; c++) {
          const v = m[c] * p3[0] + m[4 + c] * p3[1] + m[8 + c] * p3[2] + m[12 + c];
          lo[c] = Math.min(lo[c], v); hi[c] = Math.max(hi[c], v);
        }
      }
    }
    for (const c of node.children || []) visit(c, m);
  };
  const identity = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
  for (const ni of json.scenes[json.scene || 0].nodes) visit(ni, identity);
  return { lo, hi };
}

/** Quadric edge-collapse simplification. Each collapse moves a vertex onto a neighbour, so normals and
 *  UVs stay valid; open borders and UV seams (border edges of the indexed mesh) never move. */
function simplify(P, I, target) {
  const nv = P.length / 3, T = Int32Array.from(I), nt = T.length / 3;
  const Q = new Float64Array(nv * 10);
  for (let t = 0; t < nt; t++) {
    const [a, b, c] = [T[t * 3], T[t * 3 + 1], T[t * 3 + 2]];
    const ux = P[b * 3] - P[a * 3], uy = P[b * 3 + 1] - P[a * 3 + 1], uz = P[b * 3 + 2] - P[a * 3 + 2];
    const vx = P[c * 3] - P[a * 3], vy = P[c * 3 + 1] - P[a * 3 + 1], vz = P[c * 3 + 2] - P[a * 3 + 2];
    let nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
    const len = Math.hypot(nx, ny, nz);
    if (!len) continue;
    nx /= len; ny /= len; nz /= len;
    const d = -(nx * P[a * 3] + ny * P[a * 3 + 1] + nz * P[a * 3 + 2]), area = len / 2;
    const q = [nx * nx, nx * ny, nx * nz, nx * d, ny * ny, ny * nz, ny * d, nz * nz, nz * d, d * d];
    for (const v of [a, b, c]) for (let k = 0; k < 10; k++) Q[v * 10 + k] += q[k] * area;
  }
  const edges = new Map();
  for (let t = 0; t < nt; t++) for (let k = 0; k < 3; k++) {
    const a = T[t * 3 + k], b = T[t * 3 + ((k + 1) % 3)], key = Math.min(a, b) * nv + Math.max(a, b);
    edges.set(key, (edges.get(key) || 0) + 1);
  }
  const locked = new Uint8Array(nv);
  for (const [key, n] of edges) if (n !== 2) { locked[Math.floor(key / nv)] = 1; locked[key % nv] = 1; }
  const tris = Array.from({ length: nv }, () => []);
  for (let t = 0; t < nt; t++) for (let k = 0; k < 3; k++) tris[T[t * 3 + k]].push(t);
  const alive = new Uint8Array(nt).fill(1), gone = new Uint8Array(nv), ver = new Uint32Array(nv);
  const cost = (a, b) => {
    const x = P[b * 3], y = P[b * 3 + 1], z = P[b * 3 + 2], q = (k) => Q[a * 10 + k] + Q[b * 10 + k];
    return q(0) * x * x + 2 * q(1) * x * y + 2 * q(2) * x * z + 2 * q(3) * x + q(4) * y * y + 2 * q(5) * y * z + 2 * q(6) * y + q(7) * z * z + 2 * q(8) * z + q(9);
  };
  const heap = [];
  const push = (a, b) => {
    if (locked[a]) return;
    const e = [cost(a, b), a, b, ver[a], ver[b]];
    let i = heap.push(e) - 1;
    while (i > 0) { const p = (i - 1) >> 1; if (heap[p][0] <= e[0]) break; heap[i] = heap[p]; i = p; }
    heap[i] = e;
  };
  const pop = () => {
    const top = heap[0], last = heap.pop();
    if (heap.length) {
      let i = 0;
      for (;;) {
        const l = 2 * i + 1, r = l + 1;
        let m = i;
        const val = (j) => (j === i ? last[0] : heap[j][0]);
        if (l < heap.length && heap[l][0] < val(m)) m = l;
        if (r < heap.length && heap[r][0] < val(m)) m = r;
        if (m === i) break;
        heap[i] = heap[m]; i = m;
      }
      heap[i] = last;
    }
    return top;
  };
  for (const key of edges.keys()) { const a = Math.floor(key / nv), b = key % nv; push(a, b); push(b, a); }
  const normal = (t, from, to) => {
    const v = [0, 1, 2].map((k) => (T[t * 3 + k] === from ? to : T[t * 3 + k]));
    const ux = P[v[1] * 3] - P[v[0] * 3], uy = P[v[1] * 3 + 1] - P[v[0] * 3 + 1], uz = P[v[1] * 3 + 2] - P[v[0] * 3 + 2];
    const vx = P[v[2] * 3] - P[v[0] * 3], vy = P[v[2] * 3 + 1] - P[v[0] * 3 + 1], vz = P[v[2] * 3 + 2] - P[v[0] * 3 + 2];
    return [uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx];
  };
  const ring = (v) => { const s = new Set(); for (const t of tris[v]) if (alive[t]) for (let k = 0; k < 3; k++) s.add(T[t * 3 + k]); s.delete(v); return s; };
  let live = nt;
  while (live > target && heap.length) {
    const [, a, b, va, vb] = pop();
    if (gone[a] || gone[b] || va !== ver[a] || vb !== ver[b]) continue;
    const ra = ring(a);
    if (!ra.has(b)) continue;
    let common = 0;
    for (const v of ring(b)) if (ra.has(v)) common++;
    if (common > 2) continue; // would pinch the surface
    let flips = false;
    for (const t of tris[a]) {
      if (!alive[t] || T[t * 3] === b || T[t * 3 + 1] === b || T[t * 3 + 2] === b) continue;
      const n0 = normal(t, -1, -1), n1 = normal(t, a, b);
      if (n0[0] * n1[0] + n0[1] * n1[1] + n0[2] * n1[2] <= 0.2 * Math.hypot(...n0) * Math.hypot(...n1)) { flips = true; break; }
    }
    if (flips) continue;
    for (const t of tris[a]) {
      if (!alive[t]) continue;
      if (T[t * 3] === b || T[t * 3 + 1] === b || T[t * 3 + 2] === b) { alive[t] = 0; live--; continue; }
      for (let k = 0; k < 3; k++) if (T[t * 3 + k] === a) T[t * 3 + k] = b;
      tris[b].push(t);
    }
    gone[a] = 1;
    tris[a] = [];
    tris[b] = tris[b].filter((t) => alive[t]);
    for (let k = 0; k < 10; k++) Q[b * 10 + k] += Q[a * 10 + k];
    ver[b]++;
    for (const v of ring(b)) { push(b, v); push(v, b); }
  }
  const out = [];
  for (let t = 0; t < nt; t++) if (alive[t]) out.push(T[t * 3], T[t * 3 + 1], T[t * 3 + 2]);
  return Uint32Array.from(out);
}

/** Scale to metres with the pivot at the base centre, and optionally cut the triangle count. */
function fixGlb(buf, { height, size, maxTris } = {}) {
  const { json, bin } = parseGlb(buf);
  if ((json.extensionsUsed || []).some((e) => /draco|meshopt/i.test(e))) throw new Error("compressed GLB (Draco/meshopt): decompress it first");
  let newBin = bin;
  if (maxTris) {
    const prims = json.meshes.flatMap((m) => m.primitives).filter((p) => (p.mode ?? 4) === 4 && p.indices !== undefined && p.attributes.POSITION !== undefined);
    const total = prims.reduce((s, p) => s + json.accessors[p.indices].count / 3, 0);
    const parts = [bin];
    let offset = bin.length;
    for (const p of prims) {
      const n = json.accessors[p.indices].count / 3, goal = Math.max(4, Math.round((maxTris * n) / total));
      if (n <= goal) continue;
      const idx = simplify(readAccessor(json, bin, p.attributes.POSITION), readAccessor(json, bin, p.indices), goal);
      const pad = (4 - (offset % 4)) % 4, data = Buffer.from(idx.buffer, idx.byteOffset, idx.byteLength);
      parts.push(Buffer.alloc(pad), data);
      offset += pad;
      json.bufferViews.push({ buffer: 0, byteOffset: offset, byteLength: data.length, target: 34963 });
      json.accessors.push({ bufferView: json.bufferViews.length - 1, componentType: 5125, count: idx.length, type: "SCALAR" });
      p.indices = json.accessors.length - 1;
      offset += data.length;
      console.error(`  ${n} -> ${idx.length / 3} triangles${idx.length / 3 > goal * 1.05 ? " (UV seams and open borders stay fixed, which limits how far this mesh can go)" : ""}`);
    }
    newBin = Buffer.concat(parts);
    json.buffers[0].byteLength = newBin.length;
  }
  if (height || size) {
    const { lo, hi } = sceneBounds(json, newBin);
    const s = height ? height / (hi[1] - lo[1]) : size / Math.max(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]);
    const scene = json.scenes[json.scene || 0];
    json.nodes.push({ name: "normalized", scale: [s, s, s], translation: [-((lo[0] + hi[0]) / 2) * s, -lo[1] * s, -((lo[2] + hi[2]) / 2) * s], children: scene.nodes });
    scene.nodes = [json.nodes.length - 1];
    console.error(`  scaled x${s.toPrecision(3)} to ${((hi[0] - lo[0]) * s).toFixed(2)} x ${((hi[1] - lo[1]) * s).toFixed(2)} x ${((hi[2] - lo[2]) * s).toFixed(2)} m, pivot at the base centre`);
  }
  return writeGlb(json, newBin);
}

// ---------- CLI ----------

const BOOL = new Set(["tile", "maps", "cutout", "trim", "holes", "force", "keep-loaded"]);
function parseArgs(argv) {
  const pos = [], opt = {};
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === "-o") opt.out = argv[++i];
    else if (a.startsWith("--")) {
      const k = a.slice(2);
      opt[k] = BOOL.has(k) ? true : argv[++i];
    } else pos.push(a);
  }
  return { pos, opt };
}
const num = (v) => (v === undefined ? undefined : Number(v));

async function main() {
  const [cmd, ...rest] = process.argv.slice(2);
  const { pos, opt } = parseArgs(rest);
  const need = (what) => { if (!pos[0]) throw new Error(`missing ${what}; node gen.mjs help`); return pos[0]; };
  const outFile = (ext) => opt.out || basename(pos[0] || "out", extname(pos[0] || "")) + ext;
  switch (cmd) {
    case "models":
      for (const m of await listModels()) console.log(`${m.id}\t${(m.capabilities || []).join(",")}\t${m.state}`);
      return;
    case "batch":
      return runBatch(need("a batch JSON file"), { force: !!opt.force, keepLoaded: !!opt["keep-loaded"] });
    case "image": case "speech": case "music": case "sound": case "3d": {
      const job = { type: cmd, id: basename(opt.out || cmd, extname(opt.out || "")), file: opt.out, model: opt.model, seed: num(opt.seed) };
      if (cmd === "3d" && existsSync(need("a prompt or image")) && /\.(png|jpe?g|webp)$/i.test(pos[0])) job.image_file = pos[0];
      else if (cmd === "speech") job.input = need("the text");
      else job.prompt = need("a prompt");
      if (opt.size && cmd === "image") job.size = opt.size;
      if (opt.voice) job.voice = opt.voice;
      if (opt.duration) job.duration_seconds = num(opt.duration);
      if (opt.lyrics) job.lyrics = existsSync(opt.lyrics) ? readFileSync(opt.lyrics, "utf8") : opt.lyrics;
      Object.assign(job, { tile: opt.tile, maps: opt.maps, cutout: opt.cutout, trim: opt.trim, holes: opt.holes, height_m: num(opt.height), size_m: num(opt.size_m), max_tris: num(opt["max-tris"]) });
      job.file ||= { image: "image.png", "3d": "model.glb" }[cmd] || cmd + ".wav";
      for (const k of Object.keys(job)) if (job[k] === undefined) delete job[k];
      const r = await runJob(job, ".", new Set());
      console.log(Object.values(r.files).join("\n"));
      return;
    }
    case "maps": {
      const f = need("a PNG");
      console.log(Object.values(writeMaps(decodePng(readFileSync(f)), f, { strength: num(opt.strength), roughness: num(opt.roughness) })).join("\n"));
      return;
    }
    case "tile":
      writeFileSync(outFile("_tile.png"), encodePng(makeTileable(decodePng(readFileSync(need("a PNG"))), num(opt.band))));
      return console.log(outFile("_tile.png"));
    case "cutout":
      writeFileSync(outFile("_cutout.png"), encodePng(cutout(decodePng(readFileSync(need("a PNG"))), { tolerance: num(opt.tolerance), trim: !!opt.trim, holes: !!opt.holes })));
      return console.log(outFile("_cutout.png"));
    case "glb":
      writeFileSync(outFile("_fixed.glb"), fixGlb(readFileSync(need("a GLB")), { height: num(opt.height), size: num(opt.size), maxTris: num(opt["max-tris"]) }));
      return console.log(outFile("_fixed.glb"));
    default:
      console.log(HELP);
  }
}

main().catch((e) => { console.error(`error: ${e.message}`); process.exitCode = 1; });
