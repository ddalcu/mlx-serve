import { t, N } from "../i18n/i18n";
import { generateVideo } from "../core/video";
import { encodeVideo, lastFramePng, openShotEncoder } from "../core/encode-video";
import { shotRequest, type Shot } from "../core/storyboard";
import type { Model } from "../core/models";
import type { VideoRequest } from "../core/video";
import type { LibraryInput } from "../core/library";
import type { Client } from "../core/client";
import type { Library } from "../core/library";
import { MediaRun } from "./media-run.svelte";
const qualities = [
  N("Fast"),
  N("Good"),
  N("Quality"),
  N("Super Quality"),
];
/** Width, height, shape and what the canvas costs: the app's own picker rows (MediaGen.swift). */
type SizeRow = [number, number, string, string?];
const ltxSizes: SizeRow[] = [
  [704, 448, N("landscape 14:9"), N("fastest")],
  [448, 704, N("portrait 9:14")],
  [768, 512, N("landscape 3:2")],
  [512, 768, N("portrait 2:3")],
  [1024, 576, N("landscape 16:9")],
  [576, 1024, N("portrait 9:16")],
  [1600, 896, N("landscape 16:9"), N("recommended")],
  [896, 1600, N("portrait 9:16")],
  [1920, 1088, N("landscape 16:9"), N("LTX's own canvas, slowest")],
  [1088, 1920, N("portrait 9:16"), N("slowest")],
];
// Speeds relative to 960x544: the DiT attends over one packed sequence, so cost grows faster than pixels.
const h3Sizes: SizeRow[] = [
  [1344, 768, N("16:9 widescreen"), N("most detail, 2.9x slower")],
  [960, 544, N("16:9 widescreen"), N("fastest, best for long clips")],
  [768, 768, N("square"), N("1.2x slower")],
  [1024, 768, N("4:3 landscape"), N("1.8x slower")],
  [768, 1024, N("3:4 portrait"), N("1.8x slower")],
  [544, 960, N("9:16 portrait"), N("fastest, best for long clips")],
  [768, 1344, N("9:16 portrait"), N("2.9x slower")],
  [1536, 672, N("21:9 cinematic"), N("2.9x slower")],
];
const sizeLabel = ([w, h, shape, note]: SizeRow) => `${w} × ${h} (${t(shape)})` + (note ? ` — ${t(note)}` : "");
function videoProfile(model: Model | undefined) {
  if (!model?.capabilities.includes("video")) return;
  const h3 = model.architecture === "minimax_h3",
    ltx = model.architecture === "AudioVideo";
  if (!h3 && !ltx) return;
  const references =
    h3 && model.id === "ddalcu/MiniMax-H3-REF2VA-MLX-Serve-8bit";
  return {
    h3,
    references,
    last: !references,
    audio: ltx,
    turbo: h3 && !references,
    chain: h3 && !references,
    decoder: model.id === "ddalcu/LTX-2.5-MLX-Serve-8bit",
    sizes: h3 ? h3Sizes : ltxSizes,
    minFrames: h3 ? 5 : 9,
    /** MiniMax's stated 4-second minimum (the lowest rung at or above it); the frame advice and a storyboard shot's floor. */
    statedFrames: h3 ? 107 : 0,
    maxFrames: h3 ? 362 : 193,
    frameStep: h3 ? 17 : 8,
  };
}
function videoQuality(model: Model | undefined, quality: string) {
  const index = Math.max(0, qualities.indexOf(quality)),
    h3 = videoProfile(model)?.h3;
  return {
    quality,
    mode: h3
      ? "one_stage"
      : ["one_stage", "one_stage", "two_stage", "two_stage_hq"][index],
    steps: (h3 ? [16, 30, 30, 50] : [8, 8, 30, 15])[index],
    cfg: index > 1 && !h3 ? 3 : 1,
    stg: index === 2 && !h3 ? 1 : 0,
    frames: h3 ? (index > 1 ? 209 : 124) : index ? 97 : 49,
    turbo: false,
  };
}
function videoDefaults(model?: Model | undefined) {
  const p = videoProfile(model),
    size = p?.sizes[p.decoder ? 2 : 0] || ltxSizes[0];
  return {
    model: model?.id || "",
    prompt: "",
    width: size[0],
    height: size[1],
    ...videoQuality(model, t("Good")),
    refine: 0,
    audioGuidance: 7,
    windows: 1,
    seed: "42",
    preview: false,
    best: false,
    decoder: p?.decoder || false,
    refSize: "match",
    advanced: false,
    media: true,
    speech: "",
    promptHeight: 110,
    storyboard: false,
    shots: [] as Shot[],
  };
}
type VideoDraft = ReturnType<typeof videoDefaults>;
type VideoInput = { base64: string; name?: string; width?: number; height?: number; duration?: number; };
export type VideoInputs = { first?: VideoInput; last?: VideoInput; audio?: VideoInput; images?: VideoInput[]; audios?: VideoInput[]; videos?: { name?: string; frames: string[]; audio?: string; }[]; };
function videoSize(m: Model | undefined, d: VideoDraft, inputs: VideoInputs = {}) {
  const p = videoProfile(m),
    audio = !!(p?.audio && inputs.audio),
    two = !!(p?.audio && (d.mode !== "one_stage" || audio)),
    grid = two ? 64 : 32;
  const max = p?.h3 ? 1536 : 1920;
  if (
    ![d.width, d.height].every(
      (n) => Number.isInteger(n) && n >= 256 && n <= max,
    )
  )
    throw Error(
      t("Clip size must be whole numbers between 256 and %@ px.", [max]),
    );
  return {
    width: Math.round(d.width / grid) * grid,
    height: Math.round(d.height / grid) * grid,
  };
}
/** Turbo turns the fast recipe off, so past the distillation's trained 4–8 a Turbo step costs a full one. */
function maxSteps(m: Model | undefined, d: VideoDraft) {
  const p = videoProfile(m);
  return p?.turbo && d.turbo ? 8 : 50;
}
/** Advice under Steps: H3 below 16 steps needs a few-step adapter, which Turbo is. */
function stepsAdvice(m: Model | undefined, d: VideoDraft) {
  return videoProfile(m)?.h3 && d.steps < 16 && !d.turbo
    ? t("Under 16 steps this model needs a distilled few-step adapter, like the Turbo LoRA. Without one the picture is rough and the soundtrack usually comes out garbled.")
    : "";
}
function frameOptions(m: Model | undefined, d: VideoDraft, inputs: VideoInputs = {}) {
  const p = videoProfile(m);
  if (!p) return [];
  let size;
  try {
    size = videoSize(m, d, inputs);
  } catch {
    return [];
  }
  const windows = p.chain ? d.windows : 1,
    out = [];
  for (let f = p.minFrames; f <= p.maxFrames; f += p.frameStep)
    if (
      (f * windows - (windows - 1)) * size.width * size.height * 3 <=
      256 * 1024 * 1024
    )
      out.push(f);
  return out;
}
function range(value: number | string, min: number, max: number, label: string) {
  const n = Number(value);
  if (!Number.isFinite(n) || n < min || n > max)
    throw Error(t("%@ must be %@–%@.", [label, min, max]));
  return n;
}
function buildVideoRequest(m: Model | undefined, d: VideoDraft, inputs: VideoInputs = {}): VideoRequest {
  const p = videoProfile(m);
  if (!p || !m)
    throw Error(t("Choose a supported video model on this server."));
  if (!d.prompt.trim()) throw Error(t("Enter a prompt."));
  const size = videoSize(m, d, inputs),
    frames = range(d.frames, p.minFrames, p.maxFrames, t("Frames"));
  if ((frames - p.minFrames) % p.frameStep)
    throw Error(t("Choose a valid frame count."));
  const windows = p.chain ? range(d.windows, 1, 6, t("Chained windows")) : 1;
  if (!Number.isInteger(windows))
    throw Error(t("Chained windows must be whole."));
  if (
    (frames * windows - (windows - 1)) * size.width * size.height * 3 >
    256 * 1024 * 1024
  )
    throw Error(
      t(
        "This clip exceeds the browser’s 256 MiB raw-frame limit. Shorten it, drop a window, or choose a smaller canvas.",
      ),
    );
  const seed = range(d.seed, 0, Number.MAX_SAFE_INTEGER, t("Seed")),
    steps = range(d.steps, 4, maxSteps(m, d), t("Steps"));
  if (!Number.isSafeInteger(seed) || !Number.isInteger(steps))
    throw Error(t("Seed and steps must be whole numbers."));
  const body = {
    model: m.id,
    prompt: d.prompt.trim(),
    ...size,
    num_frames: frames,
    steps,
    seed,
    preview: d.preview,
  };
   const b: VideoRequest = body;
  if (d.preview) Object.assign(b, { preview_frames: 1, preview_max_side: 256 });
  const audio = p.audio ? inputs.audio : undefined,
    upgrade = !!audio && d.mode === "one_stage";
  if (p.audio) {
    if (!["one_stage", "two_stage", "two_stage_hq"].includes(d.mode))
      throw Error(t("Choose a valid pipeline mode."));
    b.pipeline = upgrade ? "two_stage" : d.mode;
    if (!upgrade) {
      b.cfg_scale = range(d.cfg, 1, 10, "CFG");
      b.stg_scale = range(d.stg, 0, 4, "STG");
      if (audio)
        b.cfg_audio_scale = range(d.audioGuidance, 1, 12, t("Audio guidance"));
    }
    if (b.pipeline !== "one_stage" && d.refine > 0)
      b.stage2_steps = range(d.refine, 1, 6, t("Refine steps"));
    if (audio) b.audio = audio.base64;
  }
  if (inputs.first) b.first_frame_image = inputs.first.base64;
  if (p.last && inputs.last) b.last_frame_image = inputs.last.base64;
  if (p.h3 && d.best) b.fast = false;
  if (p.decoder && d.decoder) b.decoder = "diffusion";
  if (p.turbo && d.turbo) b.turbo = true;
  if (windows > 1) b.chain_windows = windows;
  if (p.references) {
    const { images = [], videos = [], audios = [] } = inputs;
    if (
      images.length > 9 ||
      videos.length > 3 ||
      audios.length > 3 ||
      images.length + videos.length + audios.length > 12
    )
      throw Error(t("Reference limit: 9 images, 3 clips, 3 audio, 12 total."));
    if (images.length) b.ref_images = images.map((v) => v.base64);
    if (audios.length) b.ref_audios = audios.map((v) => v.base64);
    if (videos.length)
      b.ref_videos = videos.map((v) => {
        let count = Math.min(v.frames.length, frames);
        count -= (((count - 5) % 17) + 17) % 17;
        if (count < 5)
          throw Error(t("Reference clips need at least 5 frames."));
        return {
          frames: v.frames.slice(0, count),
          ...(v.audio ? { audio: v.audio } : {}),
        };
      });
    if (!["match", "max"].includes(d.refSize))
      throw Error(t("Choose reference detail."));
    if (d.refSize !== "match") b.ref_image_size = d.refSize;
  }
  if (JSON.stringify(b).length > 60 * 1024 * 1024)
    throw Error(
      t(
        "References exceed the 60 MiB upload budget. Remove a reference or use smaller files.",
      ),
    );
  return b;
}
class VideoController extends MediaRun<LibraryInput & { id?: string; elapsedMs: number; codec?: string }> {
  protected readonly busy = t("A video request is already running.");
  protected readonly saveFailed = N("Not saved in this browser: %@. Download the video or retry saving.");
  protected readonly cancelledPhase = "cancelled";
  protected readonly cancelled = t("Generation was cancelled. The server may still be finishing the request.");
  started = $state(0);
  preview = $state<Record<string, unknown> | null>(null);
  request: typeof generateVideo;
  encode: typeof encodeVideo;
  openEncoder = openShotEncoder;
  lastFrame = lastFramePng;

  constructor(library: Pick<Library, "add">, generate: typeof generateVideo = generateVideo, encode: typeof encodeVideo = encodeVideo) {
    super(library);
    this.request = generate;
    this.encode = encode;
  }

  cancel() {
    super.cancel();
    this.preview = null;
  }

  async generate(client: Client, request: VideoRequest, server: string) {
    const run = this.begin(Number(request.steps) || 0);
    this.started = Date.now();
    this.preview = null;
    try {
      const generated = await this.request(client, request, {
        signal: run.signal,
        onProgress: this.progress(run, (e) => {
          if (typeof e.preview === "string") this.preview = e;
        }),
      });
      if (this.run !== run) return;
      this.phase = "encoding";
      this.message = t("Encoding video…");
      this.preview = null;
      const encoded = await this.encode(generated.raw, { signal: run.signal });
      if (this.run !== run || run.signal.aborted) return;
      await this.complete({
        type: "video",
        model: String(request.model),
        prompt: request.prompt,
        server,
        blob: encoded.blob,
        createdAt: Date.now(),
        elapsedMs: generated.elapsedMs + encoded.encodeMs,
        codec: encoded.codec,
      });
    } catch (e) {
      this.fail(run, e);
    } finally {
      this.end(run);
    }
  }

  /**
   * A storyboard: each shot is its own request opening on the last frame of the shot before, encoded onto
   * one timeline as it arrives, so only one shot's raw frames are held at a time.
   */
  async generateStoryboard(client: Client, base: VideoRequest, shots: { prompt: string; frames: number }[], server: string) {
    const run = this.begin(Number(base.steps) || 0);
    this.preview = null;
    let encoder: Awaited<ReturnType<typeof openShotEncoder>> | undefined,
      previous: string | undefined,
      elapsedMs = 0,
      shot = 0;
    try {
      for (; shot < shots.length; shot++) {
        const prefix = t("Shot %@ of %@ · ", [shot + 1, shots.length]);
        this.started = Date.now();
        this.message = prefix + t("Loading model…");
        const request = shotRequest(base, shots[shot]!.prompt, shots[shot]!.frames, shot, shots.length, previous);
        const generated = await this.request(client, request, {
          signal: run.signal,
          onProgress: this.progress(run, (e) => {
            if (typeof e.preview === "string") this.preview = e;
          }, prefix),
        });
        if (this.run !== run) return;
        this.preview = null;
        elapsedMs += generated.elapsedMs;
        if (shot < shots.length - 1) previous = await this.lastFrame(generated.raw);
        encoder ??= await this.openEncoder(generated.raw, { signal: run.signal });
        await encoder.add(generated.raw, shot ? 1 : 0);
        if (this.run !== run) return;
      }
      this.phase = "encoding";
      this.message = t("Joining shots…");
      const encoded = await encoder!.finish();
      if (this.run !== run || run.signal.aborted) return;
      await this.complete({
        type: "video",
        model: String(base.model),
        prompt: base.prompt,
        server,
        blob: encoded.blob,
        createdAt: Date.now(),
        elapsedMs: elapsedMs + encoded.encodeMs,
        codec: encoded.codec,
      });
    } catch (e) {
      encoder?.close();
      const reason = e instanceof Error ? e.message : t("Generation failed.");
      this.fail(run, shot < shots.length ? Error(t("Shot %@ of %@: %@", [shot + 1, shots.length, reason])) : e);
    } finally {
      this.end(run);
    }
  }
}

export { qualities, ltxSizes, h3Sizes, sizeLabel, maxSteps, stepsAdvice, videoProfile, videoQuality, videoDefaults, videoSize, frameOptions, buildVideoRequest, VideoController };
