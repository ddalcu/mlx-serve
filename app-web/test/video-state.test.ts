import { describe, expect, it } from "vitest";
import type { Client } from "../src/lib/core/client";
import type { Model } from "../src/lib/core/models";
import { buildVideoRequest, frameOptions, h3Sizes, VideoController, videoDefaults, videoProfile, videoQuality, videoSize, type VideoInputs } from "../src/lib/state/video-state.svelte";
import { MemoryLibrary } from "./support/memory-library";

const model = (id: string, architecture: string): Model => ({ id, capabilities: ["video"], architecture, meta: {} });
const ltx = model("org/ltx-2.3", "AudioVideo");
const ltx25 = model("ddalcu/LTX-2.5-MLX-Serve-8bit", "AudioVideo");
const h3 = model("org/minimax-h3", "minimax_h3");
const refs = model("ddalcu/MiniMax-H3-REF2VA-MLX-Serve-8bit", "minimax_h3");
const draft = (m: Model, patch: Record<string, unknown> = {}) => ({ ...videoDefaults(m), prompt: "a fox in snow", ...patch });
/** H3's default canvas is too big for long clips under the 256 MiB raw-frame budget. */
const small = { width: 768, height: 768 };
const img = { base64: "IMG" };
const wav = { base64: "WAV" };

describe("videoProfile", () => {
  it("tells LTX from H3 and from the reference-to-video partition", () => {
    expect(videoProfile(ltx)).toMatchObject({ h3: false, audio: true, last: true, turbo: false, chain: false, minFrames: 9, maxFrames: 193, frameStep: 8, decoder: false });
    expect(videoProfile(ltx25)).toMatchObject({ decoder: true });
    expect(videoProfile(h3)).toMatchObject({ h3: true, audio: false, last: true, turbo: true, chain: true, references: false, minFrames: 5, maxFrames: 362, frameStep: 17 });
    expect(videoProfile(refs)).toMatchObject({ h3: true, references: true, last: false, turbo: false, chain: false });
  });

  it("offers nothing for other models", () => {
    expect(videoProfile(undefined)).toBeUndefined();
    expect(videoProfile(model("x", "mystery"))).toBeUndefined();
    expect(videoProfile({ ...ltx, capabilities: ["chat"] })).toBeUndefined();
  });
});

describe("videoQuality", () => {
  it("LTX climbs from one stage to two-stage HQ; H3 stays one-stage", () => {
    expect(videoQuality(ltx, "Fast")).toMatchObject({ mode: "one_stage", steps: 8, frames: 49, cfg: 1 });
    expect(videoQuality(ltx, "Quality")).toMatchObject({ mode: "two_stage", steps: 30, cfg: 3, stg: 1, frames: 97 });
    expect(videoQuality(ltx, "Super Quality")).toMatchObject({ mode: "two_stage_hq", steps: 15, cfg: 3, stg: 0 });
    expect(videoQuality(h3, "Super Quality")).toMatchObject({ mode: "one_stage", steps: 50, cfg: 1, frames: 209 });
  });
});

describe("videoSize", () => {
  it("snaps to the pipeline's grid: 32 for one stage, 64 for two", () => {
    expect(videoSize(ltx, draft(ltx, { width: 700, height: 450 }))).toEqual({ width: 704, height: 448 });
    expect(videoSize(ltx, draft(ltx, { width: 700, height: 450, mode: "two_stage" }))).toEqual({ width: 704, height: 448 });
    expect(videoSize(ltx, draft(ltx, { width: 740, height: 450, mode: "two_stage" }))).toEqual({ width: 768, height: 448 });
    expect(videoSize(ltx, draft(ltx, { width: 740, height: 450 }), { audio: wav })).toEqual({ width: 768, height: 448 });
  });

  it("refuses sizes outside the model's range", () => {
    expect(() => videoSize(ltx, draft(ltx, { width: 100 }))).toThrow(/between 256 and 1920/);
    expect(() => videoSize(h3, draft(h3, { width: 1600 }))).toThrow(/between 256 and 1536/);
    expect(() => videoSize(ltx, draft(ltx, { height: 500.5 }))).toThrow(/whole numbers/);
  });
});

describe("frameOptions", () => {
  it("lists the frame counts the model steps through, within the raw-frame budget", () => {
    const small = frameOptions(ltx, draft(ltx, { width: 256, height: 256 }));
    expect(small[0]).toBe(9);
    expect(small.slice(0, 3)).toEqual([9, 17, 25]);
    const big = frameOptions(ltx, draft(ltx, { width: 1920, height: 1088 }));
    expect(big.length).toBeLessThan(small.length);
    expect(Math.max(...big) * 1920 * 1088 * 3).toBeLessThanOrEqual(256 * 1024 * 1024);
  });

  it("offers none for a bad size or an unknown model", () => {
    expect(frameOptions(ltx, draft(ltx, { width: 10 }))).toEqual([]);
    expect(frameOptions(undefined, draft(ltx))).toEqual([]);
  });
});

describe("buildVideoRequest", () => {
  it("LTX one-stage: the plain request with guidance at its neutral values", () => {
    expect(buildVideoRequest(ltx, draft(ltx))).toEqual({ model: "org/ltx-2.3", prompt: "a fox in snow", width: 704, height: 448, num_frames: 97, steps: 8, seed: 42, preview: false, pipeline: "one_stage", cfg_scale: 1, stg_scale: 0 });
  });

  it("an audio clip upgrades one-stage to two-stage and leaves guidance to the server", () => {
    const body = buildVideoRequest(ltx, draft(ltx), { audio: wav });
    expect(body).toMatchObject({ pipeline: "two_stage", audio: "WAV" });
    expect(body).not.toHaveProperty("cfg_scale");
    const guided = buildVideoRequest(ltx, draft(ltx, { mode: "two_stage", cfg: 3, audioGuidance: 5 }), { audio: wav });
    expect(guided).toMatchObject({ pipeline: "two_stage", cfg_scale: 3, cfg_audio_scale: 5 });
  });

  it("refine steps apply only past one stage", () => {
    expect(buildVideoRequest(ltx, draft(ltx, { mode: "two_stage_hq", refine: 3 }))).toMatchObject({ pipeline: "two_stage_hq", stage2_steps: 3 });
    expect(buildVideoRequest(ltx, draft(ltx, { refine: 3 }))).not.toHaveProperty("stage2_steps");
  });

  it("carries first and last frame images, the 2.5 decoder and previews", () => {
    expect(buildVideoRequest(ltx25, draft(ltx25, { preview: true }), { first: img, last: { base64: "END" } })).toMatchObject({ first_frame_image: "IMG", last_frame_image: "END", decoder: "diffusion", preview: true, preview_frames: 1, preview_max_side: 256 });
  });

  it("H3: chained windows, the slow recipe and Turbo are opt-ins", () => {
    const body = buildVideoRequest(h3, draft(h3, { ...small, windows: 2, best: true, turbo: true, steps: 8, frames: 39 }));
    expect(body).toMatchObject({ chain_windows: 2, fast: false, turbo: true });
    expect(() => buildVideoRequest(h3, draft(h3, { ...small, turbo: true, steps: 30, frames: 39 }))).toThrow(/Steps must be 4–16/);
    expect(body).not.toHaveProperty("pipeline");
    expect(buildVideoRequest(h3, draft(h3, { ...small, frames: 39 }))).not.toHaveProperty("chain_windows");
  });

  it("the reference partition sends its references and takes no last frame", () => {
    const inputs: VideoInputs = { images: [img, img], audios: [wav], videos: [{ frames: Array(22).fill("F"), audio: "A" }], last: img };
    const body = buildVideoRequest(refs, draft(refs, { ...small, frames: 39, refSize: "max" }), inputs);
    expect(body).toMatchObject({ ref_images: ["IMG", "IMG"], ref_audios: ["WAV"], ref_image_size: "max" });
    expect(body).not.toHaveProperty("last_frame_image");
    const clip = (body.ref_videos as { frames: string[]; audio?: string }[])[0]!;
    expect(clip.frames.length).toBe(22);
    expect(clip.audio).toBe("A");
  });

  it("trims a reference clip to a frame count the model accepts, and refuses one that is too short", () => {
    const long = buildVideoRequest(refs, draft(refs, { ...small, frames: 39 }), { videos: [{ frames: Array(30).fill("F") }] });
    expect((long.ref_videos as { frames: string[] }[])[0]!.frames.length).toBe(22);
    expect(() => buildVideoRequest(refs, draft(refs, { ...small, frames: 39 }), { videos: [{ frames: Array(4).fill("F") }] })).toThrow(/at least 5 frames/);
  });

  it("enforces the reference limits", () => {
    expect(() => buildVideoRequest(refs, draft(refs, { ...small, frames: 39 }), { images: Array(10).fill(img) })).toThrow(/Reference limit/);
    expect(() => buildVideoRequest(refs, draft(refs, { ...small, frames: 39 }), { images: Array(9).fill(img), audios: Array(3).fill(wav), videos: [{ frames: Array(5).fill("F") }] })).toThrow(/Reference limit/);
  });

  it("validates prompt, frames, seed, steps and the raw-frame budget", () => {
    expect(() => buildVideoRequest(undefined, draft(ltx))).toThrow(/supported video model/);
    expect(() => buildVideoRequest(ltx, draft(ltx, { prompt: " " }))).toThrow("Enter a prompt.");
    expect(() => buildVideoRequest(ltx, draft(ltx, { frames: 10 }))).toThrow("Choose a valid frame count.");
    expect(() => buildVideoRequest(ltx, draft(ltx, { frames: 500 }))).toThrow(/Frames must be 9–193/);
    expect(() => buildVideoRequest(ltx, draft(ltx, { seed: "-1" }))).toThrow(/Seed must be/);
    expect(() => buildVideoRequest(ltx, draft(ltx, { seed: "1.5" }))).toThrow(/whole numbers/);
    expect(() => buildVideoRequest(ltx, draft(ltx, { steps: 2 }))).toThrow(/Steps must be 4–50/);
    expect(() => buildVideoRequest(ltx, draft(ltx, { width: 1920, height: 1088, frames: 193 }))).toThrow(/256 MiB raw-frame limit/);
    expect(() => buildVideoRequest(h3, draft(h3, { ...small, windows: 7, frames: 22 }))).toThrow(/Chained windows must be 1–6/);
  });

  it("every size the picker offers is a size the model accepts", () => {
    for (const [width, height] of h3Sizes) expect(() => buildVideoRequest(h3, draft(h3, { width, height, frames: 22 }))).not.toThrow();
  });
});

describe("VideoController", () => {
  const raw = { frames: 9 } as never;
  const encoded = { blob: new Blob(["mp4"], { type: "video/mp4" }), encodeMs: 30, codec: "avc1" };
  const request = { model: "m/ltx", prompt: "a fox", steps: 4 } as never;
  const make = (generate: (...a: never[]) => Promise<unknown>, encode: (...a: never[]) => Promise<unknown>) => {
    const library = new MemoryLibrary();
    return { library, c: new VideoController(library.asLibrary, generate as never, encode as never) };
  };

  it("generates, encodes, then saves the finished clip", async () => {
    const phases: string[] = [];
    const { c, library } = make(
      async (_c: never, _r: never, o: { onProgress: (e: object) => void }) => {
        o.onProgress({ step: 2, total: 4, stage: "denoise" });
        phases.push(`${c.phase}:${c.step}/${c.total}`);
        return { raw, elapsedMs: 100 };
      },
      async () => {
        phases.push(c.phase);
        return encoded;
      },
    );
    await c.generate({} as Client, request, "http://s");
    expect(phases).toEqual(["running:2/4", "encoding"]);
    expect(c.phase).toBe("completed");
    expect(c.result).toMatchObject({ type: "video", model: "m/ltx", elapsedMs: 130, codec: "avc1" });
    expect([...library.items.values()].map((i) => [i.type, i.model, i.prompt, i.server])).toEqual([["video", "m/ltx", "a fox", "http://s"]]);
  });

  it("a failed request ends as failed with the reason, saving nothing", async () => {
    const { c, library } = make(async () => { throw new Error("server said no"); }, async () => encoded);
    await c.generate({} as Client, request, "http://s");
    expect(c.phase).toBe("failed");
    expect(c.message).toContain("server said no");
    expect(c.run).toBeNull();
    expect(library.items.size).toBe(0);
  });

  it("a failed encode ends as failed too", async () => {
    const { c, library } = make(async () => ({ raw, elapsedMs: 1 }), async () => { throw new Error("no WebCodecs"); });
    await c.generate({} as Client, request, "http://s");
    expect(c.phase).toBe("failed");
    expect(c.message).toContain("no WebCodecs");
    expect(library.items.size).toBe(0);
  });

  it("cancelling mid-request says the server may still be working, and saves nothing", async () => {
    let release!: () => void;
    const gate = new Promise<void>((r) => (release = r));
    const { c, library } = make(async () => { await gate; return { raw, elapsedMs: 1 }; }, async () => encoded);
    const run = c.generate({} as Client, request, "http://s");
    c.cancel();
    release();
    await run;
    expect(c.phase).toBe("cancelled");
    expect(c.message).toContain("may still be finishing");
    expect(library.items.size).toBe(0);
  });

  it("only one request at a time", async () => {
    let release!: () => void;
    const gate = new Promise<void>((r) => (release = r));
    const { c } = make(async () => { await gate; return { raw, elapsedMs: 1 }; }, async () => encoded);
    const first = c.generate({} as Client, request, "http://s");
    await expect(c.generate({} as Client, request, "http://s")).rejects.toThrow("A video request is already running.");
    expect(c.phase).toBe("running");
    release();
    await first;
    expect(c.phase).toBe("completed");
  });
});
