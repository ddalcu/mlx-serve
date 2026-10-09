import { describe, expect, it } from "vitest";
import type { Client } from "../src/lib/core/client";
import type { generateImage } from "../src/lib/core/images";
import type { generateVideo } from "../src/lib/core/video";
import { ImageController } from "../src/lib/state/image-state.svelte";
import { VideoController } from "../src/lib/state/video-state.svelte";
import { MemoryLibrary } from "./support/memory-library";

const client = {} as Client;
const png = new Blob(["png"], { type: "image/png" });
const body = { model: "flux", prompt: "a fox", steps: 4 };
const tick = () => new Promise((resolve) => setTimeout(resolve, 0));

/** An image request the test finishes by hand. */
function pending() {
  let finish!: (value: { blob: Blob; elapsedMs: number; wireBytes: number }) => void;
  let reject!: (error: Error) => void;
  let progress!: (e: Record<string, unknown>) => void;
  const request = ((_c: Client, _b: unknown, options: { onProgress: typeof progress }) => {
    progress = options.onProgress;
    return new Promise((resolve, fail) => ((finish = resolve), (reject = fail)));
  }) as unknown as typeof generateImage;
  return { request, finish: (v = { blob: png, elapsedMs: 1500, wireBytes: 3 }) => finish(v), reject: (e: Error) => reject(e), progress: (e: Record<string, unknown>) => progress(e) };
}

describe("MediaRun through ImageController", () => {
  it("reports progress, completes, and saves the result to the library with its id", async () => {
    const library = new MemoryLibrary();
    const job = pending();
    const c = new ImageController(library, job.request);
    const done = c.generate(client, body, "http://host:1");
    expect([c.phase, c.total, c.message]).toEqual(["running", 4, "Loading model…"]);
    job.progress({ step: 2, total: 4, stage: "denoise" });
    expect([c.step, c.total, c.message]).toEqual([2, 4, "denoise…"]);
    job.finish();
    await done;
    expect(c.phase).toBe("completed");
    expect(c.run).toBeNull();
    expect(c.result?.elapsedMs).toBe(1500);
    expect(c.result?.id).toBeTruthy();
    expect(library.items.size).toBe(1);
    expect([...library.items.values()][0]).toMatchObject({ type: "image", model: "flux", prompt: "a fox", server: "http://host:1" });
  });

  it("refuses a second request while one runs, and a cancelled run's late reply is dropped", async () => {
    const library = new MemoryLibrary();
    const job = pending();
    const c = new ImageController(library, job.request);
    const done = c.generate(client, body, "http://host:1");
    await expect(c.generate(client, body, "http://host:1")).rejects.toThrow(/already running/);
    c.cancel();
    expect(c.phase).toBe("idle");
    expect(c.message).toMatch(/Cancelled/);
    job.finish();
    await done;
    expect(c.result).toBeNull();
    expect(library.items.size).toBe(0);
  });

  it("a failed request leaves the error as the message, and a failed save keeps the result for retry", async () => {
    const failing = pending();
    const c = new ImageController(new MemoryLibrary(), failing.request);
    const done = c.generate(client, body, "http://host:1");
    failing.reject(new Error("model exploded"));
    await done;
    expect([c.phase, c.message]).toEqual(["failed", "model exploded"]);

    const library = new MemoryLibrary();
    library.add = async () => {
      throw new Error("quota");
    };
    const job = pending();
    const d = new ImageController(library, job.request);
    const run = d.generate(client, body, "http://host:1");
    job.finish();
    await run;
    expect(d.phase).toBe("completed");
    expect(d.saveError).toContain("quota");
    expect(d.result?.id).toBeUndefined();
    library.add = async () => "saved-id";
    await d.save();
    expect(d.saveError).toBe("");
    expect(d.result?.id).toBe("saved-id");
  });
});

describe("VideoController", () => {
  const raw = { rgb: new Uint8Array(3), width: 1, height: 1, frames: 1, fps: 24, durationSeconds: 1 };

  it("goes through encoding, shows previews while generating, and saves the encoded clip", async () => {
    const library = new MemoryLibrary();
    let onProgress!: (e: Record<string, unknown>) => void;
    const request = (async (_c: Client, _r: unknown, options: { onProgress: typeof onProgress }) => {
      onProgress = options.onProgress;
      await tick();
      onProgress({ step: 1, total: 2, stage: "x", preview: "data:image/jpeg;base64,AA==" });
      await tick();
      return { raw, elapsedMs: 1000 };
    }) as unknown as typeof generateVideo;
    const phases: string[] = [];
    let seenPreview: unknown;
    const c = new VideoController(library, request, (async () => {
      phases.push(c.phase);
      seenPreview = c.preview;
      return { blob: new Blob(["mp4"], { type: "video/mp4" }), encodeMs: 250, container: "mp4", codec: "avc1" };
    }) as never);
    await c.generate(client, { model: "ltx", prompt: "p", steps: 2 } as never, "http://host:1");
    expect(phases).toEqual(["encoding"]);
    expect(seenPreview).toBeNull();
    expect(c.phase).toBe("completed");
    expect(c.result).toMatchObject({ type: "video", elapsedMs: 1250, codec: "avc1" });
    expect(library.items.size).toBe(1);
  });

  it("cancelling reports the cancelled phase and drops the preview", async () => {
    const c = new VideoController(new MemoryLibrary(), (() => new Promise(() => {})) as never);
    void c.generate(client, { model: "ltx", prompt: "p", steps: 2 } as never, "http://host:1");
    c.preview = { preview: "x" };
    c.cancel();
    expect(c.phase).toBe("cancelled");
    expect(c.preview).toBeNull();
  });
});
