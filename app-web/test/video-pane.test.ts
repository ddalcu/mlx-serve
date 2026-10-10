import { flushSync } from "svelte";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { VideoRequest } from "../src/lib/core/video";
import { chatModel, mockApi, videoModel } from "./support/mock-api";
import { mountApp } from "./support/mount";

let ui: Awaited<ReturnType<typeof mountApp>> | undefined;
afterEach(() => ui?.unmount());
const text = (el: Element | null | undefined) => el?.textContent?.replace(/\s+/g, " ").trim() ?? "";
const encoded = { blob: new Blob(["mp4"], { type: "video/mp4" }), encodeMs: 40, codec: "avc1.42E01E" };

/** The pane against the mock server, with the generation round-trip and the WebCodecs encode replaced by fakes. */
async function videoPane(generate: (request: VideoRequest) => Promise<unknown> = async () => ({ raw: {}, elapsedMs: 50 })) {
  const api = mockApi([videoModel, chatModel]);
  ui = await mountApp({ hash: "#video", api });
  await vi.waitFor(() => expect(ui!.app.video.ready && ui!.app.video.d.model !== "").toBe(true));
  const sent: VideoRequest[] = [];
  const c = ui.app.video.c;
  c.request = (async (_client: unknown, request: VideoRequest) => {
    sent.push(request);
    return generate(request);
  }) as never;
  c.encode = (async () => encoded) as never;
  flushSync();
  return { ui, api, sent };
}
const generate = () => ui!.q<HTMLButtonElement>("#video-generate")!;
const saved = () => [...ui!.library.items.values()].filter((i) => i.type === "video");

describe("Video pane", () => {
  it("opens on the video model and waits for a prompt", async () => {
    const { ui } = await videoPane();
    expect(text(ui.q(".image-model-card strong"))).toBe("ltx");
    expect(generate().disabled).toBe(true);
    expect(text(ui.q("#video-validation"))).toBe("Enter a prompt.");
    await ui.type("#video-prompt", "A fox runs through snow");
    expect(generate().disabled).toBe(false);
    expect(text(ui.q("#video-validation"))).toBe("");
  });

  it("sends the request the model's settings describe, then shows and saves the clip", async () => {
    const { ui, sent } = await videoPane();
    await ui.type("#video-prompt", "A fox runs through snow");
    await ui.click(generate());
    await vi.waitFor(() => expect(ui.q("#video-preview video")).not.toBeNull());
    expect(sent).toEqual([expect.objectContaining({ model: "m/ltx", prompt: "A fox runs through snow", width: 704, height: 448, num_frames: 97, steps: 8, seed: 42, pipeline: "one_stage" })]);
    expect(ui.q<HTMLVideoElement>("#video-preview video")!.src).toMatch(/^blob:/);
    expect(text(ui.q("#video-actions"))).toContain("Download MP4");
    await vi.waitFor(() => expect(saved().map((i) => [i.model, i.prompt])).toEqual([["m/ltx", "A fox runs through snow"]]));
  });

  it("explains a failure as text, keeping the prompt", async () => {
    const { ui } = await videoPane(async () => {
      throw new Error("decoder <script>x</script> failed");
    });
    await ui.type("#video-prompt", "A fox runs through snow");
    await ui.click(generate());
    await vi.waitFor(() => expect(text(ui.q("#video-preview h2"))).toBe("Generation failed"));
    expect(text(ui.q("#video-preview .empty-state p"))).toContain("decoder <script>x</script> failed");
    expect(ui.q("#video-preview script")).toBeNull();
    expect(ui.q<HTMLTextAreaElement>("#video-prompt")!.value).toBe("A fox runs through snow");
    expect(saved()).toEqual([]);
  });

  it("says what is wrong with the settings instead of sending them", async () => {
    const { ui, sent } = await videoPane();
    await ui.type("#video-prompt", "A fox runs through snow");
    await ui.type("#video-width", "100");
    expect(generate().disabled).toBe(true);
    expect(text(ui.q("#video-validation"))).toBe("Clip size must be whole numbers between 256 and 1920 px.");
    expect(sent).toEqual([]);
  });

  it("shows the live progress while the server works, and Cancel stops it", async () => {
    let release!: () => void;
    const gate = new Promise<void>((r) => (release = r));
    const { ui } = await videoPane(async () => {
      await gate;
      return { raw: {}, elapsedMs: 1 };
    });
    await ui.type("#video-prompt", "A fox runs through snow");
    await ui.click(generate());
    await vi.waitFor(() => expect(ui.q(".video-progress")).not.toBeNull());
    expect(ui.q<HTMLButtonElement>("#video-cancel")!.hidden).toBe(false);
    await ui.click("#video-cancel");
    release();
    await vi.waitFor(() => expect(text(ui.q("#video-preview h2"))).toBe("Cancelled"));
    expect(saved()).toEqual([]);
  });
});
