import { flushSync } from "svelte";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { VideoRequest } from "../src/lib/core/video";
import { chatModel, mockApi, otherModel, PNG_1X1, videoModel } from "./support/mock-api";
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

const h3Model = { id: "m/h3", capabilities: ["video"], loaded: false, meta: { architecture: "minimax_h3" } };

/** The pane on an H3 model, with shots generated, framed and encoded by fakes. */
async function h3Pane(models: object[] = [h3Model, chatModel]) {
  const api = mockApi(models);
  ui = await mountApp({ hash: "#video", api });
  await vi.waitFor(() => expect(ui!.app.video.ready && ui!.app.video.d.model === "m/h3").toBe(true));
  const sent: VideoRequest[] = [],
    added: number[] = [];
  const c = ui.app.video.c;
  c.request = (async (_client: unknown, request: VideoRequest) => {
    sent.push(request);
    return { raw: { frames: sent.length }, elapsedMs: 5 };
  }) as never;
  c.lastFrame = async () => "LASTFRAME";
  c.openEncoder = async () => ({
    add: async (_raw, skip) => void added.push(skip),
    finish: async () => ({ ...encoded, container: "mp4", heapBeforeBytes: null, peakHeapBytes: null, workingBufferBytes: 0 }),
    close() {},
  });
  ui.app.video.d.width = 960;
  ui.app.video.d.height = 544;
  flushSync();
  return { ui, api, sent, added };
}
const button = (label: string) => ui!.qa<HTMLButtonElement>("button").find((b) => text(b) === label)!;

describe("Video pane, in step with the app", () => {
  it("the size presets say which canvas is fastest", async () => {
    const { ui } = await h3Pane();
    const rows = ui.qa("#video-presets-menu [role=menuitem]").map(text);
    expect(rows).toContain("960 × 544 (16:9 widescreen) — fastest, best for long clips");
    expect(rows).toContain("1344 × 768 (16:9 widescreen) — most detail, 2.9x slower");
  });

  it("a storyboard runs its shots in order, handing each the last frame of the one before", async () => {
    const { ui, sent, added } = await h3Pane();
    await ui.click(button("Storyboard"));
    expect(text(ui.q("label[for=video-prompt]"))).toBe("Story");
    expect(ui.q("#video-frames")).toBeNull();
    expect(text(ui.q("#video-validation"))).toBe("Add a shot.");
    await ui.click("#video-add-shot");
    await ui.click("#video-add-shot");
    const prompts = ui.qa<HTMLTextAreaElement>(".video-shot textarea");
    expect(prompts.length).toBe(2);
    expect(text(ui.q("#video-validation"))).toBe("Every shot needs a prompt.");
    prompts[0]!.value = "A fox wakes.";
    prompts[0]!.dispatchEvent(new Event("input", { bubbles: true }));
    prompts[1]!.value = "It runs.";
    prompts[1]!.dispatchEvent(new Event("input", { bubbles: true }));
    flushSync();
    expect(text(ui.q("#video-quality-note"))).toMatch(/^1-stage, 30 steps, 2 shots · \d+ s$/);
    await ui.click(generate());
    await vi.waitFor(() => expect(ui.q("#video-preview video")).not.toBeNull());
    expect(sent.map((r) => [r.prompt, r.first_frame_image])).toEqual([["A fox wakes.", undefined], ["It runs.", "LASTFRAME"]]);
    expect(added).toEqual([0, 1]);
    await vi.waitFor(() => expect(saved().length).toBe(1));
  });

  it("Enhance shows the model working, then plans the shots from the story", async () => {
    const { ui, api } = await h3Pane();
    let release!: () => void;
    api.replies.push({ reasoning: "Two beats.", content: "=== SHOT 1 | 10s ===\nA fox wakes.\n=== SHOT 2 | 10s ===\nIt runs.", hold: new Promise<void>((r) => (release = r)) });
    await ui.click(button("Storyboard"));
    await ui.type("#video-prompt", "A fox's morning");
    await ui.click("#video-enhance");
    expect(text(ui.q("#rewrite-clip-note"))).toMatch(/Enhance writes a storyboard: about \d+ shots of up to 6 s each\./);
    await ui.click("#rewrite-run");
    await vi.waitFor(() => expect(text(ui.q("#rewrite-status"))).toMatch(/^Writing… 0:0\d$/));
    release();
    await vi.waitFor(() => expect(ui.q<HTMLButtonElement>("#rewrite-apply")!.disabled).toBe(false));
    const asked = api.requests.find((r) => r.path === "/v1/chat/completions")!.body;
    expect(asked.messages[0].content).toContain("=== SHOT 1 |");
    await ui.click("#rewrite-apply");
    expect(ui.app.video.d.shots.map((s) => s.prompt)).toEqual(["A fox wakes.", "It runs."]);
  });

  it("Enhance shows a vision model the first frame", async () => {
    const { ui, api } = await h3Pane();
    ui.app.video.inputs.first = { base64: PNG_1X1, name: "start.png", width: 1, height: 1 };
    await ui.type("#video-prompt", "They dance");
    await ui.click("#video-enhance");
    await ui.click("#rewrite-run");
    await vi.waitFor(() => expect(ui.q<HTMLButtonElement>("#rewrite-run")!.disabled).toBe(false));
    const seen = api.requests.find((r) => r.path === "/v1/chat/completions")!.body.messages[1].content;
    expect(seen[0].text).toContain("attached picture");
    expect(seen[1].image_url.url).toBe(`data:image/png;base64,${PNG_1X1}`);
    expect(ui.q("#rewrite-blind")).toBeNull();
  });

  it("Enhance says when the chat model cannot see the first frame, and leaves the picture out", async () => {
    const { ui, api } = await h3Pane([h3Model, otherModel]);
    ui.app.video.inputs.first = { base64: PNG_1X1, name: "start.png", width: 1, height: 1 };
    await ui.type("#video-prompt", "They dance");
    await ui.click("#video-enhance");
    await ui.click("#rewrite-run");
    await vi.waitFor(() => expect(ui.q("#rewrite-blind")).not.toBeNull());
    const sent = api.requests.find((r) => r.path === "/v1/chat/completions")!.body.messages[1].content;
    expect(typeof sent).toBe("string");
    expect(sent).not.toContain("attached picture");
  });

  it("an empty reply says so instead of leaving a blank box", async () => {
    const { ui, api } = await h3Pane();
    api.replies.push({ content: "  " });
    await ui.type("#video-prompt", "A fox");
    await ui.click("#video-enhance");
    await ui.click("#rewrite-run");
    await vi.waitFor(() => expect(text(ui.q("#rewrite-status"))).toBe("The model finished without writing anything. Try again."));
  });
});
