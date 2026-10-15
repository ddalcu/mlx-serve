import { flushSync } from "svelte";
import { afterEach, describe, expect, it, vi } from "vitest";
import { chatModel, imageModel, mockApi, speechModel, videoModel } from "./support/mock-api";
import { mountApp } from "./support/mount";

let ui: Awaited<ReturnType<typeof mountApp>> | undefined;
afterEach(() => {
  ui?.unmount();
  vi.unstubAllGlobals();
});

/** A fetch that holds one route until released. */
function holdRoute(api: ReturnType<typeof mockApi>, path: string) {
  let release!: () => void;
  const gate = new Promise<void>((r) => (release = r));
  vi.stubGlobal("fetch", async (input: string | URL | Request, init?: RequestInit) => {
    if (new URL(String(input)).pathname === path) await gate;
    return api.fetch(input, init);
  });
  return release;
}
/** Leave the pane for the chat and come back. */
async function roundTrip(view: "image" | "audio" | "video") {
  ui!.app.go("chat");
  flushSync();
  await vi.waitFor(() => expect(ui!.q(".chat-screen")).not.toBeNull());
  ui!.app.go(view);
  flushSync();
}
const saved = (type: string) => [...ui!.library.items.values()].filter((i) => i.type === type);

describe("a media generation outlives its pane", () => {
  it("image: leaving for the chat keeps the run, and the picture lands when it finishes", async () => {
    const api = mockApi([imageModel, chatModel]);
    ui = await mountApp({ hash: "#image", api });
    await vi.waitFor(() => expect(ui!.app.image.ready && ui!.app.image.d.model !== "").toBe(true));
    const release = holdRoute(api, "/v1/images/generations");
    await ui.type("#image-prompt", "a red fox");
    await ui.click("#image-generate");
    await roundTrip("image");
    expect(ui.app.image.c.run).not.toBeNull();
    expect(ui.q("#image-generate")!.textContent).toContain("Cancel");
    release();
    await vi.waitFor(() => expect(ui!.q("#image-result")).not.toBeNull());
    await vi.waitFor(() => expect(saved("image")).toHaveLength(1));
  });

  it("audio: leaving for the chat keeps the run, and the clip lands when it finishes", async () => {
    const api = mockApi([speechModel, chatModel]);
    ui = await mountApp({ hash: "#audio", api });
    await vi.waitFor(() => expect(ui!.app.audio.ready && ui!.app.audio.d.model !== "").toBe(true));
    const release = holdRoute(api, "/v1/audio/speech");
    await ui.type("#audio-prompt", "Hello there");
    await ui.click("#audio-generate");
    await roundTrip("audio");
    expect(ui.app.audio.c.run).not.toBeNull();
    release();
    await vi.waitFor(() => expect(ui!.q("#audio-result")).not.toBeNull());
    await vi.waitFor(() => expect(saved("speech")).toHaveLength(1));
  });

  it("video: leaving for the chat keeps the run, and the clip lands when it finishes", async () => {
    const api = mockApi([videoModel, chatModel]);
    ui = await mountApp({ hash: "#video", api });
    await vi.waitFor(() => expect(ui!.app.video.ready && ui!.app.video.d.model !== "").toBe(true));
    let release!: () => void;
    const gate = new Promise<void>((r) => (release = r));
    const c = ui.app.video.c;
    c.request = (async () => {
      await gate;
      return { raw: {}, elapsedMs: 1 };
    }) as never;
    c.encode = (async () => ({ blob: new Blob(["mp4"], { type: "video/mp4" }), encodeMs: 1, codec: "avc1.42E01E" })) as never;
    await ui.type("#video-prompt", "A fox runs through snow");
    await ui.click("#video-generate");
    await roundTrip("video");
    expect(ui.app.video.c.run).not.toBeNull();
    await vi.waitFor(() => expect(ui!.q(".video-progress")).not.toBeNull());
    release();
    await vi.waitFor(() => expect(ui!.q("#video-preview video")).not.toBeNull());
    await vi.waitFor(() => expect(saved("video")).toHaveLength(1));
  });
});

describe("leaving the page during a media generation", () => {
  it("asks the browser to confirm while a run is in progress, and not otherwise", async () => {
    const api = mockApi([imageModel, chatModel]);
    ui = await mountApp({ hash: "#image", api });
    await vi.waitFor(() => expect(ui!.app.image.ready && ui!.app.image.d.model !== "").toBe(true));
    const unload = () => {
      const event = new Event("beforeunload", { cancelable: true });
      window.dispatchEvent(event);
      return event.defaultPrevented;
    };
    expect(unload()).toBe(false);
    const release = holdRoute(api, "/v1/images/generations");
    await ui.type("#image-prompt", "a red fox");
    await ui.click("#image-generate");
    expect(unload()).toBe(true);
    release();
    await vi.waitFor(() => expect(ui!.app.image.c.run).toBeNull());
    expect(unload()).toBe(false);
  });
});
