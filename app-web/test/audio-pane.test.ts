import { flushSync } from "svelte";
import { afterEach, describe, expect, it, vi } from "vitest";
import { AudioWorkspace } from "../src/lib/state/audio-workspace.svelte";
import { chatModel, mockApi, musicModel, soundModel, speechModel } from "./support/mock-api";
import { mountApp } from "./support/mount";

let ui: Awaited<ReturnType<typeof mountApp>> | undefined;
afterEach(() => ui?.unmount());
const text = (el: Element | null | undefined) => el?.textContent?.replace(/\s+/g, " ").trim() ?? "";

async function audioPane() {
  const api = mockApi([speechModel, musicModel, soundModel, chatModel]);
  ui = await mountApp({ hash: "#audio", api });
  await vi.waitFor(() => expect(ui!.app.audio.ready && ui!.app.audio.d.model !== "").toBe(true));
  flushSync();
  return { ui, api };
}
const generate = () => ui!.q<HTMLButtonElement>("#audio-generate")!;
const posts = (api: ReturnType<typeof mockApi>) => api.requests.filter((r) => r.path.startsWith("/v1/audio/"));
const saved = () => [...ui!.library.items.values()].filter((i) => i.type !== "chat");
const tab = async (name: string) => ui!.click(ui!.qa<HTMLButtonElement>(".audio-tabs button").find((b) => text(b) === name)!);

describe("Audio pane", () => {
  it("starts on Voice with the speech model and waits for text", async () => {
    const { ui } = await audioPane();
    expect(ui.qa(".audio-tabs button").map((b) => [text(b), b.getAttribute("aria-pressed")])).toEqual([["Voice", "true"], ["Music", "false"], ["Sound Effects", "false"]]);
    expect(ui.app.audio.d.model).toBe("m/kokoro");
    expect(generate().disabled).toBe(true);
    expect(text(ui.q("#audio-validation"))).toBe("Enter text to be generated.");
    await ui.type("#audio-prompt", "Hello there");
    expect(generate().disabled).toBe(false);
  });

  it("speaks the text, offers the audio and saves it as speech", async () => {
    const { ui, api } = await audioPane();
    await ui.type("#audio-prompt", "Hello there");
    await ui.click(generate());
    await vi.waitFor(() => expect(ui.q("#audio-result")).not.toBeNull());
    expect(posts(api)[0]).toMatchObject({ path: "/v1/audio/speech", body: { model: "m/kokoro", input: "Hello there", voice: "af_heart", speed: 1, stream: true } });
    expect(ui.q<HTMLAudioElement>("#audio-result")!.src).toMatch(/^blob:/);
    expect(text(ui.q(".audio-file"))).toContain("kokoro.wav");
    await vi.waitFor(() => expect(saved().map((i) => [i.type, i.model, i.prompt])).toEqual([["speech", "m/kokoro", "Hello there"]]));
  });

  it("shows a failure as text and keeps the text to retry", async () => {
    const { ui, api } = await audioPane();
    api.failures.push({ status: 500, error: "no <i>voice</i> loaded" });
    await ui.type("#audio-prompt", "Hello there");
    await ui.click(generate());
    await vi.waitFor(() => expect(text(ui.q("#audio-preview h2"))).toBe("Failed"));
    expect(text(ui.q("#audio-preview .empty-state p"))).toContain("no <i>voice</i> loaded");
    expect(ui.q("#audio-preview i")).toBeNull();
    expect(ui.q<HTMLTextAreaElement>("#audio-prompt")!.value).toBe("Hello there");
  });

  it("Music needs lyrics or Instrumental for models that sing, and sends its own route", async () => {
    const { ui, api } = await audioPane();
    await tab("Music");
    expect(ui.app.audio.tab).toBe("music");
    expect(ui.app.audio.d.model).toBe("m/ace");
    await ui.type("#audio-prompt", "upbeat synthwave");
    await ui.type("input[name=seed]", "3");
    expect(generate().disabled).toBe(false);
    await ui.click(generate());
    await vi.waitFor(() => expect(ui.q("#audio-result")).not.toBeNull());
    expect(posts(api)[0]).toMatchObject({ path: "/v1/audio/music-generations", body: { model: "m/ace", prompt: "upbeat synthwave", duration_seconds: 60, seed: 3, response_format: "wav", stream: true } });
    await vi.waitFor(() => expect(saved().map((i) => i.type)).toEqual(["music"]));
  });

  it("Sound Effects sends duration, steps and a seed to its own route", async () => {
    const { ui, api } = await audioPane();
    await tab("Sound Effects");
    expect(ui.app.audio.d.model).toBe("m/sat");
    await ui.type("#audio-prompt", "rain on a window");
    await ui.type("input[name=seed]", "9");
    await ui.click(generate());
    await vi.waitFor(() => expect(ui.q("#audio-result")).not.toBeNull());
    expect(posts(api)[0]).toMatchObject({ path: "/v1/audio/sound-generations", body: { model: "m/sat", prompt: "rain on a window", duration_seconds: 10, steps: 8, seed: 9, stream: true } });
    await vi.waitFor(() => expect(saved().map((i) => i.type)).toEqual(["sound"]));
  });

  it("keeps a separate draft for each tab", async () => {
    const { ui } = await audioPane();
    await ui.type("#audio-prompt", "spoken words");
    await tab("Music");
    expect(ui.q<HTMLTextAreaElement>("#audio-prompt")!.value).toBe("");
    await ui.type("#audio-prompt", "a style");
    await tab("Voice");
    expect(ui.q<HTMLTextAreaElement>("#audio-prompt")!.value).toBe("spoken words");
  });

  it("restores drafts, tab and all, in the next session", async () => {
    const { ui } = await audioPane();
    await ui.type("#audio-prompt", "spoken words");
    await tab("Music");
    await ui.type("#audio-prompt", "a style");
    ui.app.audio.flush();
    const later = new AudioWorkspace(ui.app.connection, ui.library.asLibrary, ui.app.audio.drafts);
    await vi.waitFor(async () => {
      await later.init();
      expect(later.tab).toBe("music");
      expect(later.panes.voice.d.prompt).toBe("spoken words");
      expect(later.panes.music.d.prompt).toBe("a style");
    });
  });
});
